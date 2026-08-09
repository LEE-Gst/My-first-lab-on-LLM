# -*- coding: utf-8 -*-
"""Neural DFA Executor：结构化 spec + 显式转移记忆 + 顺序软检索。
架构：
  - 结构化编码下，transition token 的 id 直接携带 (q, sym)，其后紧跟 next 状态 id；
  - 记忆：key = k_proj(state_emb(q)+sym_emb(sym))，value = v_proj(state_emb(next))；
  - 执行：r0 = state_emb(0)，每步 query = q_proj(r + sym_emb(input))，
          软注意力检索 value 更新 r（LayerNorm 透传），状态头解码状态；
  - 训练：状态监督 + 判定监督 + 注意力监督（过程监督：每一步应查哪条转移表条目）。
"""
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsm.data import BYTE_START, encode_structured_dfa_spec, make_balanced_dfa


@dataclass
class ExecutorConfig:
    d_model: int = 256
    max_k: int = 32
    max_trans: int = 64
    attention_weight: float = 0.5


class NeuralDFAExecutor(nn.Module):
    def __init__(self, cfg: ExecutorConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.TRANSITION_OFFSET = 70
        self.state_emb = nn.Embedding(cfg.max_k, d)
        self.sym_emb = nn.Embedding(2, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.q_proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.state_head = nn.Linear(d, cfg.max_k)
        self.verdict_head = nn.Linear(d, 2)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, trans_tokens: torch.Tensor, nxt_tokens: torch.Tensor,
                trans_mask: torch.Tensor, sym_ids: torch.Tensor):
        """trans_tokens (B,nT) transition token ids；nxt_tokens (B,nT) next 状态 id；
        trans_mask (B,nT) 有效位；sym_ids (B,L) 输入符号。
        返回 state_logits (B,L,max_k), verdict_logits (B,2), attn_logits (B,L,nT)。"""
        B, nT = trans_tokens.shape
        L = sym_ids.shape[1]
        dev = trans_tokens.device

        q_idx = ((trans_tokens - self.TRANSITION_OFFSET) // 2).clamp(0, self.cfg.max_k - 1)
        sym_idx = ((trans_tokens - self.TRANSITION_OFFSET) % 2).clamp(0, 1)
        keys = self.k_proj(self.state_emb(q_idx) + self.sym_emb(sym_idx))
        vals = self.v_proj(self.state_emb(nxt_tokens.clamp(0, self.cfg.max_k - 1)))

        r = self.state_emb(torch.zeros(B, dtype=torch.long, device=dev))
        sym_e = self.sym_emb(sym_ids)
        state_logits_list, attn_list = [], []
        for t in range(L):
            q = self.q_proj(r + sym_e[:, t])
            scores = torch.bmm(q.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)  # (B,nT)
            scores = scores.masked_fill(~trans_mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            retr = torch.bmm(attn.unsqueeze(1), vals).squeeze(1)
            r = self.norm(retr)
            state_logits_list.append(self.state_head(r))
            attn_list.append(scores)

        state_logits = torch.stack(state_logits_list, dim=1)
        attn_logits = torch.stack(attn_list, dim=1)
        verdict_logits = self.verdict_head(self.norm(r))
        return state_logits, verdict_logits, attn_logits


# ============================================================
# 批量构造
# ============================================================

def build_executor_batch(dfa_word_list: List[Tuple[dict, tuple]], max_trans: int = 64,
                         device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """从 (dfa, word) 列表构造：trans_tokens, nxt_tokens, trans_mask, sym_ids。
    同一批 L 相同。"""
    B = len(dfa_word_list)
    L = len(dfa_word_list[0][1])
    trans_tokens = torch.zeros((B, max_trans), dtype=torch.long, device=device)
    nxt_tokens = torch.zeros((B, max_trans), dtype=torch.long, device=device)
    trans_mask = torch.zeros((B, max_trans), dtype=torch.bool, device=device)
    syms = []
    for i, (dfa, word) in enumerate(dfa_word_list):
        spec = encode_structured_dfa_spec(dfa)
        nF = spec[3]
        t0 = 4 + nF
        k = spec[1]
        for j in range(k * 2):  # m=2
            trans_tokens[i, j] = spec[t0 + 2 * j]
            nxt_tokens[i, j] = spec[t0 + 2 * j + 1]
            trans_mask[i, j] = True
        syms.append(list(word))
    sym_ids = torch.tensor(syms, dtype=torch.long, device=device)
    return {
        "trans_tokens": trans_tokens,
        "nxt_tokens": nxt_tokens,
        "trans_mask": trans_mask,
        "sym_ids": sym_ids,
    }


def batch_truth(dfa_word_list) -> Tuple[torch.Tensor, torch.Tensor]:
    """真实状态轨迹 (B,L) 与判定 (B,)。"""
    states_list, verdicts = [], []
    for dfa, word in dfa_word_list:
        q = dfa["start"]
        states = []
        for sym in word:
            q = dfa["delta"][(q, sym)]
            states.append(q)
        states_list.append(states)
        verdicts.append(1 if q in set(dfa["accepting"]) else 0)
    return torch.tensor(states_list, dtype=torch.long), torch.tensor(verdicts, dtype=torch.long)


def attention_targets(dfa_word_list, states: torch.Tensor, max_trans: int = 64,
                      device: str = 'cuda') -> torch.Tensor:
    """过程监督：t 时刻应查的转移表条目 j = (prev_state)*2 + sym_t。返回 (B,L)。"""
    B = states.shape[0]
    L = states.shape[1]
    target = torch.zeros((B, L), dtype=torch.long, device=device)
    for i, (dfa, word) in enumerate(dfa_word_list):
        q = dfa["start"]
        for t in range(L):
            target[i, t] = q * 2 + word[t]
            q = states[i, t].item()
    return target


# ============================================================
# 评测（M0 协议，结构化编码）
# ============================================================

@torch.no_grad()
def evaluate_executor(model: NeuralDFAExecutor, k: int, L: int, n: int = 200,
                      seed: int = 42, device: str = 'cuda') -> Dict[str, float]:
    model.eval()
    rng = random.Random(seed)
    f_size = k // 2
    samples = []
    for _ in range(n):
        dfa = make_balanced_dfa(k, 2, f_size, seed=rng.randint(0, 2**31))
        word = tuple(rng.randrange(2) for _ in range(L))
        samples.append((dfa, word))

    all_preds, all_verdicts = [], []
    for i in range(0, n, 32):
        batch = samples[i:i+32]
        inp = build_executor_batch(batch, device=device)
        sl, vl, _ = model(inp["trans_tokens"], inp["nxt_tokens"], inp["trans_mask"], inp["sym_ids"])
        for b in range(len(batch)):
            masked = sl[b].clone()
            masked[:, k:] = float('-inf')
            preds = masked.argmax(dim=-1).cpu().tolist()
            all_preds.append(preds)
            all_verdicts.append(1 if preds[-1] in set(batch[b][0]["accepting"]) else 0)

    correct = 0
    true_verdicts = []
    for (dfa, word), preds in zip(samples, all_preds):
        q = dfa["start"]
        true = []
        for sym in word:
            q = dfa["delta"][(q, sym)]
            true.append(q)
        true_verdicts.append(1 if q in set(dfa["accepting"]) else 0)
        if all(a == b for a, b in zip(preds, true)):
            correct += 1

    beta = sum(all_verdicts) / len(all_verdicts)
    p_pos = f_size / k
    alpha_bias = max(beta * p_pos, (1 - beta) * (1 - p_pos))
    acc = correct / n
    verdict_acc = sum(1 for a, b in zip(all_verdicts, true_verdicts) if a == b) / n
    model.train()
    return {
        "k": k, "L": L, "p_pos": p_pos, "n": n, "beta": beta,
        "alpha_bias": alpha_bias, "acc": acc,
        "alpha_struct": acc - alpha_bias, "verdict_acc": verdict_acc,
    }


# ============================================================
# 训练
# ============================================================

class ExecutorTrainer:
    def __init__(self, model: NeuralDFAExecutor, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device

    def train(self, steps: int, batch_size: int = 32, k_range: Tuple[int, int] = (2, 10),
              L_range: Tuple[int, int] = (20, 50), lr: float = 3e-4, warmup: int = 200,
              log_every: int = 100, eval_every: int = 500,
              output_dir: str = "./checkpoints/dfa_executor", seed: int = 42):
        os.makedirs(output_dir, exist_ok=True)
        rng = random.Random(seed)
        torch.manual_seed(seed)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        metrics = []
        t0 = time.time()
        for step in range(steps):
            k = rng.randint(k_range[0], k_range[1])
            L = rng.randint(L_range[0], L_range[1])
            batch = []
            for _ in range(batch_size):
                dfa = make_balanced_dfa(k, 2, k // 2, seed=rng.randint(0, 2**31))
                word = tuple(rng.randrange(2) for _ in range(L))
                batch.append((dfa, word))
            inp = build_executor_batch(batch, device=self.device)
            sl, vl, al = self.model(inp["trans_tokens"], inp["nxt_tokens"], inp["trans_mask"], inp["sym_ids"])
            states, verdicts = batch_truth(batch)
            states, verdicts = states.to(self.device), verdicts.to(self.device)
            attn_t = attention_targets(batch, states, device=self.device)

            opt.zero_grad()
            state_loss = F.cross_entropy(sl.transpose(1, 2), states)
            verdict_loss = F.cross_entropy(vl, verdicts)
            al_masked = al.masked_fill(~inp["trans_mask"].unsqueeze(1), float('-inf'))
            attn_loss = F.cross_entropy(al_masked.reshape(-1, al_masked.size(-1)), attn_t.reshape(-1))
            loss = state_loss + verdict_loss + self.model.cfg.attention_weight * attn_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            if (step + 1) % log_every == 0 or step == 0:
                print(f"  step {step+1}/{steps} | loss={loss.item():.3f} "
                      f"state={state_loss.item():.3f} verdict={verdict_loss.item():.3f} "
                      f"attn={attn_loss.item():.3f} | {time.time()-t0:.0f}s")
                t0 = time.time()

            if (step + 1) % eval_every == 0:
                configs = [(3, 50), (6, 50), (6, 100), (10, 50), (10, 100), (10, 200)]
                results = []
                for kk, LL in configs:
                    res = evaluate_executor(self.model, kk, LL, n=150, seed=60000 + kk * 100 + LL, device=self.device)
                    results.append(res)
                    print(f"    k={kk} L={LL}: acc={res['acc']:.3f} α={res['alpha_struct']:+.3f}")
                metrics.append({"step": step + 1, "results": results})
                torch.save(self.model.state_dict(), os.path.join(output_dir, f"step{step+1}.pt"))
                with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)

        torch.save(self.model.state_dict(), os.path.join(output_dir, "final.pt"))
        print("=== Training complete ===")
        return metrics


if __name__ == "__main__":
    trainer = ExecutorTrainer(NeuralDFAExecutor(ExecutorConfig(d_model=256, max_k=32)))
    trainer.train(steps=4000, batch_size=32, k_range=(2, 10), L_range=(20, 50),
                  output_dir="./checkpoints/dfa_executor")
