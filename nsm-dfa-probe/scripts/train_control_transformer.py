# -*- coding: utf-8 -*-
"""控制实验：同样结构化编码 + 同样状态监督下，小型因果 Transformer 能否执行 DFA？
对比 Neural DFA Executor（+注意力过程监督）结果，论证显式记忆与串行查询的必要性。
"""
import json
import math
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsm.data import BYTE_SEP, BYTE_SYM0, encode_structured_dfa_spec, make_balanced_dfa
from nsm.dfa_executor import evaluate_executor


@dataclass
class ControlConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    max_seq_len: int = 512
    max_k: int = 64
    vocab_size: int = 256


class ControlTransformer(nn.Module):
    """因果 Transformer：输入 [结构化 spec + word 字节]，输出 word 每步的状态。"""

    def __init__(self, cfg: ControlConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4, dropout=0.0,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.state_head = nn.Linear(cfg.d_model, cfg.max_k)
        self.verdict_head = nn.Linear(cfg.d_model, 2)
        self.register_buffer('causal_mask', nn.Transformer.generate_square_subsequent_mask(cfg.max_seq_len))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        for name, p in self.named_parameters():
            if 'embedding' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        mask = self.causal_mask[:L, :L]
        x = self.encoder(x, mask=mask, is_causal=True)
        return self.state_head(x), self.verdict_head(x)


def build_control_batch(dfa_word_list, device='cuda'):
    """结构化 spec + word 字节（无 END），返回 input_ids, state_labels(位置对齐), verdicts。"""
    B = len(dfa_word_list)
    L_word = len(dfa_word_list[0][1])
    seqs = []
    for dfa, word in dfa_word_list:
        spec = encode_structured_dfa_spec(dfa)
        seqs.append(spec + [BYTE_SYM0 + s for s in word])
    S = max(len(s) for s in seqs)
    input_ids = torch.zeros((B, S), dtype=torch.long, device=device)
    state_labels = torch.full((B, S), -100, dtype=torch.long, device=device)
    verdicts = torch.zeros(B, dtype=torch.long, device=device)
    for i, (dfa, word) in enumerate(dfa_word_list):
        seq = seqs[i]
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        spec_len = len(encode_structured_dfa_spec(dfa))
        q = dfa["start"]
        for t, sym in enumerate(word):
            q = dfa["delta"][(q, sym)]
            state_labels[i, spec_len + t] = q
        verdicts[i] = 1 if q in set(dfa["accepting"]) else 0
    return input_ids, state_labels, verdicts


def evaluate_control(model, k, L, n=150, seed=42, device='cuda'):
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
        input_ids, _, _ = build_control_batch(batch, device=device)
        sl, _ = model(input_ids)
        for b, (dfa, word) in enumerate(batch):
            spec_len = len(encode_structured_dfa_spec(dfa))
            logits = sl[b, spec_len:spec_len+L]
            masked = logits.clone()
            masked[:, k:] = float('-inf')
            preds = masked.argmax(-1).cpu().tolist()
            all_preds.append(preds)
            all_verdicts.append(1 if preds[-1] in set(dfa["accepting"]) else 0)
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
    return {"k": k, "L": L, "acc": acc, "alpha_struct": acc - alpha_bias,
            "verdict_acc": verdict_acc, "beta": beta}


def main():
    device = 'cuda'
    model = ControlTransformer(ControlConfig()).to(device)
    print(f"ControlTransformer params: {model.count_params()/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    steps = 4000
    warmup = 200

    def lr_lambda(s):
        if s < warmup:
            return s / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    rng = random.Random(42)
    torch.manual_seed(42)
    t0 = time.time()
    for step in range(steps):
        k = rng.randint(2, 10)
        L = rng.randint(20, 50)
        batch = []
        for _ in range(32):
            dfa = make_balanced_dfa(k, 2, k // 2, seed=rng.randint(0, 2**31))
            word = tuple(rng.randrange(2) for _ in range(L))
            batch.append((dfa, word))
        input_ids, state_labels, verdicts = build_control_batch(batch, device)
        opt.zero_grad()
        sl, vl = model(input_ids)
        state_loss = F.cross_entropy(sl.transpose(1, 2), state_labels, ignore_index=-100)
        B = input_ids.size(0)
        last_idx = (state_labels != -100).sum(dim=1) - 1
        final_h = vl[torch.arange(B), last_idx]
        verdict_loss = F.cross_entropy(final_h, verdicts)
        loss = state_loss + verdict_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        if (step + 1) % 100 == 0:
            print(f"  step {step+1}/{steps} | loss={loss.item():.3f} state={state_loss.item():.3f} | {time.time()-t0:.0f}s")
            t0 = time.time()
        if (step + 1) % 1000 == 0:
            print("  EVAL:")
            for kk, LL in [(3, 50), (6, 50), (6, 100), (10, 50), (10, 100), (10, 200)]:
                res = evaluate_control(model, kk, LL, n=100, seed=50000 + kk * 100 + LL, device=device)
                print(f"    k={kk} L={LL}: acc={res['acc']:.3f} alpha={res['alpha_struct']:+.3f}")

    os.makedirs("./checkpoints/control_transformer", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/control_transformer/final.pt")
    results = []
    print("\n=== Final control evaluation ===")
    for kk, LL in [(3, 50), (6, 50), (6, 100), (10, 50), (10, 100), (10, 200)]:
        res = evaluate_control(model, kk, LL, n=200, seed=60000 + kk * 100 + LL, device=device)
        results.append(res)
        print(f"k={kk} L={LL}: acc={res['acc']:.3f} alpha_struct={res['alpha_struct']:+.3f}")
    with open("./checkpoints/control_transformer/eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
