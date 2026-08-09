# -*- coding: utf-8 -*-
"""NSM 在 DFA 上的三分量分解评测（复用 M0 协议）。"""
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from nsm.data import BYTE_END, BYTE_SEP, BYTE_SYM0, encode_dfa_spec, encode_structured_dfa_spec, make_balanced_dfa, run_dfa
from nsm.model import NSMByteLM


@dataclass
class EvalConfig:
    k: int
    p_pos: float
    f_size: int
    L: int
    n: int = 200
    seed: int = 42


def encode_prompt(dfa, word, structured: bool = False) -> List[int]:
    """只编码 spec + input（无 END）。"""
    spec = encode_structured_dfa_spec(dfa) if structured else encode_dfa_spec(dfa)
    input_bytes = [BYTE_SYM0 + sym for sym in word]
    return spec + input_bytes


def batch_encode(dfas_words, max_len=None, device='cuda', structured: bool = False):
    """把 (dfa, word) 列表编码成 padded tensor。"""
    tokens = [encode_prompt(dfa, word, structured=structured) for dfa, word in dfas_words]
    if max_len is None:
        max_len = max(len(t) for t in tokens)
    batch = []
    for t in tokens:
        batch.append(t + [0] * (max_len - len(t)))
    return torch.tensor(batch, dtype=torch.long, device=device)


def evaluate_model(model: NSMByteLM, cfg: EvalConfig, device='cuda', structured: bool = False) -> Dict[str, float]:
    model.eval()
    rng = random.Random(cfg.seed)

    # 生成测试样本
    samples = []
    for _ in range(cfg.n):
        dfa = make_balanced_dfa(cfg.k, 2, cfg.f_size, seed=rng.randint(0, 2**31))
        word = tuple(rng.randrange(2) for _ in range(cfg.L))
        samples.append((dfa, word))

    # 分批推理
    batch_size = 32
    all_state_preds = []
    all_verdict_preds = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        input_ids = batch_encode(batch, device=device, structured=structured)
        with torch.no_grad():
            _, state_logits, verdict_logits, _ = model(input_ids)
        for b, (dfa, word) in enumerate(batch):
            spec_len_i = len(encode_structured_dfa_spec(dfa) if structured else encode_dfa_spec(dfa))
            input_state_logits = state_logits[b, spec_len_i:spec_len_i+cfg.L]
            k = dfa["k"]
            masked = input_state_logits.clone()
            masked[:, k:] = float('-inf')
            state_preds = masked.argmax(dim=-1).cpu().tolist()
            all_state_preds.append(state_preds)
            final_state = state_preds[-1]
            verdict = 1 if final_state in set(dfa["accepting"]) else 0
            all_verdict_preds.append(verdict)

    # 计算指标
    correct = 0
    for (dfa, word), state_preds in zip(samples, all_state_preds):
        true_states = []
        q = dfa["start"]
        for sym in word:
            q = dfa["delta"][(q, sym)]
            true_states.append(q)
        if all(a == b for a, b in zip(state_preds, true_states)):
            correct += 1

    # 偏置基线
    true_verdicts = [1 if run_dfa(dfa, word) else 0 for dfa, word in samples]
    beta = sum(all_verdict_preds) / len(all_verdict_preds)
    p_pos = cfg.p_pos
    alpha_bias = max(beta * p_pos, (1 - beta) * (1 - p_pos))
    acc = correct / len(samples)

    return {
        "k": cfg.k,
        "L": cfg.L,
        "p_pos": p_pos,
        "n": cfg.n,
        "beta": beta,
        "alpha_bias": alpha_bias,
        "acc": acc,
        "alpha_struct": acc - alpha_bias,  # 随机 DFA alpha_surf ≈ 0
        "verdict_acc": sum(1 for a, b in zip(all_verdict_preds, true_verdicts) if a == b) / len(samples),
    }


def run_eval_suite(model: NSMByteLM, device='cuda', structured: bool = False) -> List[Dict[str, float]]:
    """跑一组配置：B6/B10 在 L=20/50/100/200。"""
    configs = [
        EvalConfig(k=6, p_pos=0.5, f_size=3, L=20),
        EvalConfig(k=6, p_pos=0.5, f_size=3, L=50),
        EvalConfig(k=6, p_pos=0.5, f_size=3, L=100),
        EvalConfig(k=6, p_pos=0.5, f_size=3, L=200),
        EvalConfig(k=10, p_pos=0.5, f_size=5, L=20),
        EvalConfig(k=10, p_pos=0.5, f_size=5, L=50),
        EvalConfig(k=10, p_pos=0.5, f_size=5, L=100),
        EvalConfig(k=10, p_pos=0.5, f_size=5, L=200),
    ]
    results = []
    for cfg in configs:
        res = evaluate_model(model, cfg, device=device, structured=structured)
        results.append(res)
        print(f"  k={cfg.k} L={cfg.L}: acc={res['acc']:.3f} alpha_struct={res['alpha_struct']:+.3f} "
              f"beta={res['beta']:.3f} verdict_acc={res['verdict_acc']:.3f}")
    return results


if __name__ == "__main__":
    from nsm.model import NSMConfig
    cfg = NSMConfig(d_model=256, n_layers=4, n_partitions=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NSMByteLM(cfg).to(device)
    run_eval_suite(model, device=device)
