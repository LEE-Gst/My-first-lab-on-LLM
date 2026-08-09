# -*- coding: utf-8 -*-
"""A2 微调：从 rec_final 出发，pyrand 重度加权（含双递归源），低学习率 + NaN 防护。"""
import json
import random

import torch

import nsm.recursive_executor as re_mod
from nsm.py_random_gen import gen_compiled
from nsm.recursive_executor import (RecExecutorConfig, RecursiveExecutor, RecTrainer,
                                    evaluate)


def gen_pyrand(rng, n=None):
    prog, _ = gen_compiled(rng)
    while prog is None:
        prog, _ = gen_compiled(rng)
    return prog


re_mod.REC_GENERATORS["pyrand"] = gen_pyrand


def main():
    device = "cuda"
    model = RecursiveExecutor(RecExecutorConfig(d_model=256, max_instr=re_mod.MAX_INSTR))
    # rec_final 是 max_instr=32 训的，pc_emb 形状不同——不能热启动，直接从头训但用混合分布
    trainer = RecTrainer(model, device=device)
    trainer.train(
        steps=2000,
        templates=["pyrand", "pyrand", "pyrand", "fact", "sum", "fib", "countdown", "randarith"],
        batch_size=96, lr=1e-4, eval_every=400,
        output_dir="./checkpoints/rec_a2c",
    )

    print("\n===== FINAL EVAL =====")
    tests = [
        ("pyrand", None, "compiled-python-subset"),
        ("fact", (1, 10), "template in-dist"),
        ("fib", (1, 6), "template in-dist"),
        ("countdown", (1, 10), "template in-dist"),
        ("randarith", None, "template in-dist"),
    ]
    results = {}
    for tpl, nr, tag in tests:
        res = evaluate(model, tpl, n=40, seed=88000, device=device, n_range=nr)
        results[f"{tpl}_{nr}_{tag}"] = res
        print(f"{tpl} n={nr} [{tag}]: exact={res['exact']:.3f} pc={res['pc_acc']:.3f} "
              f"reg={res['reg_acc']:.3f} sp={res['sp_acc']:.3f}")
    with open("./checkpoints/rec_a2c/final_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved")


if __name__ == "__main__":
    main()
