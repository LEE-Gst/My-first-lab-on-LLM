# -*- coding: utf-8 -*-
"""Phase 2 完整训练 + 扩展评测。"""
import json
import os

from nsm.dfa_executor import ExecutorConfig, ExecutorTrainer, NeuralDFAExecutor, evaluate_executor

os.makedirs("./checkpoints/dfa_executor_final", exist_ok=True)
model = NeuralDFAExecutor(ExecutorConfig(d_model=256, max_k=64, max_trans=128))
trainer = ExecutorTrainer(model)
trainer.train(steps=4000, batch_size=32, k_range=(2, 10), L_range=(20, 50),
              eval_every=1000, output_dir="./checkpoints/dfa_executor_final")

print()
print("=== Extended evaluation ===")
cfgs = [
    (3, 50), (3, 200),
    (6, 50), (6, 200),
    (10, 50), (10, 100), (10, 200), (10, 500), (10, 1000),
    (20, 50), (20, 100),
    (32, 50),
]
results = []
for kk, LL in cfgs:
    res = evaluate_executor(model, kk, LL, n=150, seed=70000 + kk * 100 + LL)
    results.append(res)
    print(f'k={kk:>3} L={LL:>4}: acc={res["acc"]:.3f} alpha_struct={res["alpha_struct"]:+.3f} '
          f'verdict_acc={res["verdict_acc"]:.3f}')
with open("./checkpoints/dfa_executor_final/extended_eval.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("saved extended_eval.json")
