# -*- coding: utf-8 -*-
"""C1 混合训练：三种程序混合 + n_instr 覆盖 6-16，解决阶段遗忘与长程序泛化。"""
import json
import os

import torch

from nsm.program_executor import (
    ProgramExecutor, ProgramExecutorConfig, ProgramTrainer, evaluate,
)

def main():
    device = "cuda"
    model = ProgramExecutor(ProgramExecutorConfig(d_model=256))
    # 从 loop 阶段最佳模型热启动
    prev = "./checkpoints/program_executor_loop/final.pt"
    if os.path.exists(prev):
        model.load_state_dict(torch.load(prev, map_location=device))
        print(f"warm start from {prev}")
    trainer = ProgramTrainer(model, device=device)
    trainer.train(steps=1500, stage="mix", batch_size=192,
                  n_instr_range=(6, 16), eval_every=250,
                  output_dir="./checkpoints/program_executor_mix")

    print("\n===== FINAL EVAL (mix model) =====")
    results = {}
    for stage in ["arith", "cf", "loop"]:
        for n_instr in [10, 14, 16]:
            res = evaluate(model, stage, n=100, n_instr=n_instr,
                           seed=80000 + n_instr, device=device)
            results[f"{stage}_n{n_instr}"] = res
            print(f"{stage} n_instr={n_instr}: exact={res['exact']:.3f} "
                  f"pc={res['pc_acc']:.3f} reg={res['reg_acc']:.3f}")
    with open("./checkpoints/program_executor_mix/final_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved")


if __name__ == "__main__":
    main()
