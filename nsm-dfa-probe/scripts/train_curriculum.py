# -*- coding: utf-8 -*-
"""Phase 1: 课程化天花板探索。
固定小 DFA 池过拟合 → 扩展池 → 提升 k → 提升 L。
每阶段保存 checkpoint + 评测（池内记忆 vs 新 DFA 泛化 vs 跨 k）。
"""
import json
import math
import os
import random
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from nsm.data import BYTE_SYM0, encode_dfa_spec, encode_sample, make_balanced_dfa
from nsm.eval import EvalConfig, evaluate_model
from nsm.model import GRULM, NSMConfig


@dataclass
class Stage:
    k: int
    L: int
    initial_pool: int
    max_pool: int
    expand_every: int
    expand_by: int
    steps: int
    lr: float = 3e-4
    d_model: int = 512
    n_layers: int = 2
    warmup: int = 200


STAGES = [
    Stage(k=3, L=20, initial_pool=5, max_pool=5, expand_every=10**9, expand_by=0, steps=3000),
    Stage(k=3, L=20, initial_pool=5, max_pool=20, expand_every=1000, expand_by=5, steps=5000),
    Stage(k=3, L=20, initial_pool=20, max_pool=100, expand_every=1000, expand_by=20, steps=5000),
    Stage(k=3, L=50, initial_pool=100, max_pool=100, expand_every=10**9, expand_by=0, steps=5000),
    Stage(k=4, L=20, initial_pool=20, max_pool=20, expand_every=10**9, expand_by=0, steps=5000),
    Stage(k=5, L=20, initial_pool=20, max_pool=20, expand_every=10**9, expand_by=0, steps=5000),
    Stage(k=6, L=20, initial_pool=20, max_pool=20, expand_every=10**9, expand_by=0, steps=5000),
]


class CurriculumRunner:
    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.rng = random.Random(seed)
        self.device = torch.device('cuda')
        self.metrics = []
        self.pool = []
        self.pool_seed = 0
        self.model = None

    def make_dfa(self, k, seed):
        return make_balanced_dfa(k, 2, k // 2, seed=seed)

    def save_model(self, name):
        torch.save({
            "model": self.model.state_dict(),
            "pool_seed": self.pool_seed,
        }, os.path.join(self.output_dir, name))

    def save_metrics(self):
        with open(os.path.join(self.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)

    @torch.no_grad()
    def eval_pool(self, k, L, n=100):
        """池内记忆：训练池 DFA 上的全轨迹准确率。"""
        self.model.eval()
        correct = 0
        for i in range(n):
            dfa = self.rng.choice(self.pool)
            word = tuple(self.rng.randrange(2) for _ in range(L))
            tokens = encode_dfa_spec(dfa) + [BYTE_SYM0 + s for s in word]
            ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            _, sl, _, _ = self.model(ids)
            preds = sl[0, len(encode_dfa_spec(dfa)):len(encode_dfa_spec(dfa)) + L].argmax(-1).cpu().tolist()
            q = dfa["start"]
            true = []
            for sym in word:
                q = dfa["delta"][(q, sym)]
                true.append(q)
            if all(a == b for a, b in zip(preds, true)):
                correct += 1
        self.model.train()
        return correct / n

    @torch.no_grad()
    def eval_new_dfa(self, k, L, n=150, seed=90000):
        """全新同 k DFA 上的全轨迹准确率（标准 M0 协议）。"""
        ec = EvalConfig(k=k, p_pos=(k // 2) / k, f_size=k // 2, L=L, n=n, seed=seed)
        res = evaluate_model(self.model, ec, device='cuda')
        return res

    def evaluate_stage(self, stage, stage_id, step_label):
        self.model.eval()
        results = {
            "stage": stage_id,
            "k": stage.k,
            "L": stage.L,
            "pool_size": len(self.pool),
            "step": step_label,
        }
        results["pool_acc"] = self.eval_pool(stage.k, stage.L)
        line = f"  EVAL s{stage_id} step{step_label}: pool_acc={results['pool_acc']:.3f}"

        # 全新同 k DFA：L, 2L, min(5L,200)
        for L in sorted({stage.L, stage.L * 2, min(stage.L * 5, 200)}):
            res = self.eval_new_dfa(stage.k, L, seed=90000 + stage.k * 1000 + L)
            results[f"newk{stage.k}L{L}"] = res
            line += f" | k{stage.k}L{L}:acc={res['acc']:.3f}/α={res['alpha_struct']:+.3f}"

        # 跨 k probe（固定 L=20）
        for k in [3, 4, 5, 6, 10]:
            res = self.eval_new_dfa(k, 20, seed=91000 + k)
            results[f"crossk{k}L20"] = res
            line += f" | ck{k}:acc={res['acc']:.3f}/α={res['alpha_struct']:+.3f}"

        print(line)
        self.model.train()
        return results

    def run_stage(self, stage: Stage, stage_id: int):
        print(f"\n===== Stage {stage_id}: k={stage.k} L={stage.L} pool={stage.initial_pool}->{stage.max_pool} steps={stage.steps} =====")

        self.pool_seed = self.rng.randint(0, 2**31)
        self.pool = [self.make_dfa(stage.k, self.pool_seed + i) for i in range(stage.initial_pool)]

        if self.model is None:
            self.model = GRULM(NSMConfig(d_model=stage.d_model, n_layers=stage.n_layers, n_partitions=1)).to(self.device)
            print(f"  new model: {self.model.count_params()/1e6:.1f}M params")

        opt = torch.optim.AdamW(self.model.parameters(), lr=stage.lr, betas=(0.9, 0.95), weight_decay=0.01)

        def lr_lambda(step):
            if step < stage.warmup:
                return step / max(1, stage.warmup)
            return 0.5 * (1 + math.cos(math.pi * (step - stage.warmup) / max(1, stage.steps - stage.warmup)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        batch_size = 32
        t0 = time.time()
        for step in range(stage.steps):
            if step > 0 and step % stage.expand_every == 0 and len(self.pool) < stage.max_pool:
                n_add = min(stage.expand_by, stage.max_pool - len(self.pool))
                self.pool.extend([self.make_dfa(stage.k, self.pool_seed + len(self.pool) + i) for i in range(n_add)])
                print(f"    step {step}: pool -> {len(self.pool)}")

            samples = []
            max_len = 0
            for _ in range(batch_size):
                dfa = self.rng.choice(self.pool)
                word = tuple(self.rng.randrange(2) for _ in range(stage.L))
                tokens, states, verdict = encode_sample(dfa, word)
                samples.append((tokens, states, verdict))
                max_len = max(max_len, len(tokens))

            padded = [t + [0] * (max_len - len(t)) for t, _, _ in samples]
            state_labels = []
            for tokens, states, _ in samples:
                spec_len = len(tokens) - len(states) - 1
                seq = [-100] * spec_len + states + [-100]
                seq = seq + [-100] * (max_len - len(seq))
                state_labels.append(seq)

            ids = torch.tensor(padded, dtype=torch.long, device=self.device)
            state_labels = torch.tensor(state_labels, dtype=torch.long, device=self.device)
            verdict_labels = torch.tensor([v for _, _, v in samples], dtype=torch.long, device=self.device)

            opt.zero_grad()
            _, sl, vl, _ = self.model(ids)
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), state_labels.view(-1), ignore_index=-100)
            B = ids.size(0)
            last_idx = (state_labels != -100).sum(dim=1) - 1
            final_h = vl[torch.arange(B), last_idx]
            loss += F.cross_entropy(final_h, verdict_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            if (step + 1) % 500 == 0:
                print(f"    step {step+1}/{stage.steps} | loss={loss.item():.3f} | {(time.time()-t0):.0f}s")
                t0 = time.time()

            if (step + 1) % 1000 == 0:
                self.save_model(f"stage{stage_id}_step{step+1}.pt")
                ev = self.evaluate_stage(stage, stage_id, step + 1)
                self.metrics.append(ev)
                self.save_metrics()

        self.save_model(f"stage{stage_id}_final.pt")
        ev = self.evaluate_stage(stage, stage_id, "final")
        self.metrics.append(ev)
        self.save_metrics()

    def run(self):
        for i, stage in enumerate(STAGES):
            self.run_stage(stage, i + 1)

        print("\n" + "=" * 100)
        print("Curriculum ceiling summary (pool memorization vs new-DFA generalization)")
        print("=" * 100)
        print(f"{'stage':<6} {'k':<3} {'L':<4} {'pool':<5} {'step':<6} {'pool_acc':<9} {'newk acc@L':<20} {'crossk3@L20':<14} {'crossk4@L20':<14} {'crossk5@L20':<14} {'crossk10@L20':<14}")
        for m in self.metrics:
            na = m.get(f"newk{m['k']}L{m['L']}", {})
            c3 = m.get("crossk3L20", {})
            c4 = m.get("crossk4L20", {})
            c5 = m.get("crossk5L20", {})
            c10 = m.get("crossk10L20", {})
            print(f"{m['stage']:<6} {m['k']:<3} {m['L']:<4} {m['pool_size']:<5} {str(m['step']):<6} "
                  f"{m['pool_acc']:.3f}     {na.get('acc', -1):.3f}            "
                  f"{c3.get('acc', -1):.3f}(α{c3.get('alpha_struct', 0):+.2f})    "
                  f"{c4.get('acc', -1):.3f}(α{c4.get('alpha_struct', 0):+.2f})    "
                  f"{c5.get('acc', -1):.3f}(α{c5.get('alpha_struct', 0):+.2f})    "
                  f"{c10.get('acc', -1):.3f}(α{c10.get('alpha_struct', 0):+.2f})")

        with open(os.path.join(self.output_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(self.metrics, ensure_ascii=False, indent=2))
        print(f"\nSaved to {self.output_dir}")


if __name__ == "__main__":
    runner = CurriculumRunner("./checkpoints/curriculum_ceiling")
    runner.run()
