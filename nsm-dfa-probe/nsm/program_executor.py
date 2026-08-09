# -*- coding: utf-8 -*-
"""C1: RISC 程序执行探针 v2——寄存器文件显式化的神经解释器。

v1 教训：单体状态向量无锚点 → 寄存器语义学不动。
v2 设计（与 DFA Executor 同哲学）：
  - 寄存器文件 R: (n_reg, d) 显式槽位，值用 val_emb 锚定
  - 取指：对指令槽的软注意力，过程监督对齐（训练时教师强制取指）
  - 指令解析：从 token id 确定性解析 (op, reg_a, reg_b/imm)（结构化编码的固有优势）
  - 语义计算：MLP([指令表示, val(reg_a), val(reg_b), val(imm)]) → 写回值 + 下一 pc
  - 写回目标由 op 语法确定（op<6 写 reg_a），模型只学"算什么"和"跳哪去"
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


# ---------------------------------------------------------------- 指令与编码
OPCODES = ["mov", "inc", "dec", "add", "sub", "seti", "jmp", "jz", "jnz", "halt"]
OPC = {name: i for i, name in enumerate(OPCODES)}
WRITE_OPS = {OPC["mov"], OPC["inc"], OPC["dec"], OPC["add"], OPC["sub"], OPC["seti"]}
OPC_OFFSET = 100
REG_OFFSET = 120
IMM_OFFSET = 140
PAD = 0
START = 200
SEP = 201

N_REG = 8
N_VAL = 16
MAX_INSTR = 16
HALT_CLASS = MAX_INSTR


def encode_instruction(ins: Tuple) -> List[int]:
    op = ins[0]
    if op in ("mov", "add", "sub"):
        return [OPC_OFFSET + OPC[op], REG_OFFSET + ins[1], REG_OFFSET + ins[2]]
    if op in ("inc", "dec"):
        return [OPC_OFFSET + OPC[op], REG_OFFSET + ins[1]]
    if op == "seti":
        return [OPC_OFFSET + OPC[op], REG_OFFSET + ins[1], IMM_OFFSET + ins[2]]
    if op == "jmp":
        return [OPC_OFFSET + OPC[op], IMM_OFFSET + ins[1]]
    if op in ("jz", "jnz"):
        return [OPC_OFFSET + OPC[op], REG_OFFSET + ins[1], IMM_OFFSET + ins[2]]
    if op == "halt":
        return [OPC_OFFSET + OPC[op]]
    raise ValueError(op)


def encode_program(program: List[Tuple]) -> Tuple[List[int], List[int]]:
    toks = [START, len(program)]
    positions = []
    for ins in program:
        positions.append(len(toks))
        toks.extend(encode_instruction(ins))
    toks.append(SEP)
    return toks, positions


# ---------------------------------------------------------------- 程序语义
def run_program(program: List[Tuple], word: Tuple[int, ...], max_steps: int = 300):
    """返回轨迹 [(pc_after, regs_after), ...]。halt 后追加一个冻结步即停。"""
    regs = [0] * N_REG
    for i, b in enumerate(word[:N_REG]):
        regs[i] = b
    pc = 0
    trace = []
    halted_steps = 0
    for _ in range(max_steps):
        if pc < 0 or pc >= len(program):
            trace.append((HALT_CLASS, list(regs)))
            halted_steps += 1
            if halted_steps >= 2:
                break
            continue
        ins = program[pc]
        op = ins[0]
        if op == "halt":
            trace.append((HALT_CLASS, list(regs)))
            pc = -1
            continue
        if op == "mov":
            regs[ins[1]] = regs[ins[2]]
        elif op == "inc":
            regs[ins[1]] = (regs[ins[1]] + 1) % N_VAL
        elif op == "dec":
            regs[ins[1]] = (regs[ins[1]] - 1) % N_VAL
        elif op == "add":
            regs[ins[1]] = (regs[ins[1]] + regs[ins[2]]) % N_VAL
        elif op == "sub":
            regs[ins[1]] = (regs[ins[1]] - regs[ins[2]]) % N_VAL
        elif op == "seti":
            regs[ins[1]] = ins[2]
        elif op == "jmp":
            pc = ins[1]
            trace.append((pc if pc < len(program) else HALT_CLASS, list(regs)))
            continue
        elif op == "jz":
            pc = ins[2] if regs[ins[1]] == 0 else pc + 1
            trace.append((pc if pc < len(program) else HALT_CLASS, list(regs)))
            continue
        elif op == "jnz":
            pc = ins[2] if regs[ins[1]] != 0 else pc + 1
            trace.append((pc if pc < len(program) else HALT_CLASS, list(regs)))
            continue
        pc += 1
        trace.append((pc if pc < len(program) else HALT_CLASS, list(regs)))
    return trace


# ---------------------------------------------------------------- 程序生成
def gen_arith(rng, n_instr):
    prog = []
    for _ in range(n_instr - 1):
        op = rng.choice(["mov", "inc", "dec", "add", "sub", "seti"])
        if op in ("mov", "add", "sub"):
            prog.append((op, rng.randrange(N_REG), rng.randrange(N_REG)))
        elif op in ("inc", "dec"):
            prog.append((op, rng.randrange(N_REG)))
        else:
            prog.append(("seti", rng.randrange(N_REG), rng.randrange(N_VAL)))
    prog.append(("halt",))
    return prog


def gen_controlflow(rng, n_instr):
    n_jumps = rng.randint(1, 3)
    hi = max(1, n_instr - 4)
    jump_positions = sorted(rng.sample(range(1, hi), min(n_jumps, hi - 1))) if hi > 1 else []
    prog = []
    for i in range(n_instr - 1):
        if i in jump_positions:
            op = rng.choice(["jz", "jnz"])
            target = min(i + rng.randint(2, 4), n_instr - 1)
            prog.append((op, rng.randrange(N_REG), target))
        else:
            op = rng.choice(["mov", "inc", "dec", "add", "sub", "seti"])
            if op in ("mov", "add", "sub"):
                prog.append((op, rng.randrange(N_REG), rng.randrange(N_REG)))
            elif op in ("inc", "dec"):
                prog.append((op, rng.randrange(N_REG)))
            else:
                prog.append(("seti", rng.randrange(N_REG), rng.randrange(N_VAL)))
    prog.append(("halt",))
    return prog


def gen_loop(rng, n_instr):
    prog = []
    for _ in range(rng.randint(1, 3)):
        prog.append(("seti", rng.randrange(N_REG), rng.randrange(N_VAL)))
    rc = rng.randrange(N_REG)
    n_iter = rng.randint(1, 5)
    loop_start = len(prog)
    prog.append(("seti", rc, n_iter))
    for _ in range(rng.randint(1, 3)):
        op = rng.choice(["mov", "inc", "dec", "add", "sub"])
        tgt = rng.choice([r for r in range(N_REG) if r != rc])
        if op in ("mov", "add", "sub"):
            prog.append((op, tgt, rng.randrange(N_REG)))
        else:
            prog.append((op, tgt))
    prog.append(("dec", rc))
    prog.append(("jnz", rc, loop_start))
    while len(prog) < n_instr - 1:
        op = rng.choice(["mov", "inc", "dec", "add", "sub", "seti"])
        if op in ("mov", "add", "sub"):
            prog.append((op, rng.randrange(N_REG), rng.randrange(N_REG)))
        elif op in ("inc", "dec"):
            prog.append((op, rng.randrange(N_REG)))
        else:
            prog.append(("seti", rng.randrange(N_REG), rng.randrange(N_VAL)))
    prog = prog[: n_instr - 1]
    prog.append(("halt",))
    return prog


GENERATORS = {"arith": gen_arith, "cf": gen_controlflow, "loop": gen_loop}


def gen_mixed(rng, n_instr):
    """混合采样三种生成器。"""
    return GENERATORS[rng.choice(["arith", "cf", "loop"])](rng, n_instr)


GENERATORS["mix"] = gen_mixed


# ---------------------------------------------------------------- 批量构造
def parse_slot_tokens(toks: List[int]):
    """从一条指令的 token id 确定性解析 (op, reg_a, reg_b, imm)。PAD/缺失给安全默认。"""
    op = (toks[0] - OPC_OFFSET) if len(toks) > 0 else 0
    reg_a = (toks[1] - REG_OFFSET) if len(toks) > 1 else 0
    reg_b, imm = 0, 0
    if len(toks) > 2:
        t2 = toks[2]
        if REG_OFFSET <= t2 < REG_OFFSET + N_REG:
            reg_b = t2 - REG_OFFSET
        elif IMM_OFFSET <= t2 < IMM_OFFSET + N_VAL:
            imm = t2 - IMM_OFFSET
    op = min(max(op, 0), len(OPCODES) - 1)
    reg_a = min(max(reg_a, 0), N_REG - 1)
    return op, reg_a, reg_b, imm


def build_batch(samples, device="cuda", max_instr=MAX_INSTR):
    """返回模型输入 + 全部监督目标。全部先在 CPU 构建，最后一次传输。"""
    B = len(samples)
    instr_tok_l = []
    slot_mask_l = []
    words_l = []
    traces = []
    progs = []
    for prog, word in samples:
        assert len(prog) <= max_instr
        toks, positions = encode_program(prog)
        slots = []
        for j in range(max_instr):
            if j < len(positions):
                pos = positions[j]
                ins_toks = toks[pos: pos + 3]
                slots.append(ins_toks + [PAD] * (3 - len(ins_toks)))
            else:
                slots.append([PAD, PAD, PAD])
        instr_tok_l.append(slots)
        slot_mask_l.append([j < len(prog) for j in range(max_instr)])
        wl = [0] * N_REG
        for r, b in enumerate(word[:N_REG]):
            wl[r] = b
        words_l.append(wl)
        progs.append(prog)
        traces.append(run_program(prog, word))

    T = max(len(tr) for tr in traces)
    for tr in traces:
        while len(tr) < T:
            tr.append(tr[-1])

    fetch_l, pc_l, reg_l, hist_l, wvm_l, wvt_l = [], [], [], [], [], []
    for i, tr in enumerate(traces):
        prev_pc = 0
        prev_regs = list(words_l[i])
        f_t, p_t, r_t, h_t, m_t, v_t = [], [], [], [], [], []
        for t in range(T):
            halted = prev_pc >= len(progs[i])
            f_t.append(prev_pc if not halted else HALT_CLASS)
            p_t.append(tr[t][0])
            r_t.append(list(tr[t][1]))
            h_t.append(list(prev_regs))
            if not halted:
                ins = progs[i][prev_pc]
                if OPC[ins[0]] in WRITE_OPS:
                    m_t.append(True)
                    v_t.append(tr[t][1][ins[1]])
                else:
                    m_t.append(False)
                    v_t.append(0)
            else:
                m_t.append(False)
                v_t.append(0)
            prev_pc = tr[t][0]
            prev_regs = list(tr[t][1])
        fetch_l.append(f_t)
        pc_l.append(p_t)
        reg_l.append(r_t)
        hist_l.append(h_t)
        wvm_l.append(m_t)
        wvt_l.append(v_t)

    tt = lambda x, dt: torch.tensor(x, dtype=dt, device=device)
    return {
        "instr_tok": tt(instr_tok_l, torch.long),
        "slot_mask": tt(slot_mask_l, torch.bool),
        "words": tt(words_l, torch.long),
        "fetch_target": tt(fetch_l, torch.long),
        "pc_target": tt(pc_l, torch.long),
        "reg_target": tt(reg_l, torch.long),
        "reg_history": tt(hist_l, torch.long),
        "wv_mask": tt(wvm_l, torch.bool),
        "wv_target": tt(wvt_l, torch.long),
        "T": T,
    }


# ---------------------------------------------------------------- 模型 v2
@dataclass
class ProgramExecutorConfig:
    d_model: int = 256
    max_instr: int = MAX_INSTR
    n_reg: int = N_REG
    n_val: int = N_VAL
    attention_weight: float = 0.5
    wv_weight: float = 1.0


class ProgramExecutor(nn.Module):
    def __init__(self, cfg: ProgramExecutorConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_emb = nn.Embedding(256, d)
        self.val_emb = nn.Embedding(cfg.n_val, d)
        self.pc_emb = nn.Embedding(cfg.max_instr + 1, d)  # + halt
        self.op_emb = nn.Embedding(len(OPCODES), d)
        self.k_pc = nn.Linear(d, d)
        self.q_pc = nn.Linear(d, d)
        # 语义 = f(op, pc, val(reg_a), val(reg_b), imm) —— RISC 指令上下文无关
        self.compute = nn.Sequential(
            nn.Linear(5 * d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, 2 * d), nn.GELU(),
        )
        self.wv_head = nn.Linear(2 * d, cfg.n_val)
        self.pc_head = nn.Linear(2 * d, cfg.max_instr + 1)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def parse_slots(self, instr_tok):
        op = (instr_tok[:, :, 0] - OPC_OFFSET).clamp(0, len(OPCODES) - 1)
        reg_a = (instr_tok[:, :, 1] - REG_OFFSET).clamp(0, N_REG - 1)
        t2 = instr_tok[:, :, 2]
        reg_b = (t2 - REG_OFFSET).clamp(0, N_REG - 1)
        imm = (t2 - IMM_OFFSET).clamp(0, N_VAL - 1)
        return op, reg_a, reg_b, imm

    def forward(self, instr_tok, slot_mask, words, T, fetch_teacher=None, reg_hist=None,
                teacher_prob=1.0):
        B, S, _ = instr_tok.shape
        dev = instr_tok.device
        keys = self.k_pc(self.pc_emb.weight[:S])
        op_s, rega_s, regb_s, imm_s = self.parse_slots(instr_tok)
        imm_emb_s = self.val_emb(imm_s)  # (B,S,d) 预计算

        R = self.val_emb(words)
        p = self.pc_emb(torch.zeros(B, dtype=torch.long, device=dev))
        batch_idx = torch.arange(B, device=dev)

        pc_l, wv_l, attn_l, R_l, p_l = [], [], [], [], []
        halted = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(T):
            # 三个状态统一教师强制：槽、寄存器历史、pc 表示
            use_hist = None
            p_read = p
            slot_auto = None
            if reg_hist is not None:
                use_hist = torch.rand(B, device=dev) < teacher_prob
                if fetch_teacher is not None:
                    pc_teacher = self.pc_emb(fetch_teacher[:, t].clamp(0, MAX_INSTR))
                    p_read = torch.where(use_hist.view(B, 1), pc_teacher, p)

            q = self.q_pc(p_read)
            scores = q @ keys.t()
            scores = scores.masked_fill(~slot_mask, float("-inf"))
            attn_l.append(scores)

            slot_auto = scores.argmax(-1)
            R_read = R
            if reg_hist is not None:
                R_truth = self.val_emb(reg_hist[:, t])
                R_read = torch.where(use_hist.view(B, 1, 1), R_truth, R)
                if fetch_teacher is not None:
                    slot = torch.where(use_hist, fetch_teacher[:, t].clamp(0, S - 1), slot_auto)
                else:
                    slot = slot_auto
            elif fetch_teacher is not None:
                slot = fetch_teacher[:, t].clamp(0, S - 1)
            else:
                slot = slot_auto

            op = op_s[batch_idx, slot]
            reg_a = rega_s[batch_idx, slot]
            reg_b = regb_s[batch_idx, slot]
            imm_e = imm_emb_s[batch_idx, slot]

            va = R_read[batch_idx, reg_a]
            vb = R_read[batch_idx, reg_b]
            h = self.compute(torch.cat([self.op_emb(op), p_read, va, vb, imm_e], dim=-1))
            wv_logits = self.wv_head(h)
            pc_logits = self.pc_head(h)
            wv_l.append(wv_logits)
            pc_l.append(pc_logits)

            write_mask = ((op < 6) & ~halted).float().unsqueeze(-1)
            wv_soft = torch.softmax(wv_logits, dim=-1) @ self.val_emb.weight
            wv_hard = self.val_emb(wv_logits.argmax(-1))
            new_val_repr = wv_hard + (wv_soft - wv_soft.detach())
            R_new = R.clone()
            R_new[batch_idx, reg_a] = R[batch_idx, reg_a] * (1 - write_mask) + new_val_repr * write_mask
            R = R_new

            pc_soft = torch.softmax(pc_logits, dim=-1) @ self.pc_emb.weight
            pc_hard = self.pc_emb(pc_logits.argmax(-1))
            p = pc_hard + (pc_soft - pc_soft.detach())
            halted = halted | (pc_logits.argmax(-1) == HALT_CLASS)

            R_l.append(R)
            p_l.append(p)

        return (torch.stack(pc_l, 1), torch.stack(wv_l, 1), torch.stack(attn_l, 1),
                torch.stack(R_l, 1), torch.stack(p_l, 1))

    def decode_regs(self, R):
        sim = R @ self.val_emb.weight.t()
        return sim.argmax(-1)

    def decode_pc(self, p):
        sim = p @ self.pc_emb.weight.t()
        return sim.argmax(-1)


# ---------------------------------------------------------------- 评测
@torch.no_grad()
def evaluate(model, gen_name: str, n: int = 100, n_instr: int = 10, word_len: int = 6,
             seed: int = 42, device: str = "cuda") -> Dict[str, float]:
    model.eval()
    rng = random.Random(seed)
    gen = GENERATORS[gen_name]
    exact = pc_hits = reg_hits = cnt = 0
    for _ in range(n):
        prog = gen(rng, n_instr)
        word = tuple(rng.randrange(2) for _ in range(word_len))
        batch = build_batch([(prog, word)], device=device)
        pc_l, wv_l, attn_l, R_l, p_l = model(batch["instr_tok"], batch["slot_mask"],
                                             batch["words"], batch["T"])
        pred_pc = pc_l[0].argmax(-1).cpu()
        pred_reg = model.decode_regs(R_l)[0].cpu()
        truth = run_program(prog, word)
        T = batch["T"]
        ok = True
        for t in range(T):
            tt_pc, tt_regs = truth[t]
            pc_ok = pred_pc[t].item() == tt_pc
            reg_ok = all(pred_reg[t][r].item() == tt_regs[r] for r in range(N_REG))
            pc_hits += pc_ok
            reg_hits += reg_ok
            cnt += 1
            if not (pc_ok and reg_ok):
                ok = False
        exact += ok
    model.train()
    return {"gen": gen_name, "n": n, "exact": exact / n,
            "pc_acc": pc_hits / cnt, "reg_acc": reg_hits / cnt}


# ---------------------------------------------------------------- 训练
class ProgramTrainer:
    def __init__(self, model: ProgramExecutor, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device

    def train(self, steps: int, stage: str = "arith", batch_size: int = 32,
              n_instr_range: Tuple[int, int] = (6, 12), word_len: int = 6,
              lr: float = 3e-4, warmup: int = 200, log_every: int = 100,
              eval_every: int = 500, output_dir: str = "./checkpoints/program_executor",
              seed: int = 42):
        os.makedirs(output_dir, exist_ok=True)
        rng = random.Random(seed)
        torch.manual_seed(seed)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

        def lr_lambda(s):
            if s < warmup:
                return s / max(1, warmup)
            return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        cfg = self.model.cfg
        metrics = []
        t0 = time.time()
        for step in range(steps):
            n_instr = rng.randint(*n_instr_range)
            samples = [(GENERATORS[stage](rng, n_instr),
                        tuple(rng.randrange(2) for _ in range(word_len)))
                       for _ in range(batch_size)]
            batch = build_batch(samples, device=self.device)
            # 教师强制前 50% 全程，50-80% 退火到 0，最后 20% 全自主（滚动练习）
            frac = step / max(1, steps)
            if frac < 0.5:
                teacher_prob = 1.0
            elif frac < 0.8:
                teacher_prob = 1.0 - (frac - 0.5) / 0.3
            else:
                teacher_prob = 0.0
            pc_l, wv_l, attn_l, R_l, p_l = self.model(
                batch["instr_tok"], batch["slot_mask"], batch["words"], batch["T"],
                fetch_teacher=batch["fetch_target"], reg_hist=batch["reg_history"],
                teacher_prob=teacher_prob)

            opt.zero_grad()
            pc_loss = F.cross_entropy(pc_l.transpose(1, 2), batch["pc_target"])
            flat_mask = batch["wv_mask"].reshape(-1)
            if flat_mask.any():
                wv_loss = F.cross_entropy(wv_l.reshape(-1, N_VAL)[flat_mask],
                                          batch["wv_target"].reshape(-1)[flat_mask])
            else:
                wv_loss = torch.zeros((), device=self.device)
            attn_masked = attn_l.masked_fill(~batch["slot_mask"].unsqueeze(1), float("-inf"))
            valid_attn = (batch["fetch_target"] < MAX_INSTR).reshape(-1)
            if valid_attn.any():
                attn_loss = F.cross_entropy(
                    attn_masked.reshape(-1, attn_masked.size(-1))[valid_attn],
                    batch["fetch_target"].reshape(-1)[valid_attn])
            else:
                attn_loss = torch.zeros((), device=self.device)
            loss = pc_loss + cfg.wv_weight * wv_loss + cfg.attention_weight * attn_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            if (step + 1) % log_every == 0 or step == 0:
                print(f"  step {step+1}/{steps} | loss={loss.item():.3f} pc={pc_loss.item():.3f} "
                      f"wv={wv_loss.item():.3f} attn={attn_loss.item():.3f} | {time.time()-t0:.0f}s",
                      flush=True)
                t0 = time.time()
            if (step + 1) % eval_every == 0:
                res = evaluate(self.model, stage, n=50, seed=60000 + step, device=self.device)
                print(f"    EVAL {stage}: exact={res['exact']:.3f} pc={res['pc_acc']:.3f} "
                      f"reg={res['reg_acc']:.3f}", flush=True)
                metrics.append({"step": step + 1, **res})
                torch.save(self.model.state_dict(), os.path.join(output_dir, f"step{step+1}.pt"))
                with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)

        torch.save(self.model.state_dict(), os.path.join(output_dir, "final.pt"))
        print("=== Training complete ===")
        return metrics


if __name__ == "__main__":
    trainer = ProgramTrainer(ProgramExecutor(ProgramExecutorConfig(d_model=256)))
    trainer.train(steps=2000, stage="arith", output_dir="./checkpoints/program_executor_arith")
