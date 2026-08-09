# -*- coding: utf-8 -*-
"""A1: 递归程序执行探针——ProgramExecutor 加值栈与调用栈，支持 call/ret 递归。

继承 C1 全部方法学：
  - 结构化编码（token id 确定性解析）
  - 内容锚定（val_emb / pc_emb / op_emb / sp_emb）
  - 过程监督（pc + 寄存器 + 栈指针 每步监督）
  - 四重教师强制（取指槽 / 寄存器历史 / pc / sp）+ 计划采样退火
  - STE（前向取纯嵌入，梯度走软路）

指令集（15）：mov inc dec add sub seti mul push pop jmp jz jnz call ret halt
状态 = (pc, R0..R7, sp, stack[0..15])。栈混合存值(val_emb)与返回地址(pc_emb)。
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
OPCODES = ["mov", "inc", "dec", "add", "sub", "seti", "mul", "push", "pop",
           "jmp", "jz", "jnz", "call", "ret", "halt"]
OPC = {name: i for i, name in enumerate(OPCODES)}
REG_WRITE_OPS = {OPC[o] for o in ["mov", "inc", "dec", "add", "sub", "seti", "mul", "pop"]}
STACK_PUSH_OPS = {OPC["push"], OPC["call"]}
STACK_POP_OPS = {OPC["pop"], OPC["ret"]}

OPC_OFFSET = 100
REG_OFFSET = 120
IMM_OFFSET = 140
PAD = 0
START = 200
SEP = 201

N_REG = 8
N_VAL = 16
MAX_INSTR = 48
STACK_D = 64
HALT_CLASS = MAX_INSTR


def encode_instruction(ins: Tuple) -> List[int]:
    op = ins[0]
    o = OPC[op]
    if op in ("mov", "add", "sub", "mul"):
        return [OPC_OFFSET + o, REG_OFFSET + ins[1], REG_OFFSET + ins[2]]
    if op in ("inc", "dec", "push", "pop"):
        return [OPC_OFFSET + o, REG_OFFSET + ins[1]]
    if op == "seti":
        return [OPC_OFFSET + o, REG_OFFSET + ins[1], IMM_OFFSET + ins[2]]
    if op in ("jmp", "call"):
        return [OPC_OFFSET + o, IMM_OFFSET + ins[1]]
    if op in ("jz", "jnz"):
        return [OPC_OFFSET + o, REG_OFFSET + ins[1], IMM_OFFSET + ins[2]]
    if op in ("ret", "halt"):
        return [OPC_OFFSET + o]
    raise ValueError(op)


def encode_program(program: List[Tuple]) -> List[int]:
    toks = [START, len(program)]
    for ins in program:
        toks.extend(encode_instruction(ins))
    toks.append(SEP)
    return toks


# ---------------------------------------------------------------- 程序语义
def run_program(program: List[Tuple], word: Tuple[int, ...], max_steps: int = 600):
    """返回轨迹 [(pc, regs, sp, stack), ...]（每步执行后）。halt 后 1 个冻结步即停。"""
    regs = [0] * N_REG
    for i, b in enumerate(word[:N_REG]):
        regs[i] = b
    stack = [0] * STACK_D
    sp = 0
    pc = 0
    trace = []
    for _ in range(max_steps):
        if pc < 0 or pc >= len(program):
            trace.append((HALT_CLASS, list(regs), sp, list(stack)))
            break
        ins = program[pc]
        op = ins[0]
        if op == "halt":
            trace.append((HALT_CLASS, list(regs), sp, list(stack)))
            break
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
        elif op == "mul":
            regs[ins[1]] = (regs[ins[1]] * regs[ins[2]]) % N_VAL
        elif op == "seti":
            regs[ins[1]] = ins[2]
        elif op == "push":
            stack[sp] = regs[ins[1]]
            sp += 1
        elif op == "pop":
            sp -= 1
            regs[ins[1]] = stack[sp]
        elif op == "call":
            stack[sp] = pc + 1
            sp += 1
            pc = ins[1]
            trace.append((pc, list(regs), sp, list(stack)))
            continue
        elif op == "ret":
            sp -= 1
            pc = stack[sp]
            trace.append((pc, list(regs), sp, list(stack)))
            continue
        elif op == "jmp":
            pc = ins[1]
            trace.append((pc, list(regs), sp, list(stack)))
            continue
        elif op == "jz":
            pc = ins[2] if regs[ins[1]] == 0 else pc + 1
            trace.append((pc, list(regs), sp, list(stack)))
            continue
        elif op == "jnz":
            pc = ins[2] if regs[ins[1]] != 0 else pc + 1
            trace.append((pc, list(regs), sp, list(stack)))
            continue
        pc += 1
        trace.append((pc, list(regs), sp, list(stack)))
    return trace


# ---------------------------------------------------------------- 递归程序模板
def gen_fact(rng, n=None):
    """fact(n) 递归。regs: R0=arg/result, R1=tmp。"""
    n = n if n is not None else rng.randint(1, 8)
    prog = [
        ("seti", 0, n),      # 0
        ("call", 3),         # 1
        ("halt",),           # 2
        ("jz", 0, 9),        # 3 fact: if n==0 -> base
        ("push", 0),         # 4
        ("dec", 0),          # 5
        ("call", 3),         # 6
        ("pop", 1),          # 7
        ("mul", 0, 1),       # 8 R0 = fact(n-1) * n
        ("ret",),            # 9
        ("seti", 0, 1),      # 10 base: wait jz target must be 10
        ("ret",),            # 11
    ]
    # 修正 jz 目标：base 在 10
    prog[3] = ("jz", 0, 10)
    return prog


def gen_sum(rng, n=None):
    """sum_to(n) 递归。"""
    n = n if n is not None else rng.randint(1, 8)
    prog = [
        ("seti", 0, n),      # 0
        ("call", 3),         # 1
        ("halt",),           # 2
        ("jz", 0, 10),       # 3 sum:
        ("push", 0),         # 4
        ("dec", 0),          # 5
        ("call", 3),         # 6
        ("pop", 1),          # 7
        ("add", 0, 1),       # 8 R0 = sum(n-1) + n
        ("ret",),            # 9
        ("seti", 0, 0),      # 10 base
        ("ret",),            # 11
    ]
    return prog


def gen_fib(rng, n=None):
    """fib(n) 递归。regs: R0=arg/result, R1=tmp, R3=const1, R4=tmp2。"""
    n = n if n is not None else rng.randint(1, 5)
    prog = [
        ("seti", 0, n),      # 0
        ("call", 3),         # 1
        ("halt",),           # 2
        ("jz", 0, 20),       # 3 fib: if n==0 -> F0
        ("seti", 3, 1),      # 4
        ("mov", 4, 0),       # 5
        ("sub", 4, 3),       # 6 R4 = n-1
        ("jz", 4, 22),       # 7 if n==1 -> F1
        ("push", 0),         # 8 save n
        ("dec", 0),          # 9
        ("call", 3),         # 10 R0 = fib(n-1)
        ("pop", 1),          # 11 R1 = n
        ("push", 0),         # 12 save fib(n-1)
        ("mov", 0, 1),       # 13
        ("dec", 0),          # 14
        ("dec", 0),          # 15 R0 = n-2
        ("call", 3),         # 16 R0 = fib(n-2)
        ("pop", 1),          # 17 R1 = fib(n-1)
        ("add", 0, 1),       # 18
        ("ret",),            # 19
        ("seti", 0, 0),      # 20 F0
        ("ret",),            # 21
        ("seti", 0, 1),      # 22 F1
        ("ret",),            # 23
    ]
    return prog


def gen_countdown(rng, n=None):
    """迭代倒数（对照：无递归）。"""
    n = n if n is not None else rng.randint(1, 8)
    prog = [
        ("seti", 0, n),
        ("jz", 0, 4),
        ("dec", 0),
        ("jmp", 1),
        ("halt",),
    ]
    return prog


def gen_randarith(rng, n=None):
    """随机直线程序：覆盖 0-15 全值域 × 所有操作（含 push/pop 栈传值）。"""
    n_instr = rng.randint(4, 10)
    prog = []
    for _ in range(n_instr):
        r = rng.random()
        if r < 0.25:
            # push/pop 对：seti Ra,v; push Ra; ...; pop Rb（栈传值，覆盖栈操作×全值域）
            ra, rb = rng.randrange(N_REG), rng.randrange(N_REG)
            prog.append(("seti", ra, rng.randrange(N_VAL)))
            prog.append(("push", ra))
            prog.append(("pop", rb))
        else:
            op = rng.choice(["mov", "inc", "dec", "add", "sub", "mul", "seti", "seti"])
            if op in ("mov", "add", "sub", "mul"):
                prog.append((op, rng.randrange(N_REG), rng.randrange(N_REG)))
            elif op in ("inc", "dec"):
                prog.append((op, rng.randrange(N_REG)))
            else:
                prog.append(("seti", rng.randrange(N_REG), rng.randrange(N_VAL)))
    prog.append(("halt",))
    return prog


REC_GENERATORS = {"fact": gen_fact, "sum": gen_sum, "fib": gen_fib,
                  "countdown": gen_countdown, "randarith": gen_randarith}


def gen_rec_mixed(rng, n_instr=None):
    """混合递归模板（每批内由调用方控制模板）。"""
    name = rng.choice(["fact", "sum", "fib", "countdown"])
    return REC_GENERATORS[name](rng)


# ---------------------------------------------------------------- 批量构造
def parse_slots(instr_tok):
    op = (instr_tok[:, :, 0] - OPC_OFFSET).clamp(0, len(OPCODES) - 1)
    reg_a = (instr_tok[:, :, 1] - REG_OFFSET).clamp(0, N_REG - 1)
    t2 = instr_tok[:, :, 2]
    reg_b = (t2 - REG_OFFSET).clamp(0, N_REG - 1)
    imm = (t2 - IMM_OFFSET).clamp(0, N_VAL - 1)
    return op, reg_a, reg_b, imm


def build_batch(samples, device="cuda", max_instr=MAX_INSTR):
    """samples: [(program, word)]。CPU 构建，一次性传输。"""
    B = len(samples)
    instr_tok_l, slot_mask_l, words_l = [], [], []
    traces, progs = [], []
    for prog, word in samples:
        assert len(prog) <= max_instr
        toks = encode_program(prog)
        positions = []
        idx = 2
        for ins in prog:
            positions.append(idx)
            idx += len(encode_instruction(ins))
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

    fetch_l, pc_l, reg_l, reghist_l = [], [], [], []
    sp_l, sphist_l, wvm_l, wvt_l = [], [], [], []
    for i, tr in enumerate(traces):
        prev_pc = 0
        prev_regs = list(words_l[i])
        prev_sp = 0
        f_t, p_t, r_t, rh_t, s_t, sh_t, m_t, v_t = [], [], [], [], [], [], [], []
        for t in range(T):
            halted = prev_pc >= len(progs[i]) or prev_pc < 0
            f_t.append(prev_pc if not halted else HALT_CLASS)
            p_t.append(tr[t][0])
            r_t.append(list(tr[t][1]))
            rh_t.append(list(prev_regs))
            s_t.append(tr[t][2])
            sh_t.append(prev_sp)
            if not halted:
                ins = progs[i][prev_pc]
                if OPC[ins[0]] in REG_WRITE_OPS:
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
            prev_sp = tr[t][2]
        fetch_l.append(f_t)
        pc_l.append(p_t)
        reg_l.append(r_t)
        reghist_l.append(rh_t)
        sp_l.append(s_t)
        sphist_l.append(sh_t)
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
        "reg_history": tt(reghist_l, torch.long),
        "sp_target": tt(sp_l, torch.long),
        "sp_history": tt(sphist_l, torch.long),
        "wv_mask": tt(wvm_l, torch.bool),
        "wv_target": tt(wvt_l, torch.long),
        "T": T,
    }


# ---------------------------------------------------------------- 模型
@dataclass
class RecExecutorConfig:
    d_model: int = 256
    max_instr: int = MAX_INSTR
    stack_d: int = STACK_D
    n_reg: int = N_REG
    n_val: int = N_VAL
    attention_weight: float = 0.5
    wv_weight: float = 1.0


class RecursiveExecutor(nn.Module):
    def __init__(self, cfg: RecExecutorConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_emb = nn.Embedding(256, d)
        self.val_emb = nn.Embedding(cfg.n_val, d)
        self.pc_emb = nn.Embedding(cfg.max_instr + 1, d)
        self.op_emb = nn.Embedding(len(OPCODES), d)
        self.k_pc = nn.Linear(d, d)
        self.q_pc = nn.Linear(d, d)
        # 语义输入：op, pc, val(reg_a), val(reg_b), imm, 栈顶 = 6d
        # （sp 机械化更新，不作为学习输入——深度泛化的关键）
        self.compute = nn.Sequential(
            nn.Linear(6 * d, 2 * d), nn.GELU(),
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

    def forward(self, instr_tok, slot_mask, words, T, fetch_teacher=None, reg_hist=None,
                teacher_prob=1.0):
        """sp 机械化：由解析出的 op 决定 ±1/0，不预测不嵌入（深度泛化自由）。
        返回 pc_logits, wv_logits, sp 轨迹(记录用), attn_logits, R 轨迹, S 轨迹。"""
        B, S, _ = instr_tok.shape
        dev = instr_tok.device
        keys = self.k_pc(self.pc_emb.weight[:S])
        op_s, rega_s, regb_s, imm_s = parse_slots(instr_tok)
        imm_emb_s = self.val_emb(imm_s)

        R = self.val_emb(words)
        S_stack = torch.zeros(B, self.cfg.stack_d, self.cfg.d_model, device=dev)
        p = self.pc_emb(torch.zeros(B, dtype=torch.long, device=dev))
        sp_idx = torch.zeros(B, dtype=torch.long, device=dev)
        batch_idx = torch.arange(B, device=dev)

        pc_l, wv_l, sp_l, attn_l, R_l, S_l = [], [], [], [], [], []
        halted = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(T):
            p_read = p
            R_read = R
            if reg_hist is not None:
                use_hist = torch.rand(B, device=dev) < teacher_prob
                if fetch_teacher is not None:
                    pc_teacher = self.pc_emb(fetch_teacher[:, t].clamp(0, MAX_INSTR))
                    p_read = torch.where(use_hist.view(B, 1), pc_teacher, p)
                R_truth = self.val_emb(reg_hist[:, t])
                R_read = torch.where(use_hist.view(B, 1, 1), R_truth, R)

            q = self.q_pc(p_read)
            scores = q @ keys.t()
            scores = scores.masked_fill(~slot_mask, float("-inf"))
            attn_l.append(scores)

            slot_auto = scores.argmax(-1)
            if fetch_teacher is not None:
                slot = torch.where(use_hist, fetch_teacher[:, t].clamp(0, S - 1), slot_auto)
            else:
                slot = slot_auto

            op = op_s[batch_idx, slot]
            reg_a = rega_s[batch_idx, slot]
            reg_b = regb_s[batch_idx, slot]
            imm_e = imm_emb_s[batch_idx, slot]

            va = R_read[batch_idx, reg_a]
            vb = R_read[batch_idx, reg_b]
            top_idx = (sp_idx - 1).clamp(0, self.cfg.stack_d - 1)
            top = S_stack[batch_idx, top_idx] * (sp_idx > 0).float().unsqueeze(-1)

            h = self.compute(torch.cat([self.op_emb(op), p_read, va, vb, imm_e, top], dim=-1))
            wv_logits = self.wv_head(h)
            pc_logits = self.pc_head(h)
            wv_l.append(wv_logits)
            pc_l.append(pc_logits)

            is_pop = (op == OPC["pop"])
            is_push = (op == OPC["push"])
            is_call = (op == OPC["call"])
            is_ret = (op == OPC["ret"])

            # 寄存器写回（STE）：写 op（mov..mul）+ pop
            write_mask = (((op < 7) | is_pop) & ~halted).float().unsqueeze(-1)
            wv_soft = torch.softmax(wv_logits, dim=-1) @ self.val_emb.weight
            wv_hard = self.val_emb(wv_logits.argmax(-1))
            new_val_repr = wv_hard + (wv_soft - wv_soft.detach())
            R_new = R.clone()
            R_new[batch_idx, reg_a] = R[batch_idx, reg_a] * (1 - write_mask) + new_val_repr * write_mask
            R = R_new

            # 栈写（锚定）：push → R[reg_a]；call → pc_emb(slot+1)；位置 = sp_idx（机械化）
            do_push = ((is_push | is_call) & ~halted).float().unsqueeze(-1)
            push_content = torch.where(is_call.unsqueeze(-1),
                                       self.pc_emb((slot + 1).clamp(0, MAX_INSTR)), va)
            S_new = S_stack.clone()
            wpos = sp_idx.clamp(0, self.cfg.stack_d - 1)
            cur_slot = S_stack[batch_idx, wpos]
            S_new[batch_idx, wpos] = cur_slot * (1 - do_push) + push_content * do_push
            S_stack = S_new

            # sp 机械化更新（由 op 语法决定，无需学习）
            delta = torch.zeros(B, dtype=torch.long, device=dev)
            delta = delta + ((is_push | is_call) & ~halted).long()
            delta = delta - ((is_pop | is_ret) & ~halted).long()
            sp_idx = (sp_idx + delta).clamp(0, self.cfg.stack_d)

            pc_soft = torch.softmax(pc_logits, dim=-1) @ self.pc_emb.weight
            pc_hard_idx = pc_logits.argmax(-1)
            pc_hard = self.pc_emb(pc_hard_idx)
            p = pc_hard + (pc_soft - pc_soft.detach())
            halted = halted | (pc_hard_idx == HALT_CLASS)

            sp_l.append(sp_idx.clone())
            R_l.append(R)
            S_l.append(S_stack)

        return (torch.stack(pc_l, 1), torch.stack(wv_l, 1), torch.stack(sp_l, 1),
                torch.stack(attn_l, 1), torch.stack(R_l, 1), torch.stack(S_l, 1))

    def decode_regs(self, R):
        sim = R @ self.val_emb.weight.t()
        return sim.argmax(-1)

    def decode_pc(self, p):
        sim = p @ self.pc_emb.weight.t()
        return sim.argmax(-1)


# ---------------------------------------------------------------- 评测
@torch.no_grad()
def evaluate(model, gen_name: str, n: int = 50, seed: int = 42, device: str = "cuda",
             n_range: Tuple[int, int] = None):
    model.eval()
    rng = random.Random(seed)
    gen = REC_GENERATORS[gen_name]
    if n_range is None:
        n_range = {"fact": (1, 8), "sum": (1, 8), "countdown": (1, 8), "fib": (1, 5)}.get(gen_name, (1, 8))
    exact = pc_hits = reg_hits = sp_hits = cnt = 0
    for _ in range(n):
        n_val = rng.randint(*n_range) if n_range else None
        prog = gen(rng, n_val)
        word = tuple(rng.randrange(2) for _ in range(6))
        batch = build_batch([(prog, word)], device=device)
        pc_l, wv_l, sp_l, attn_l, R_l, S_l = model(
            batch["instr_tok"], batch["slot_mask"], batch["words"], batch["T"])
        pred_pc = pc_l[0].argmax(-1).cpu()
        pred_sp = sp_l[0].cpu()
        pred_reg = model.decode_regs(R_l)[0].cpu()
        truth = run_program(prog, word)
        T = batch["T"]
        ok = True
        for t in range(T):
            tt_pc, tt_regs, tt_sp, _ = truth[t]
            pc_ok = pred_pc[t].item() == tt_pc
            reg_ok = all(pred_reg[t][r].item() == tt_regs[r] for r in range(N_REG))
            sp_ok = pred_sp[t].item() == tt_sp
            pc_hits += pc_ok
            reg_hits += reg_ok
            sp_hits += sp_ok
            cnt += 1
            if not (pc_ok and reg_ok and sp_ok):
                ok = False
        exact += ok
    model.train()
    return {"gen": gen_name, "n": n, "exact": exact / n,
            "pc_acc": pc_hits / cnt, "reg_acc": reg_hits / cnt, "sp_acc": sp_hits / cnt}


# ---------------------------------------------------------------- 训练
class RecTrainer:
    def __init__(self, model: RecursiveExecutor, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device

    def train(self, steps: int, templates: List[str] = None, batch_size: int = 96,
              n_range: Tuple[int, int] = (1, 8), lr: float = 3e-4, warmup: int = 200,
              log_every: int = 100, eval_every: int = 500,
              output_dir: str = "./checkpoints/recursive_executor", seed: int = 42):
        os.makedirs(output_dir, exist_ok=True)
        rng = random.Random(seed)
        torch.manual_seed(seed)
        if templates is None:
            templates = ["fact", "sum", "countdown"]
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

        def lr_lambda(s):
            if s < warmup:
                return s / max(1, warmup)
            return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        cfg = self.model.cfg
        metrics = []
        nan_count = 0
        t0 = time.time()
        TPL_N = {"fact": (1, 10), "sum": (1, 10), "countdown": (1, 10), "fib": (1, 6),
                 "randarith": (1, 1)}
        for step in range(steps):
            tpl = templates[step % len(templates)]  # 每批同一模板（控制 T 方差）
            lo, hi = TPL_N.get(tpl, n_range)
            samples = [(REC_GENERATORS[tpl](rng, rng.randint(lo, hi)),
                        tuple(rng.randrange(2) for _ in range(6)))
                       for _ in range(batch_size)]
            batch = build_batch(samples, device=self.device)

            # 教师强制：前 40% 全程，40-70% 退火到 0.3，之后保持 0.3 下限（防长轨迹暴露崩溃）
            frac = step / max(1, steps)
            if frac < 0.4:
                teacher_prob = 1.0
            elif frac < 0.7:
                teacher_prob = 1.0 - 0.7 * (frac - 0.4) / 0.3
            else:
                teacher_prob = 0.3

            pc_l, wv_l, sp_trace, attn_l, R_l, S_l = self.model(
                batch["instr_tok"], batch["slot_mask"], batch["words"], batch["T"],
                fetch_teacher=batch["fetch_target"], reg_hist=batch["reg_history"],
                teacher_prob=teacher_prob)

            opt.zero_grad()
            pc_loss = F.cross_entropy(pc_l.transpose(1, 2), batch["pc_target"])
            sp_loss = torch.zeros((), device=self.device)  # sp 机械化，无损失
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
            if not torch.isfinite(loss):
                nan_count += 1
                print(f"  step {step+1}: NaN/Inf loss (#{nan_count}), rollback + halve LR", flush=True)
                last_ckpt = os.path.join(output_dir, "last_good.pt")
                if os.path.exists(last_ckpt):
                    self.model.load_state_dict(torch.load(last_ckpt, map_location=self.device))
                else:
                    # 无可用 checkpoint（训练初期）：重新初始化权重
                    self.model.apply(lambda m: m.reset_parameters()
                                     if hasattr(m, "reset_parameters") else None)
                for g in opt.param_groups:
                    g["lr"] = max(g["lr"] * 0.5, 1e-5)
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            if (step + 1) % 200 == 0:
                torch.save(self.model.state_dict(), os.path.join(output_dir, "last_good.pt"))

            if (step + 1) % log_every == 0 or step == 0:
                print(f"  step {step+1}/{steps} [{tpl}] | loss={loss.item():.3f} "
                      f"pc={pc_loss.item():.3f} sp={sp_loss.item():.3f} wv={wv_loss.item():.3f} "
                      f"attn={attn_loss.item():.3f} | {time.time()-t0:.0f}s", flush=True)
                t0 = time.time()
            if (step + 1) % eval_every == 0:
                for tpl in templates:
                    res = evaluate(self.model, tpl, n=30, seed=65000 + step, device=self.device,
                                   n_range=n_range)
                    print(f"    EVAL {tpl}: exact={res['exact']:.3f} pc={res['pc_acc']:.3f} "
                          f"reg={res['reg_acc']:.3f} sp={res['sp_acc']:.3f}", flush=True)
                    metrics.append({"step": step + 1, **res})
                torch.save(self.model.state_dict(), os.path.join(output_dir, f"step{step+1}.pt"))
                with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)

        torch.save(self.model.state_dict(), os.path.join(output_dir, "final.pt"))
        print("=== Training complete ===")
        return metrics


if __name__ == "__main__":
    trainer = RecTrainer(RecursiveExecutor(RecExecutorConfig(d_model=256)))
    print("params:", round(trainer.model.count_params() / 1e6, 2), "M")
    trainer.train(steps=500, templates=["fact"], batch_size=96,
                  output_dir="./checkpoints/recursive_smoke")
