# -*- coding: utf-8 -*-
"""NL 中间层实验：自然语言 DFA 规格 → LLM 解析 → 结构化 spec → Executor/算法执行。
对比条件：
  A 纯 LLM 直接执行（NL 规则 + 字符串 → 逐位状态）
  B 纯 LLM CoT 执行
  C LLM 解析 → 确定性算法执行
  D LLM 解析 → Neural DFA Executor 执行
"""
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from nsm.data import make_balanced_dfa
from nsm.dfa_executor import NeuralDFAExecutor, ExecutorConfig, build_executor_batch


# ============================================================
# NL 描述生成
# ============================================================

STATE_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H"]

MOVE_SYNONYMS = ["goes to", "moves to", "jumps to", "enters", "becomes", "transitions to"]
IF_PHRASES = ["if", "when", "whenever", "upon"]


def nl_describe(dfa, template: int, seed: int) -> str:
    rng = random.Random(seed)
    k = dfa["k"]
    names = STATE_NAMES[:k]
    accepting = [names[q] for q in dfa["accepting"]]
    trans = [(names[q], sym, names[nxt]) for q in range(k) for sym in range(2)
             for nxt in [dfa["delta"][(q, sym)]]]

    if template == 1:
        # 表格式
        lines = [
            f"There are {k} states: {', '.join(names)}.",
            f"The start state is {names[dfa['start']]}.",
            f"The accepting states are {', '.join(accepting)}.",
            "The transition rules are:",
        ]
        for q, sym, nxt in trans:
            lines.append(f"  {q} -{sym}-> {nxt}")
        return "\n".join(lines)

    if template == 2:
        # 规则式：每条 "In state X, on symbol 0, go to Y; on symbol 1, go to Z."
        lines = [
            f"There are {k} states: {', '.join(names)}.",
            f"Start: {names[dfa['start']]}. Accepting: {', '.join(accepting)}.",
        ]
        for q in range(k):
            n0 = names[dfa["delta"][(q, 0)]]
            n1 = names[dfa["delta"][(q, 1)]]
            lines.append(f"In state {names[q]}, on symbol 0 go to {n0}, on symbol 1 go to {n1}.")
        return "\n".join(lines)

    # template 3: 打乱顺序 + 同义词 + 从句式
    rng.shuffle(trans)
    lines = [
        f"We have a machine with {k} states named {', '.join(names)}.",
        f"It starts in {names[dfa['start']]} and accepts in {', '.join(accepting)}.",
    ]
    for q, sym, nxt in trans:
        move = rng.choice(MOVE_SYNONYMS)
        phrase = rng.choice(IF_PHRASES)
        lines.append(f"{phrase} we are in {q} and see symbol {sym}, it {move} {nxt}.")
    return "\n".join(lines)


# ============================================================
# LLM 调用（Ollama）
# ============================================================

def ollama_complete(prompt: str, model: str = "qwen2.5:7b",
                    temperature: float = 0.0, max_tokens: int = 1024) -> str:
    import urllib.request
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }).encode("utf-8")
    req = urllib.request.Request("http://localhost:11434/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "")


PARSE_PROMPT = """You are a formalizer. Translate the following natural-language description of a finite automaton into a strict JSON object.

State names map to integers by their ORDER OF FIRST APPEARANCE in the description: the first state name mentioned becomes 0, the second becomes 1, etc.

Output JSON format (nothing else, no markdown):
{"k": <number of states>, "start": <int>, "accepting": [<ints>], "delta": [[q, symbol, next_q], ...]}

The description:
---
{description}
---
JSON:"""


def make_parse_prompt(nl: str) -> str:
    return PARSE_PROMPT.replace("{description}", nl)


FEW_SHOT_EXAMPLE_NL = """We have a machine with 2 states named A, B.
It starts in A and accepts in B.
In state A, on symbol 0 go to B, on symbol 1 go to A.
In state B, on symbol 0 go to A, on symbol 1 go to B."""

FEW_SHOT_EXAMPLE_JSON = """{"k": 2, "start": 0, "accepting": [1], "delta": [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]}"""

FEW_SHOT_PREAMBLE = """You are a formalizer. Translate natural-language descriptions of finite automata into strict JSON.

Example:
Description:
---
{example_nl}
---
JSON: {example_json}

State names map to integers by ORDER OF FIRST APPEARANCE: first mentioned becomes 0, second becomes 1, etc.
Output format (nothing else, no markdown):
{"k": <int>, "start": <int>, "accepting": [<ints>], "delta": [[q, symbol, next_q], ...]}

Description:
---
{description}
---
JSON:"""


def make_parse_prompt_fewshot(nl: str) -> str:
    return FEW_SHOT_PREAMBLE.replace("{example_nl}", FEW_SHOT_EXAMPLE_NL) \
                            .replace("{example_json}", FEW_SHOT_EXAMPLE_JSON) \
                            .replace("{description}", nl)


# T3 同构示例：打乱顺序 + 同义词 + 从句式
FEW_SHOT_T3_NL = """We have a machine with 2 states named A, B.
It starts in A and accepts in B.
when we are in B and see symbol 1, it jumps to B.
if we are in A and see symbol 0, it enters B.
whenever we are in A and see symbol 1, it goes to A.
upon we are in B and see symbol 0, it transitions to A."""

FEW_SHOT_T3_JSON = """{"k": 2, "start": 0, "accepting": [1], "delta": [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]]}"""


def make_parse_prompt_fewshot_t3(nl: str) -> str:
    return FEW_SHOT_PREAMBLE.replace("{example_nl}", FEW_SHOT_T3_NL) \
                            .replace("{example_json}", FEW_SHOT_T3_JSON) \
                            .replace("{description}", nl)




DIRECT_PROMPT = """Here is a finite automaton described in natural language:
---
{nl}
---
Then the machine reads a binary string from left to right, one symbol at a time.

Now answer: the machine starts in its start state. After reading each symbol, what state is it in?

Binary string: {bits}

Output ONLY the state sequence after each symbol, one state name per line, in order, nothing else.
States:"""

COT_PROMPT = """Here is a finite automaton described in natural language:
---
{nl}
---
Binary string: {bits}

Work through the string step by step. For each position, write "pos i: state X -> on symbol b -> state Y". Then on the final line, write "FINAL: <state after each symbol as a list>" with one state per position in order, separated by commas. States:"""


# ============================================================
# 解析与评测
# ============================================================

def parse_spec(text: str) -> Optional[dict]:
    """从 LLM 输出解析 JSON spec。"""
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    if "delta" not in data or "k" not in data or "accepting" not in data:
        return None
    try:
        delta = {}
        for q, sym, nxt in data["delta"]:
            delta[(int(q), int(sym))] = int(nxt)
        spec = {
            "k": int(data["k"]),
            "start": int(data.get("start", 0)),
            "accepting": [int(a) for a in data["accepting"]],
            "delta": delta,
        }
        # 校验完整性
        if any((q, s) not in delta for q in range(spec["k"]) for s in range(2)):
            return None
        if spec["start"] >= spec["k"] or any(a >= spec["k"] for a in spec["accepting"]):
            return None
        if any(nxt >= spec["k"] for nxt in delta.values()):
            return None
        if not spec["accepting"]:
            return None
        return spec
    except Exception:
        return None


def state_seq_from_llm(text: str, k: int, L: int, names: List[str]) -> Optional[List[int]]:
    """从 LLM 直接/CoT 输出解析状态序列。接受状态名（A,B...）或编号（0,1...）。"""
    tokens = re.findall(r'[A-Ha-h]|\b\d+\b', text)
    seq = []
    for t in tokens:
        if t.upper() in names:
            seq.append(names.index(t.upper()))
        elif t.isdigit():
            v = int(t)
            if 0 <= v < k:
                seq.append(v)
        if len(seq) == L:
            return seq
    return seq if len(seq) == L else None


def truth_states(spec, word) -> List[int]:
    q = spec["start"]
    out = []
    for sym in word:
        if (q, sym) not in spec["delta"]:
            return []
        q = spec["delta"][(q, sym)]
        out.append(q)
    return out


def spec_to_dfa(spec) -> dict:
    return {"k": spec["k"], "m": 2, "start": spec["start"],
            "accepting": sorted(spec["accepting"]), "delta": spec["delta"]}


def run_executor(model: NeuralDFAExecutor, spec, word, device='cuda'):
    """用 Executor 执行（需要 spec 转成结构化 token 输入，复用 build_executor_batch 的 dfa 格式）。"""
    dfa = spec_to_dfa(spec)
    batch = [(dfa, tuple(word))]
    inp = build_executor_batch(batch, device=device)
    sl, vl, _ = model(inp["trans_tokens"], inp["nxt_tokens"], inp["trans_mask"], inp["sym_ids"])
    k = spec["k"]
    masked = sl[0].clone()
    masked[:, k:] = float('-inf')
    preds = masked.argmax(-1).cpu().tolist()
    return preds


def eval_condition_A(spec, word, names, nl_desc) -> Tuple[bool, Optional[List[int]]]:
    """纯 LLM 直接。"""
    bits = " ".join(str(b) for b in word)
    out = ollama_complete(DIRECT_PROMPT.format(nl=nl_desc, bits=bits))
    seq = state_seq_from_llm(out, spec["k"], len(word), names)
    if seq is None:
        return False, None
    true = truth_states(spec, word)
    return seq == true, seq


def eval_condition_B(spec, word, names, nl_desc) -> Tuple[bool, Optional[List[int]]]:
    """纯 LLM CoT。"""
    bits = " ".join(str(b) for b in word)
    out = ollama_complete(COT_PROMPT.format(nl=nl_desc, bits=bits), max_tokens=2048)
    seq = state_seq_from_llm(out, spec["k"], len(word), names)
    if seq is None:
        return False, None
    true = truth_states(spec, word)
    return seq == true, seq


def eval_condition_C(spec, word, parsed_spec) -> Tuple[bool, Optional[List[int]]]:
    """LLM 解析 → 确定性算法。"""
    if parsed_spec is None:
        return False, None
    true = truth_states(spec, word)
    pred = truth_states(parsed_spec, word)
    return pred == true, pred


def eval_condition_D(spec, word, parsed_spec, model, device='cuda') -> Tuple[bool, Optional[List[int]]]:
    """LLM 解析 → Executor。"""
    if parsed_spec is None:
        return False, None
    true = truth_states(spec, word)
    pred = run_executor(model, parsed_spec, word, device=device)
    return pred == true, pred


def permutation_repair(spec: dict) -> dict:
    """利用先验：每个符号的转移列必须是 [0..k-1] 的一个排列。
    若某列有缺失/重复，把重复位置替换为缺失值（最小编辑修复）。"""
    k = spec["k"]
    for sym in range(2):
        col = [spec["delta"][(q, sym)] for q in range(k)]
        present = set(col)
        missing = [q for q in range(k) if q not in present]
        if not missing:
            continue
        dup_positions = [q for q in range(k) if col.count(col[q]) > 1 and col[q] not in missing]
        # 把重复的列值换成缺失值
        for pos, m in zip(dup_positions, missing):
            spec["delta"][(pos, sym)] = m
        # 兜底：仍有缺失则按顺序填
        for pos in range(k):
            col2 = [spec["delta"][(q, sym)] for q in range(k)]
            if pos in [q for q in range(k) if col2.count(col2[q]) > 1]:
                miss2 = [q for q in range(k) if q not in set(col2)]
                if miss2:
                    spec["delta"][(pos, sym)] = miss2[0]
    return spec


def parse_single(nl, model="qwen2.5:7b", fewshot: bool = False) -> Optional[dict]:
    """单次解析。"""
    prompt = make_parse_prompt_fewshot(nl) if fewshot else make_parse_prompt(nl)
    out = ollama_complete(prompt, model=model)
    return parse_spec(out)


def parse_with_vote(nl, model="qwen2.5:7b", n_votes: int = 3, temperature: float = 0.3,
                    fewshot: bool = False) -> Optional[dict]:
    """多次采样解析 + 逐条转移投票。"""
    votes = []
    for _ in range(n_votes):
        prompt = make_parse_prompt_fewshot(nl) if fewshot else make_parse_prompt(nl)
        out = ollama_complete(prompt, model=model, temperature=temperature)
        spec = parse_spec(out)
        if spec is not None:
            votes.append(spec)
    if not votes:
        return None
    # 多数投票：k/start 取众数，delta 逐条目投票，accepting 投票
    from collections import Counter
    k = Counter(v["k"] for v in votes).most_common(1)[0][0]
    start = Counter(v["start"] for v in votes).most_common(1)[0][0]
    accept_counter = Counter(tuple(sorted(v["accepting"])) for v in votes if v["k"] == k)
    accepting = list(accept_counter.most_common(1)[0][0])
    delta = {}
    for q in range(k):
        for s in range(2):
            c = Counter(v["delta"].get((q, s)) for v in votes if v["k"] == k)
            delta[(q, s)] = c.most_common(1)[0][0]
    return {"k": k, "start": start, "accepting": accepting, "delta": delta}


def spec_matches(spec, parsed_spec) -> bool:
    """解析出的 spec 与 ground truth 完全一致。"""
    if parsed_spec is None:
        return False
    return (spec["k"] == parsed_spec["k"]
            and spec["start"] == parsed_spec["start"]
            and set(spec["accepting"]) == set(parsed_spec["accepting"])
            and all(spec["delta"][(q, s)] == parsed_spec["delta"].get((q, s))
                    for q in range(spec["k"]) for s in range(2)))


def main():
    device = 'cuda'
    executor = NeuralDFAExecutor(ExecutorConfig(d_model=256, max_k=64, max_trans=128))
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "essential",
                             "dfa_executor_k32_final.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "essential",
                                 "dfa_executor_final_final.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    executor.load_state_dict(ckpt)
    executor.to(device)
    executor.eval()

    rng = random.Random(7)
    n_samples = 6
    samples = []
    for i in range(n_samples):
        dfa = make_balanced_dfa(4, 2, 2, seed=rng.randint(0, 2**31))
        L = 8
        word = tuple(rng.randrange(2) for _ in range(L))
        spec = {"k": dfa["k"], "start": dfa["start"], "accepting": dfa["accepting"], "delta": dfa["delta"]}
        samples.append((spec, word, STATE_NAMES[:4]))

    for template in [1, 2, 3]:
        print(f"\n===== Template {template} =====")
        stats = {c: {"ok": 0, "spec_ok": 0, "acc": 0} for c in ["A", "B", "C", "D"]}
        for si, (spec, word, names) in enumerate(samples):
            nl_desc = nl_describe(spec_to_dfa(spec), template, seed=100 + si)
            print(f"\n-- sample {si} (L={len(word)}) --")
            print(nl_desc[:200].replace("\n", " | "))

            out = ollama_complete(make_parse_prompt(nl_desc))
            parsed = parse_spec(out)
            sm = spec_matches(spec, parsed)
            print(f"  parse_ok={parsed is not None} spec_match={sm}")
            if parsed is not None and not sm:
                print(f"  GT k={spec['k']} start={spec['start']} acc={spec['accepting']}")
                print(f"  PD k={parsed['k']} start={parsed['start']} acc={parsed['accepting']}")

            for cond in ["A", "B"]:
                ok, seq = (eval_condition_A if cond == "A" else eval_condition_B)(spec, word, names, nl_desc)
                stats[cond]["ok"] += 1
                stats[cond]["acc"] += ok
                print(f"  {cond}: seq_ok={seq is not None} acc={ok}")

            for cond, fn in [("C", eval_condition_C), ("D", eval_condition_D)]:
                if cond == "C":
                    ok, seq = eval_condition_C(spec, word, parsed)
                else:
                    ok, seq = eval_condition_D(spec, word, parsed, executor, device)
                stats[cond]["ok"] += 1
                stats[cond]["acc"] += ok
                print(f"  {cond}: spec_ok={parsed is not None} acc={ok}")

        print(f"\n-- Template {template} summary --")
        for c in ["A", "B", "C", "D"]:
            print(f"  {c}: {stats[c]['acc']}/{n_samples} = {stats[c]['acc']/n_samples:.2f}")


if __name__ == "__main__":
    main()
