# -*- coding: utf-8 -*-
"""A2 数据：随机 Python 子集程序生成器（编译后供训练/评测）。"""
import random

from nsm.py_subset_compiler import compile_python
from nsm.recursive_executor import run_program


def gen_recursive_src(rng: random.Random) -> str:
    """随机递归函数：fact/sum 家族（单递归）+ fib 家族（双递归），随机常数/变量名/结构扰动。"""
    fname = rng.choice(["f", "g", "rec", "solve"])
    arg = rng.choice(["n", "x", "k", "m"])
    n_init = rng.randint(1, 8)
    double = rng.random() < 0.35

    if double:
        b0 = rng.randint(0, 2)
        b1 = rng.randint(1, 3)
        d1 = 1
        d2 = 2
        src = f"""
def {fname}({arg}):
    if {arg} == 0:
        return {b0}
    if {arg} == 1:
        return {b1}
    return {fname}({arg} - {d1}) + {fname}({arg} - {d2})

result = {fname}({n_init})
"""
        return src

    base_val = rng.randint(0, 3)
    op = rng.choice(["*", "+"])
    dec = rng.randint(1, 2)
    order = rng.random() < 0.5
    ret_expr = f"{arg} {op} {fname}({arg} - {dec})" if order else f"{fname}({arg} - {dec}) {op} {arg}"
    dead = ""
    if rng.random() < 0.3:
        dead = f"\n    tmp = {rng.randint(0, 15)}"
    src = f"""
def {fname}({arg}):
    if {arg} == 0:
        return {base_val}{dead}
    return {ret_expr}

result = {fname}({n_init})
"""
    return src


def gen_iterative_src(rng: random.Random) -> str:
    """随机迭代函数：while 循环家族。"""
    fname = rng.choice(["f", "g", "proc"])
    arg = rng.choice(["n", "x", "k"])
    acc = rng.choice(["s", "acc", "total", "r"])
    acc_init = rng.randint(0, 3)
    op = rng.choice(["+", "*"])
    n_init = rng.randint(1, 8)
    body = f"        {acc} = {acc} {op} {arg}"
    src = f"""
def {fname}({arg}):
    {acc} = {acc_init}
    while {arg} != 0:
{body}
        {arg} = {arg} - 1
    return {acc}

result = {fname}({n_init})
"""
    return src


def gen_straight_src(rng: random.Random) -> str:
    """随机直线程序：赋值 + 算术。"""
    n_stmt = rng.randint(2, 6)
    lines = []
    vars_pool = ["a", "b", "c", "d"]
    for i in range(n_stmt):
        v = vars_pool[i]
        if i == 0 or rng.random() < 0.4:
            lines.append(f"{v} = {rng.randint(0, 15)}")
        else:
            prev = rng.choice(vars_pool[:i])
            op = rng.choice(["+", "-", "*"])
            if rng.random() < 0.5:
                lines.append(f"{v} = {prev} {op} {rng.randint(0, 15)}")
            else:
                prev2 = rng.choice(vars_pool[:i])
                lines.append(f"{v} = {prev} {op} {prev2}")
    lines.append(f"result = {vars_pool[rng.randrange(min(n_stmt, 4))]}")
    return "\n".join(lines) + "\n"


def gen_compiled(rng: random.Random, kind: str = None):
    """生成 (程序指令列表, 真值轨迹)。失败时重试。"""
    kind = kind or rng.choice(["recursive", "iterative", "straight"])
    for _ in range(20):
        try:
            if kind == "recursive":
                src = gen_recursive_src(rng)
            elif kind == "iterative":
                src = gen_iterative_src(rng)
            else:
                src = gen_straight_src(rng)
            prog = compile_python(src)
            if len(prog) > 44:
                continue
            tr = run_program(prog, (0,) * 6)
            if len(tr) > 320:
                continue
            return prog, src
        except Exception:
            continue
    return None, None


if __name__ == "__main__":
    rng = random.Random(0)
    for kind in ["recursive", "iterative", "straight"]:
        prog, src = gen_compiled(rng, kind)
        print(f"===== {kind} =====")
        print(src)
        print(f"-> {len(prog)} instructions")
