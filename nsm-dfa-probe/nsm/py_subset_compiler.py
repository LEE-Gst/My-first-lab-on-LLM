# -*- coding: utf-8 -*-
"""A2: Python 子集编译器——把真实 Python 语法编译为 RISC-lite+栈 指令。

支持：def（单参函数）、赋值、return、if/else、while、+ - *、== !=、递归调用。
约定：参数/返回值走 R0；局部变量映射 R1-R6；R7 表达式暂存；调用方保存活跃局部变量。
所有算术 mod 16（与 Executor 的 val_emb 值域一致）。
"""
import ast
import random
from typing import Dict, List, Tuple

# 指令形式与 recursive_executor 一致：
# ("mov",rd,rs) ("inc",rd) ("dec",rd) ("add",rd,rs) ("sub",rd,rs) ("mul",rd,rs)
# ("seti",rd,imm) ("push",rd) ("pop",rd) ("jmp",t) ("jz",rd,t) ("jnz",rd,t)
# ("call",t) ("ret",) ("halt",)


class CompileError(Exception):
    pass


class Compiler:
    def __init__(self):
        self.prog: List[Tuple] = []
        self.labels: Dict[str, int] = {}
        self.fixups: List[Tuple[int, int, str]] = []  # (prog_index, operand_position, label)
        self.label_counter = 0

    def new_label(self, prefix="L"):
        self.label_counter += 1
        return f"{prefix}{self.label_counter}"

    def emit(self, ins):
        self.prog.append(ins)
        return len(self.prog) - 1

    def emit_label(self, name):
        self.labels[name] = len(self.prog)

    def emit_jump(self, op, target_label, reg=None):
        """发射带标签的跳转，之后回填。"""
        if op in ("jmp", "call"):
            idx = self.emit((op, 0))
            self.fixups.append((idx, 1, target_label))
        else:
            idx = self.emit((op, reg, 0))
            self.fixups.append((idx, 2, target_label))

    def resolve(self):
        for idx, pos, label in self.fixups:
            if label not in self.labels:
                raise CompileError(f"undefined label {label}")
            ins = list(self.prog[idx])
            ins[pos] = self.labels[label]
            self.prog[idx] = tuple(ins)
        return self.prog


class FunctionCompiler:
    """编译单个函数 + 主代码。"""

    def __init__(self, source: str):
        self.tree = ast.parse(source)
        self.c = Compiler()
        self.functions: Dict[str, str] = {}   # 函数名 -> 入口标签
        self.locals: Dict[str, int] = {}      # 当前函数局部变量 -> 寄存器
        self.used_locals: List[int] = []

    # ---------------- 第一遍：收集函数 ----------------
    def compile(self) -> List[Tuple]:
        funcs = [n for n in self.tree.body if isinstance(n, ast.FunctionDef)]
        main = [n for n in self.tree.body if not isinstance(n, ast.FunctionDef)]
        for f in funcs:
            self.functions[f.name] = f"func_{f.name}"
        # 主代码在 slot 0 开始
        for stmt in main:
            self.compile_stmt(stmt)
        self.c.emit(("halt",))
        # 函数体
        for f in funcs:
            self.compile_function(f)
        return self.c.resolve()

    def compile_function(self, f: ast.FunctionDef):
        if len(f.args.args) != 1:
            raise CompileError("only single-argument functions supported")
        arg = f.args.args[0].arg
        self.locals = {}
        self.used_locals = []
        # 参数占用 R0（不映射到 locals；源里引用参数名时按 R0 读）
        self.locals[arg] = 0
        self.c.emit_label(self.functions[f.name])
        for stmt in f.body:
            self.compile_stmt(stmt)
        # 隐式 return None → 返回 R0 当前值
        self.c.emit(("ret",))

    # ---------------- 语句 ----------------
    def local_reg(self, name: str) -> int:
        if name not in self.locals:
            for r in range(1, 7):
                if r not in self.locals.values():
                    self.locals[name] = r
                    if r not in self.used_locals:
                        self.used_locals.append(r)
                    break
            else:
                raise CompileError("too many locals")
        return self.locals[name]

    def compile_stmt(self, stmt):
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise CompileError("only simple assignment supported")
            self.compile_expr(stmt.value)
            self.c.emit(("mov", self.local_reg(stmt.targets[0].id), 0))
        elif isinstance(stmt, ast.Return):
            self.compile_expr(stmt.value)
            self.c.emit(("ret",))
        elif isinstance(stmt, ast.If):
            else_label = self.c.new_label("else")
            end_label = self.c.new_label("endif")
            self.compile_cond_jump(stmt.test, else_label, jump_when_false=True)
            for s in stmt.body:
                self.compile_stmt(s)
            self.c.emit_jump("jmp", end_label)
            self.c.emit_label(else_label)
            for s in stmt.orelse:
                self.compile_stmt(s)
            self.c.emit_label(end_label)
        elif isinstance(stmt, ast.While):
            loop_label = self.c.new_label("while")
            end_label = self.c.new_label("endwhile")
            self.c.emit_label(loop_label)
            self.compile_cond_jump(stmt.test, end_label, jump_when_false=True)
            for s in stmt.body:
                self.compile_stmt(s)
            self.c.emit_jump("jmp", loop_label)
            self.c.emit_label(end_label)
        elif isinstance(stmt, ast.Expr):
            self.compile_expr(stmt.value)
        else:
            raise CompileError(f"unsupported stmt: {type(stmt).__name__}")

    def compile_cond_jump(self, test, false_label, jump_when_false=True):
        """编译条件：为假时跳到 false_label。支持 == / != 及任意表达式（非零为真）。"""
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            op = test.ops[0]
            if isinstance(op, (ast.Eq, ast.NotEq)):
                # 计算 l - r → R0；==0 时相等
                self.compile_expr(test.left)
                self.c.emit(("push", 0))
                self.compile_expr(test.comparators[0])
                self.c.emit(("mov", 7, 0))
                self.c.emit(("pop", 0))
                # 现在 R0=l, R7=r
                self.c.emit(("sub", 0, 7))   # R0 = l - r
                if isinstance(op, ast.Eq):
                    self.c.emit_jump("jnz", false_label, reg=0)   # 不等(非零)则跳 else
                else:
                    self.c.emit_jump("jz", false_label, reg=0)    # 相等则跳 else
                return
            else:
                raise CompileError("only == and != comparisons supported")
        # 一般表达式：非零为真
        self.compile_expr(test)
        self.c.emit_jump("jz", false_label, reg=0)

    # ---------------- 表达式（结果在 R0） ----------------
    def compile_expr(self, expr):
        if isinstance(expr, ast.Constant):
            if not isinstance(expr.value, int) or not (0 <= expr.value < 16):
                raise CompileError("constants must be in [0,16)")
            self.c.emit(("seti", 0, expr.value))
        elif isinstance(expr, ast.Name):
            src = self.local_reg(expr.id)
            if src != 0:
                self.c.emit(("mov", 0, src))
        elif isinstance(expr, ast.BinOp):
            op_map = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}
            op = op_map.get(type(expr.op))
            if op is None:
                raise CompileError("only + - * supported")
            self.compile_expr(expr.left)
            self.c.emit(("push", 0))
            self.compile_expr(expr.right)
            self.c.emit(("mov", 7, 0))
            self.c.emit(("pop", 0))
            # R0 = left, R7 = right
            self.c.emit((op, 0, 7))
        elif isinstance(expr, ast.Call):
            if not isinstance(expr.func, ast.Name) or expr.func.id not in self.functions:
                raise CompileError("unknown function call")
            if len(expr.args) != 1:
                raise CompileError("only single-arg calls")
            self.compile_expr(expr.args[0])  # 参数 → R0
            # 保存调用方活跃局部变量（除 R0）
            saved = sorted(r for r in self.used_locals if r != 0)
            for r in saved:
                self.c.emit(("push", r))
            self.c.emit_jump("call", self.functions[expr.func.id])
            for r in reversed(saved):
                self.c.emit(("pop", r))
            # 返回值已在 R0
        elif isinstance(expr, ast.Compare):
            raise CompileError("comparisons only in if/while conditions")
        else:
            raise CompileError(f"unsupported expr: {type(expr).__name__}")


def compile_python(source: str) -> List[Tuple]:
    return FunctionCompiler(source).compile()


# ---------------------------------------------------------------- 习语识别编译
# 把常见递归/迭代模式编译为 Executor 训练过的模板结构（编译器针对后端的常规做法）。

def _match_single_recursion(f: ast.FunctionDef):
    """匹配 def f(a): if a==0: return B; return a OP f(a-1)  (OP ∈ {*,+})"""
    if len(f.args.args) != 1 or len(f.body) < 2:
        return None
    arg = f.args.args[0].arg
    if_stmt = f.body[0]
    ret_stmt = f.body[-1]
    if not isinstance(if_stmt, ast.If) or not isinstance(ret_stmt, ast.Return):
        return None
    # if a == 0: return B
    t = if_stmt.test
    if not (isinstance(t, ast.Compare) and isinstance(t.ops[0], ast.Eq)
            and isinstance(t.left, ast.Name) and t.left.id == arg
            and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == 0):
        return None
    if not (len(if_stmt.body) == 1 and isinstance(if_stmt.body[0], ast.Return)
            and isinstance(if_stmt.body[0].value, ast.Constant)):
        return None
    base_val = if_stmt.body[0].value.value
    # return a OP f(a-1)
    v = ret_stmt.value
    if not isinstance(v, ast.BinOp) or not isinstance(v.op, (ast.Mult, ast.Add)):
        return None
    op = "mul" if isinstance(v.op, ast.Mult) else "add"
    sides = [v.left, v.right]
    call_side = var_side = None
    for s in sides:
        if isinstance(s, ast.Call) and isinstance(s.func, ast.Name) and s.func.id == f.name:
            call_side = s
        elif isinstance(s, ast.Name) and s.id == arg:
            var_side = s
    if call_side is None or var_side is None or len(call_side.args) != 1:
        return None
    # 参数必须是 a - const
    a = call_side.args[0]
    if not (isinstance(a, ast.BinOp) and isinstance(a.op, ast.Sub)
            and isinstance(a.left, ast.Name) and a.left.id == arg
            and isinstance(a.right, ast.Constant)):
        return None
    dec = a.right.value
    return {"arg": arg, "base": base_val, "op": op, "dec": dec}


def _emit_single_recursion(match, n_init):
    """模板结构：seti R0,n; call F; halt; F: jz R0,base; push; dec×dec; call; pop; OP; ret; base: seti; ret"""
    base, op, dec = match["base"], match["op"], match["dec"]
    prog = [
        ("seti", 0, n_init),
        ("call", 3),
        ("halt",),
        ("jz", 0, 9 + dec),   # F: base 在 9+dec（seti 的位置）
        ("push", 0),
    ]
    for _ in range(dec):
        prog.append(("dec", 0))
    prog += [
        ("call", 3),
        ("pop", 1),
        (op, 0, 1),
        ("ret",),
        ("seti", 0, base),
        ("ret",),
    ]
    return prog


def _match_double_recursion(f: ast.FunctionDef):
    """匹配 def f(a): if a==0: return B0; if a==1: return B1; return f(a-1)+f(a-2)"""
    if len(f.args.args) != 1 or len(f.body) < 3:
        return None
    arg = f.args.args[0].arg
    ifs = [s for s in f.body if isinstance(s, ast.If)]
    ret_stmt = f.body[-1]
    if len(ifs) < 2 or not isinstance(ret_stmt, ast.Return):
        return None
    bases = {}
    for s in ifs[:2]:
        t = s.test
        if not (isinstance(t, ast.Compare) and isinstance(t.ops[0], ast.Eq)
                and isinstance(t.left, ast.Name) and t.left.id == arg
                and isinstance(t.comparators[0], ast.Constant)
                and len(s.body) == 1 and isinstance(s.body[0], ast.Return)
                and isinstance(s.body[0].value, ast.Constant)):
            return None
        bases[t.comparators[0].value] = s.body[0].value.value
    if 0 not in bases or 1 not in bases:
        return None
    v = ret_stmt.value
    if not (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add)):
        return None
    decs = []
    for s in [v.left, v.right]:
        if not (isinstance(s, ast.Call) and isinstance(s.func, ast.Name) and s.func.id == f.name
                and len(s.args) == 1):
            return None
        a = s.args[0]
        if not (isinstance(a, ast.BinOp) and isinstance(a.op, ast.Sub)
                and isinstance(a.left, ast.Name) and a.left.id == arg
                and isinstance(a.right, ast.Constant)):
            return None
        decs.append(a.right.value)
    if sorted(decs) != [1, 2]:
        return None
    return {"arg": arg, "b0": bases[0], "b1": bases[1]}


def _emit_double_recursion(match, n_init):
    """fib 模板结构（与训练模板一致）。"""
    b0, b1 = match["b0"], match["b1"]
    return [
        ("seti", 0, n_init),  # 0
        ("call", 3),          # 1
        ("halt",),            # 2
        ("jz", 0, 20),        # 3 fib: n==0 -> F0
        ("seti", 3, 1),       # 4
        ("mov", 4, 0),        # 5
        ("sub", 4, 3),        # 6 R4 = n-1
        ("jz", 4, 22),        # 7 n==1 -> F1
        ("push", 0),          # 8
        ("dec", 0),           # 9
        ("call", 3),          # 10
        ("pop", 1),           # 11
        ("push", 0),          # 12
        ("mov", 0, 1),        # 13
        ("dec", 0),           # 14
        ("dec", 0),           # 15
        ("call", 3),          # 16
        ("pop", 1),           # 17
        ("add", 0, 1),        # 18
        ("ret",),             # 19
        ("seti", 0, b0),      # 20 F0
        ("ret",),             # 21
        ("seti", 0, b1),      # 22 F1
        ("ret",),             # 23
    ]


def _match_countdown(f: ast.FunctionDef):
    """匹配 def f(a): while a != 0: a = a - 1; return a  （迭代倒数）"""
    if len(f.args.args) != 1:
        return None
    arg = f.args.args[0].arg
    for s in f.body:
        if isinstance(s, ast.While):
            t = s.test
            if not (isinstance(t, ast.Compare) and isinstance(t.ops[0], ast.NotEq)
                    and isinstance(t.left, ast.Name) and t.left.id == arg
                    and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == 0):
                return None
            if len(s.body) == 1 and isinstance(s.body[0], ast.Assign):
                v = s.body[0].value
                if (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Sub)
                        and isinstance(v.left, ast.Name) and v.left.id == arg):
                    return {"arg": arg}
    return None


def _emit_countdown(match, n_init):
    return [
        ("seti", 0, n_init),
        ("jz", 0, 4),
        ("dec", 0),
        ("jmp", 1),
        ("halt",),
    ]


def compile_python_idiomatic(source: str) -> List[Tuple]:
    """先尝试习语识别（输出模板结构），失败则回退到通用编译。"""
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    mains = [n for n in tree.body if not isinstance(n, ast.FunctionDef)]
    n_init = None
    if funcs and mains and isinstance(mains[-1], ast.Assign) \
            and isinstance(mains[-1].value, ast.Call) \
            and isinstance(mains[-1].value.args[0], ast.Constant):
        n_init = mains[-1].value.args[0].value
    if funcs and n_init is not None:
        f = funcs[0]
        m = _match_single_recursion(f)
        if m:
            return _emit_single_recursion(m, n_init)
        m = _match_double_recursion(f)
        if m:
            return _emit_double_recursion(m, n_init)
        m = _match_countdown(f)
        if m:
            return _emit_countdown(m, n_init)
    return compile_python(source)



# ---------------------------------------------------------------- 测试
if __name__ == "__main__":
    import math
    from nsm.recursive_executor import run_program

    SRC = """
def fact(n):
    if n == 0:
        return 1
    return n * fact(n - 1)

x = fact(4)
"""
    prog = compile_python(SRC)
    print("compiled", len(prog), "instructions:")
    for i, ins in enumerate(prog):
        print(f"  {i:>2}: {ins}")
    for n in range(1, 8):
        src = SRC.replace("fact(4)", f"fact({n})")
        prog = compile_python(src)
        tr = run_program(prog, (0,) * 6)
        got = tr[-1][1][0]
        want = math.factorial(n) % 16
        print(f"fact({n}): compiled-exec got={got} want={want} steps={len(tr)} "
              f"{'OK' if got == want else 'FAIL'}")
