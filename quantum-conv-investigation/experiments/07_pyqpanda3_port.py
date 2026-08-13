# -*- coding: utf-8 -*-
"""
实验 07: n=4 量子卷积电路的 pyqpanda3 移植版 (实验05 §5.1 + §5.2)
====================================================================
将 GitHub 仓库 LEE-Gst/My-lab/quantum-conv-investigation 实验 05 的
n=4 量子电路 (QFT + LCU 块编码 + QFT†) 从 Qiskit 移植到本源量子 pyqpanda3,
用于本源量子平台 (CPUQVM 模拟器 + OriginIR 导出/真机提交) 的可运行性验证。

与实验 05 的关系:
  * §5.1 精确验证  -> 本脚本 [1]-[5] 节 (含平台约定校准与块级自检)
  * §5.2 噪声仿真  -> 本脚本 [6] 节 (与 Aer 退极化结果同口径对照)
  * §5.3 墙钟换算  -> 不复制 (纯解析, 见实验 05_quantum_wallclock.py)

验证目标 (与原实验一致):
  * |<anc=0 分量 | y_target>|^2 = 404/480 ≈ 0.8417 (精确命中)
  * P(anc=0) ≈ 0.8417
  * 后选择系统分布 ≈ {00:0.2995, 01:0.2005, 10:0.2005, 11:0.2995}

电路结构 (3 比特: q0,q1 系统 + q2 辅助):
  态制备 |x>=(3,1,4,2)/√30 → QFT(2) → 4 个受控块编码 B_k(LCU) → QFT†(2)
  → 测量, 后选择 anc=0

设计要点 (相对原移植版的提升, 见仓库 README 复现注意):
  1. 平台约定校准: 状态向量索引序 / counts 键序 / 概率列表序全部用探针
     电路实测判定, 不硬编码 (键序在 pyqpanda 3.8.5 与 pyqpanda3 间不同);
  2. 子电路矩阵自检: QFT / 态制备 / 每个 C²(B_k) 的 8×8 矩阵逐一与理论
     对比 (全局相位不敏感比较), 链式 << 执行序等平台陷阱在此暴露;
  3. d 值由数值求解 diag(d) = Q·(A_c/4)·Q† 得到并断言, 不再硬编码;
  4. 分支相位一致性: det(B(d)) = -1 对所有 d 成立 (s=√(1-|d|²) 实数),
     故四个分支共享同一全局相位, 无物理效应 —— 但要求所有分支的实现
     采用同一相位约定 (精确 B 或 SU(2) 剥离版二选一, 不可混用);
  5. 通用 ZYZ 回退分支修正: 删除原脚本中错误的 CP(0,1,φ) 相位实现
     (它只在分支 3 生效); c2_rz/c2_rx 工具本身只能实现 det=1 的酉,
     通用路径自然落在 SU(2) 剥离约定上, 全通用模式内部自洽;
  6. 理想分布用概率接口 (get_prob_dict), 不再烧 20 万 shots;
  7. 输出 PASS/FAIL 判定, 可被 CI 直接消费。

用法:
  python 07_pyqpanda3_port.py            # local: 校准+自检+精确验证+资源+OriginIR
  python 07_pyqpanda3_port.py noise      # + 退极化 p 三档噪声对照 (实验05 §5.2)
  python 07_pyqpanda3_port.py all        # local + noise
  python 07_pyqpanda3_port.py chip       # local + 导出 OriginIR 文件 (真机提交指引)

依赖: numpy + pyqpanda3>=0.4.0 (pip install pyqpanda3 -i 国内镜像)
"""
import os
import sys

import numpy as np

# Windows GBK 控制台防护: 输出统一 UTF-8 (含 ²/×/π 等符号)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pyqpanda3.core import (
    QCircuit, QProg, CPUQVM, NoiseModel,
    H, X, RY, CRY, CRZ, CP, CU, CNOT, SWAP, TOFFOLI, measure,
    depolarizing_error, GateType,
)

# ============================================================
# 0. 经典参考 + d 值数值求解 (自检)
# ============================================================
c = np.array([1, 2, 1, 0], float)
x = np.array([3, 1, 4, 2], float) / np.sqrt(30)
y_target = np.array([11, 9, 9, 11], float) / np.sqrt(404)

Ac = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        Ac[i, j] = c[(i - j) % 4]
y = Ac @ x

# Q = ω=+i 的 4x4 QFT 矩阵 (qiskit QFTGate 约定 = numpy FFT 的共轭)
Q = np.array([[np.exp(2j * np.pi * j * k / 4) / 2 for k in range(4)]
              for j in range(4)])


def solve_dvals():
    """数值求解 diag(d) = Q·(A_c/4)·Q†, 并断言对角化成立。"""
    D = Q @ (Ac / 4) @ Q.conj().T
    off = np.abs(D - np.diag(np.diag(D))).max()
    assert off < 1e-12, "A_c 未被 Q 对角化 (off-diag=%.2e)" % off
    dvals = np.diag(D)
    # 与理论解 (1, i/2, 0, -i/2) 交叉自检
    expected = np.array([1.0, 0.5j, 0.0, -0.5j])
    assert np.abs(dvals - expected).max() < 1e-12, "d 数值解与理论不符"
    return dvals


DVAL_THEORY = np.array([1.0, 0.5j, 0.0, -0.5j])
VERDICT = []


def check(name, ok, detail=""):
    """PASS/FAIL 判定行, 汇总到 VERDICT。"""
    tag = "[PASS]" if ok else "[FAIL]"
    print("  %s %-40s %s" % (tag, name, detail))
    VERDICT.append(ok)


# ============================================================
# 1. 电路构建 (pyqpanda3; 门参数 = 比特索引 int)
# ============================================================

def qft2():
    """QFT(2): H(1)·CP(0→1,π/2)·H(0)·SWAP, ω=+i 约定 (≡ qiskit QFTGate(2))。

    执行序: pyqpanda3 的 QCircuit << 为左→右 (第 3 节矩阵自检验证)。"""
    cir = QCircuit(2)
    cir << H(1)
    cir << CP(0, 1, np.pi / 2)
    cir << H(0)
    cir << SWAP(0, 1)
    return cir


def iqft2():
    cir = QCircuit(2)
    cir << SWAP(0, 1)
    cir << H(0)
    cir << CP(0, 1, -np.pi / 2)
    cir << H(1)
    return cir


def c2_rz(theta):
    """C²(Rz(θ)) = TOFFOLI·CRZ(q0,q2,-θ/2)·TOFFOLI·CRZ(q0,q2,θ/2)
    恒等式: X·Rz(φ)·X = Rz(-φ)。目标 q2, 控制 q0,q1。"""
    cir = QCircuit(3)
    cir << TOFFOLI(0, 1, 2)
    cir << CRZ(0, 2, -theta / 2)
    cir << TOFFOLI(0, 1, 2)
    cir << CRZ(0, 2, theta / 2)
    return cir


def c2_rx(theta):
    """C²(Rx(θ)) = H(q2)·C²(Rz(θ))·H(q2)"""
    cir = QCircuit(3)
    cir << H(2)
    cir << c2_rz(theta)
    cir << H(2)
    return cir


# ---- 噪声友好原语 (只用可注册噪声的门: H/X/RY/RZ/CNOT/CP/SWAP/CU) ----
# pyqpanda3 0.4.0 的 NoiseModel 仅接受有限门类型 (CRY/CRZ/TOFFOLI 均被拒),
# 故噪声阶段用以下精确等价分解 (矩阵自检 + 全电路 404/480 双重保证等价)。

def cry_nf(c, t, theta, nq=3):
    """CRY(c,t,θ) 精确分解: RY(t,θ/2)·CNOT(c,t)·RY(t,-θ/2)·CNOT(c,t)"""
    cir = QCircuit(nq)
    cir << RY(t, theta / 2) << CNOT(c, t) << RY(t, -theta / 2) << CNOT(c, t)
    return cir


def crz_nf(c, t, theta, nq=3):
    """CRZ(c,t,θ) = CU(c,t,0,θ,0,0) (实测: CU 的 φ 参数 = Rz)"""
    cir = QCircuit(nq)
    cir << CU(c, t, 0, theta, 0, 0)
    return cir


def ccx_nf(c0, c1, t):
    """精确 Toffoli = H(t)·CP(c0,c1,π/2)·C²Rz(π)·H(t)

    C²Rz(θ) = CRZ(c0,t,θ/2)·CNOT(c1,c0)·CRZ(c0,t,-θ/2)·CNOT(c1,c0)·CRZ(c1,t,θ/2)
    (numpy 数值验证精确成立); C²Z = CP(c0,c1,π/2)·C²Rz(π), C²X = H·C²Z·H。"""
    cir = QCircuit(3)
    cir << H(t)
    cir << CP(c0, c1, np.pi / 2)
    cir << crz_nf(c0, t, np.pi / 2)
    cir << CNOT(c1, c0)
    cir << crz_nf(c0, t, -np.pi / 2)
    cir << CNOT(c1, c0)
    cir << crz_nf(c1, t, np.pi / 2)
    cir << H(t)
    return cir


def c2_rz_nf(theta):
    """C²(Rz(θ)) 噪声友好版: CCX·CRZ·CCX·CRZ (X·Rz(φ)·X = Rz(-φ))"""
    cir = QCircuit(3)
    cir << ccx_nf(0, 1, 2)
    cir << crz_nf(0, 2, -theta / 2)
    cir << ccx_nf(0, 1, 2)
    cir << crz_nf(0, 2, theta / 2)
    return cir


def c2_rx_nf(theta):
    """C²(Rx(θ)) 噪声友好版 = H(q2)·C²(Rz(θ))·H(q2)"""
    cir = QCircuit(3)
    cir << H(2)
    cir << c2_rz_nf(theta)
    cir << H(2)
    return cir


def zyz_su2(U):
    """ZYZ 分解 U = Rz(α)·Rx(β)·Rz(γ) (U ∈ SU(2)), 返回 (α, β, γ)。

    推导: U[0,0] = cos(β/2)e^{-i(α+γ)/2}, U[0,1] = -i·sin(β/2)e^{-i(α-γ)/2}。"""
    a, b = U[0, 0], U[0, 1]
    beta = 2 * np.arccos(np.clip(abs(a), 0, 1))
    if abs(a) > 1e-12:
        alpha = -np.angle(a) - np.angle(b) - np.pi / 2
        gamma = -np.angle(a) + np.angle(b) + np.pi / 2
    else:  # |a|=0 -> β=π, 令 γ=0
        alpha = -2 * np.angle(b) - np.pi
        gamma = 0.0
    return alpha, beta, gamma


def c2_block(k, dval, use_generic=False, noise_friendly=False):
    """系统态 |k⟩ (q0=k&1, q1=(k>>1)&1) 时对辅助 q2 施加 B(d), 其余分支恒等。

    B(d) = [[d, s], [s, -conj(d)]], s = √(1-|d|²); det(B) = -|d|² - s² = -1 对一切 d。

    相位约定 (本实验核心陷阱, 对应实验05 README "分支相位不可丢"):
      - 分支 k 上实现的酉 V_k 与 B_k 之间允许差一个公共全局相位 e^{iφ},
        但要求 φ 对所有分支相同;
      - c2_rz/c2_rx 只能构成 det=1 的酉, 故"通用路径"自然实现
        U_k = B_k·e^{-iπ/2} (det=1), 四分支一致 -> 公共全局相位, 无害;
      - 特例路径实现 B_k 精确版 (det=-1), 四分支也一致 -> 同样无害;
      - 两者不可混用: 特例与通用混搭会产生分支相对相位, 破坏 LCU。

    noise_friendly=True 时用噪声可注册门 (H/X/RY/RZ/CNOT/CP/SWAP/CU) 的
    精确等价分解, 供噪声阶段使用 (矩阵自检保证等价)。"""
    s = np.sqrt(max(1 - abs(dval) ** 2, 0.0))
    B = np.array([[dval, s], [s, -np.conj(dval)]], complex)
    t2 = (c2_rz_nf, c2_rx_nf, ccx_nf) if noise_friendly else (c2_rz, c2_rx, TOFFOLI)
    rz, rx, ccx = t2

    if not use_generic:
        # ---- 特例路径: 精确 B (det=-1), 最小门数 ----
        if abs(dval - 1) < 1e-9:            # B = Z: H·TOFFOLI·H
            inner = QCircuit(3)
            inner << H(2) << ccx(0, 1, 2) << H(2)
        elif abs(dval) < 1e-9:              # B = X: TOFFOLI
            inner = QCircuit(3)
            inner << ccx(0, 1, 2)
        elif abs(dval - 0.5j) < 1e-9:       # B = (i/2)I + (√3/2)X = X·R_x(-π/3)
            inner = QCircuit(3)
            inner << rx(-np.pi / 3) << ccx(0, 1, 2)
        elif abs(dval + 0.5j) < 1e-9:       # B = (-i/2)I + (√3/2)X = X·R_x(π/3)
            inner = QCircuit(3)
            inner << rx(np.pi / 3) << ccx(0, 1, 2)
        else:
            use_generic = True              # 含 d=-1: 原脚本该分支与 d=1 相同, 是错的
    if use_generic:
        # ---- 通用路径: U = B·e^{-iπ/2} ∈ SU(2), ZYZ 分解 ----
        phi = np.angle(np.linalg.det(B)) / 2
        U = B * np.exp(-1j * phi)
        assert abs(np.linalg.det(U) - 1) < 1e-12, "相位剥离后应为 SU(2)"
        alpha, beta, gamma = zyz_su2(U)
        inner = QCircuit(3)
        inner << rz(alpha)
        inner << rx(beta)
        inner << rz(gamma)
    # ctrl_state 包装: 控制比特为 0 的分支先 X 翻转
    cir = QCircuit(3)
    if not (k & 1):
        cir << X(0)
    if not ((k >> 1) & 1):
        cir << X(1)
    cir << inner
    if not ((k >> 1) & 1):
        cir << X(1)
    if not (k & 1):
        cir << X(0)
    return cir


def state_prep(noise_friendly=False):
    """态制备 |x⟩ = (3,1,4,2)/√30, 索引 = q1·2+q0 (q0 低位)。

    角度: P(q0=0)=(3²+4²)/30=5/6; q0=0 时 q1 ∝ (3,4); q0=1 时 q1 ∝ (1,2)。"""
    th0 = 2 * np.arccos(np.sqrt(5 / 6))
    th1 = 2 * np.arccos(3 / 5)
    th2 = 2 * np.arccos(1 / np.sqrt(5))
    cry = cry_nf if noise_friendly else CRY
    cry_args = (0, 1, th1, 2) if noise_friendly else (0, 1, th1)
    cry_args2 = (0, 1, th2, 2) if noise_friendly else (0, 1, th2)
    cir = QCircuit(2)
    cir << RY(0, th0)
    cir << X(0) << cry(*cry_args) << X(0)   # 控制 q0=0
    cir << cry(*cry_args2)                  # 控制 q0=1
    return cir


def build_full_circuit(dvals=None, use_generic=False, noise_friendly=False):
    """完整 3 比特电路: 态制备 + QFT + LCU + QFT†。dvals 缺省用数值解。"""
    if dvals is None:
        dvals = solve_dvals()
    circuit = QCircuit(3)
    circuit << state_prep(noise_friendly)
    circuit << qft2()
    for k in range(4):
        circuit << c2_block(k, dvals[k], use_generic, noise_friendly)
    circuit << iqft2()
    return circuit


# ============================================================
# 2. 平台约定校准 (探针电路, 不靠猜)
# ============================================================

def calibrate():
    """实测三条约定: 状态向量索引序 / counts 键序 / 概率列表序。"""
    qvm = CPUQVM()
    sv_w = {}   # sv_w[q] = 比特 q 在状态向量索引中的权重
    for q in (0, 1, 2):
        qvm.run(QProg() << X(q), 1)
        sv = np.array(qvm.result().get_state_vector(), complex)
        idx = int(np.argmax(np.abs(sv)))
        assert abs(abs(sv[idx]) - 1) < 1e-9
        sv_w[q] = idx
    assert sorted(sv_w.values()) == [1, 2, 4], "sv 索引权重异常: %s" % sv_w

    keypos = {}  # keypos[q] = 比特 q 在 counts 键中的位置
    for q in (0, 1, 2):
        prog = QProg()
        prog << X(q) << measure(0, 0) << measure(1, 1) << measure(2, 2)
        qvm.run(prog, 8)
        key = next(iter(qvm.result().get_counts()))
        keypos[q] = key.index("1")
    assert sorted(keypos.values()) == [0, 1, 2], "counts 键序异常: %s" % keypos

    # 概率列表序 vs 状态向量序
    qvm.run(QProg() << X(2) << measure(0, 0) << measure(1, 1) << measure(2, 2), 8)
    pl = qvm.result().get_prob_list()
    assert abs(pl[sv_w[2]] - 1) < 1e-9, "prob_list 与 sv 索引序不一致"

    qvm2 = CPUQVM()
    return qvm2, sv_w, keypos


def circuit_matrix(cir):
    """QCircuit 矩阵 (约定: [b,a] = <b|U|a>, 实测确认)。"""
    return np.array(cir.matrix(), complex)


# ============================================================
# 3. 子电路矩阵自检
# ============================================================
def selfcheck_subcircuits(dvals):
    print("[2] 子电路矩阵自检")
    print("    (矩阵约定 [b,a]=<b|U|a>; 比较剥离全局相位)")
    dim = 4

    # --- QFT / QFT† ---
    Mq = circuit_matrix(qft2())
    # QFT 电路的列序: 输入索引 = q0+2q1, 输出 = 同序; U = Q (ω=+i)
    ovl = abs(np.vdot(Mq.ravel(), Q.ravel())) / dim
    check("QFT2 = Q (ω=+i)", ovl > 1 - 1e-9, "|<U|Q>|/4 = %.10f" % ovl)
    Miq = circuit_matrix(iqft2())
    ovl2 = abs(np.vdot(Miq.ravel(), Q.conj().T.ravel())) / dim
    check("QFT2† = Q†", ovl2 > 1 - 1e-9, "|<U|Q†>|/4 = %.10f" % ovl2)

    # --- 态制备 ---
    Mp = circuit_matrix(state_prep())
    xs = np.zeros(dim, complex)
    # 索引 = q1·2+q0: |00>→3, |01>→1, |10>→4, |11>→2
    xs[0], xs[1], xs[2], xs[3] = 3, 1, 4, 2
    xs = xs / np.sqrt(30)
    # U|00> = 第 0 列 = x
    ok = np.abs(Mp[:, 0] - xs).max() < 1e-9
    check("state_prep |00> -> x", ok,
          "max|Δ| = %.1e" % np.abs(Mp[:, 0] - xs).max())

    # --- 噪声友好态制备 (CRY 显式分解) ---
    Mpnf = circuit_matrix(state_prep(noise_friendly=True))
    oknf = np.abs(Mpnf[:, 0] - xs).max() < 1e-9
    check("state_prep(nf) |00> -> x", oknf,
          "max|Δ| = %.1e" % np.abs(Mpnf[:, 0] - xs).max())

    # --- 每个 C²(B_k): 特例路径 (dvals) 与通用路径 (全量含 d=-1) ---
    # 相位约定: 特例路径实现精确 B (det=-1); 通用路径实现 B·e^{-iπ/2}
    # (det=1, c2_rz/c2_rx 工具族的自然约定)。自检目标必须按实现约定取。
    for name, ds, k_of in (
            ("特例路径", dvals, range(4)),
            ("通用路径(含d=-1)", np.append(dvals, -1.0), list(range(4)) + [0])):
        for i, d in enumerate(ds):
            k = k_of[i]
            use_generic = (name.startswith("通用")) or not (
                abs(d - 1) < 1e-9 or abs(d) < 1e-9 or
                abs(d - 0.5j) < 1e-9 or abs(d + 0.5j) < 1e-9)
            for nf in (False, True):
                cir = c2_block(k, d, use_generic, noise_friendly=nf)
                M = circuit_matrix(cir)
                s = np.sqrt(max(1 - abs(d) ** 2, 0.0))
                B = np.array([[d, s], [s, -np.conj(d)]], complex)
                if use_generic:                 # 剥离 det 相位后的 SU(2) 目标
                    Bt = B * np.exp(-1j * np.angle(np.linalg.det(B)) / 2)
                else:
                    Bt = B
                # 理论: 分支 k 上对辅助作用 Bt, 其余分支恒等; 索引 j = s_q + 4·a
                T = np.eye(8, dtype=complex)
                for a_in in range(2):
                    for a_out in range(2):
                        T[k + 4 * a_out, k + 4 * a_in] = Bt[a_out, a_in]
                ovl = abs(np.vdot(M.ravel(), T.ravel())) / 8
                tag = "[%s%s]" % (name, ",nf" if nf else "")
                check("C²(B(d=%s)) k=%d %s" % (fmt_d(d), k, tag),
                      ovl > 1 - 1e-9, "|<U|T>|/8 = %.10f" % ovl)


def fmt_d(d):
    """复数 d 的紧凑显示, 避免 str(complex) 的括号噪音。"""
    re_, im = float(np.real(d)), float(np.imag(d))
    if abs(im) < 1e-12:
        return "%g" % re_
    return "%g%+gi" % (re_, im)


# ============================================================
# 4. 完整电路精确验证
# ============================================================
def verify_exact(qvm, keypos, dvals):
    print("[3] n=4 电路精确验证 (CPUQVM 状态向量)")
    circuit = build_full_circuit(dvals)
    prog = QProg()
    prog << circuit << measure(0, 0) << measure(1, 1) << measure(2, 2)
    qvm.run(prog, 1)
    sv = np.array(qvm.result().get_state_vector(), complex)

    amp0 = np.array(sv[0:4])          # anc=0 分量 (sv 索引 = q0+2q1+4q2, q0 低位)
    print("    anc=0 分量 (未归一):", np.round(amp0, 6))
    print("    y_target           :", np.round(y_target, 6))
    ovl = np.abs(np.vdot(amp0, y_target)) ** 2
    check("|<anc=0|y_target>|² = 404/480", abs(ovl - 404 / 480) < 1e-9,
          "= %.9f (期望 0.841667)" % ovl)
    p0 = (np.abs(sv[0:4]) ** 2).sum()
    check("P(anc=0) [状态向量]", abs(p0 - 404 / 480) < 1e-9, "= %.9f" % p0)

    # 全通用路径交叉验证 (回退路径在完整电路层面的闭环)
    circuit_g = build_full_circuit(dvals, use_generic=True)
    prog_g = QProg()
    prog_g << circuit_g << measure(0, 0) << measure(1, 1) << measure(2, 2)
    qvm.run(prog_g, 1)
    sv_g = np.array(qvm.result().get_state_vector(), complex)
    ovl_g = np.abs(np.vdot(sv_g[0:4], y_target)) ** 2
    check("通用路径全电路 404/480", abs(ovl_g - 404 / 480) < 1e-9, "= %.9f" % ovl_g)

    # 噪声友好版全电路闭环 (噪声阶段用的等价电路)
    circuit_nf = build_full_circuit(dvals, noise_friendly=True)
    prog_nf = QProg()
    prog_nf << circuit_nf << measure(0, 0) << measure(1, 1) << measure(2, 2)
    qvm.run(prog_nf, 1)
    sv_nf = np.array(qvm.result().get_state_vector(), complex)
    ovl_nf = np.abs(np.vdot(sv_nf[0:4], y_target)) ** 2
    check("噪声友好版全电路 404/480", abs(ovl_nf - 404 / 480) < 1e-9, "= %.9f" % ovl_nf)
    return prog, circuit


# ============================================================
# 5. 理想分布 (概率接口 + 采样复核)
# ============================================================
def ideal_distribution(qvm, prog, keypos, shots=200000):
    print("[4] 理想分布 (状态向量精确 + %d shots 采样复核)" % shots)
    # --- 精确: 从状态向量算 |amp|² (prob_dict/prob_list 为频率统计, 需大 shots) ---
    qvm.run(prog, 1)                      # 确保 result 为本电路 (勿依赖上游调用序)
    sv = np.array(qvm.result().get_state_vector(), complex)
    p_exact = (np.abs(sv[0:4]) ** 2) / (np.abs(sv[0:4]) ** 2).sum()
    expected = {"00": 121 / 404, "01": 81 / 404, "10": 81 / 404, "11": 121 / 404}
    dist_exact = {"00": p_exact[0], "01": p_exact[1], "10": p_exact[2], "11": p_exact[3]}
    print("    后选择系统分布 (sv 精确):", {k: round(v, 4) for k, v in dist_exact.items()})
    ok = all(abs(dist_exact[k] - expected[k]) < 1e-9 for k in expected)
    check("后选择分布 == 理论 (精确)", ok)

    # --- 采样复核: prob_dict + counts (200k, 容差 = 4σ ≈ 0.004) ---
    qvm.run(prog, shots)
    anc_at = keypos[2]
    pd = qvm.result().get_prob_dict()
    p0_prob = sum(v for k, v in pd.items() if k[anc_at] == "0")
    check("prob_dict P(anc=0) ≈ 0.8417", abs(p0_prob - 404 / 480) < 0.004,
          "= %.4f" % p0_prob)
    cnt = qvm.result().get_counts()
    p0_s = sum(v for k, v in cnt.items() if k[anc_at] == "0") / shots
    check("counts P(anc=0) ≈ 0.8417", abs(p0_s - 404 / 480) < 0.004,
          "= %.4f" % p0_s)
    sub_s = {k[:anc_at] + k[anc_at + 1:]: v for k, v in cnt.items() if k[anc_at] == "0"}
    tot_s = sum(sub_s.values())
    dist_s = {k: v / tot_s for k, v in sorted(sub_s.items())}
    print("    采样后选择分布:", {k: round(v, 4) for k, v in dist_s.items()})
    ok_s = all(abs(dist_s.get(k, 0) - expected[k]) < 0.004 for k in expected)
    check("采样后选择分布 ≈ 理论", ok_s)


# ============================================================
# 6. 噪声仿真 (退极化, 对照实验05 §5.2)
# ============================================================
def make_noise_model(p):
    """所有门统一退极化率 p (作用于噪声友好版电路, 每比特一通道)。

    pyqpanda3 0.4.0 的 NoiseModel 限制 (实测):
      - CRY/CRZ/TOFFOLI 等复合门类型无法注册错误 (add 直接报错);
      - 2 比特错误 (QuantumError(dict 两字母键)) 仅在 U4 等少数类型可挂,
        对 CNOT/CP/SWAP 报 "tensor qubit num error";
      - 1 比特错误可注册到 H/X/RY/RZ 及 CNOT/CP/SWAP/CU, 语义为
        "通道作用于该门的每一个比特" (实测 p=1 时两比特全混)。
    故噪声阶段把电路改写为仅含可注册门的精确等价分解 (矩阵自检保证),
    并统一施加 1q 退极化 p:
      - 1 比特门 (H/X/RY/RZ): 该比特退极化 p;
      - 2 比特门 (CNOT/CP/SWAP/CU): 两比特各自退极化 p (领头阶等效 2q 错误率 2p)。
    与实验05 Aer (cx 2q 退极化 p + u 1q 退极化 p) 属同族模型, 但门分解
    与错误分配方式不同, 同 p 下数值不可直接相等, 只做趋势/量级对照。"""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p),
        [GateType.RY, GateType.X, GateType.H, GateType.RZ])
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p),
        [GateType.CNOT, GateType.CP, GateType.SWAP, GateType.CU])
    return nm


def noise_stage(qvm, dvals, keypos, shots=200000):
    print("[5] 噪声仿真: 退极化 p 扫描 (对照实验05 §5.2)")
    ideal_dist = np.array([121, 81, 81, 121], float) / 404
    anc_at = keypos[2]
    # 噪声友好版电路 (全部门可注册噪声, 与精确版等价, 见 [3] 闭环验证)
    circuit_nf = build_full_circuit(dvals, noise_friendly=True)
    prog = QProg()
    prog << circuit_nf << measure(0, 0) << measure(1, 1) << measure(2, 2)
    print("    实验05 Aer 参考: p=1e-4 -> P=0.8385 F=1.0000 | "
          "p=1e-3 -> P=0.8222 F=0.9999 | p=1e-2 -> P=0.6895 F=0.9983")
    for p in (1e-4, 1e-3, 1e-2):
        nm = make_noise_model(p)
        qvm.run(prog, shots, nm)
        cnt = qvm.result().get_counts()
        p0 = sum(v for k, v in cnt.items() if k[anc_at] == "0") / shots
        sub = {k[:anc_at] + k[anc_at + 1:]: v for k, v in cnt.items()
               if k[anc_at] == "0"}
        tot = sum(sub.values())
        if tot > 0:
            qn = np.array([sub.get("00", 0), sub.get("01", 0),
                           sub.get("10", 0), sub.get("11", 0)], float) / tot
            fid = (np.sum(np.sqrt(qn * ideal_dist))) ** 2
        else:
            fid = 0.0
        print("    p=%.0e  P(anc=0)=%.4f  后选择保真度 F=%.4f  采样开销 x%.2f"
              % (p, p0, fid, 0.841667 / max(p0, 1e-9)))
    # 退极化 p=1 的健全性检查: 应逼近完全混合 (P≈0.5)
    qvm.run(prog, shots, make_noise_model(1.0))
    cnt = qvm.result().get_counts()
    p0max = sum(v for k, v in cnt.items() if k[anc_at] == "0") / shots
    check("p=1 健全性 (P≈0.5)", abs(p0max - 0.5) < 0.005, "P=%.4f" % p0max)


# ============================================================
# 7. 门资源账本 (原生 + cx/u 折算, 对照实验05)
# ============================================================
def resource_ledger(circuit):
    print("[6] 门资源账本")
    ops = circuit.count_ops()
    print("    原生门 (pyqpanda3 count_ops):", dict(sorted(ops.items())))
    d = circuit.depth()
    print("    深度:", d)
    # 折算 (标准分解): TOFFOLI≈6cx, CRZ/CRY/CP≈2cx, SWAP≈3cx
    cx_est = (ops.get("CCX", 0) * 6 + ops.get("CRZ", 0) * 2 +
              ops.get("CRY", 0) * 2 + ops.get("CP", 0) * 2 +
              ops.get("SWAP", 0) * 3)
    u_est = (ops.get("RY", 0) + ops.get("X", 0) + ops.get("H", 0) +
             ops.get("RZ", 0) + ops.get("CRZ", 0) * 2 + ops.get("CRY", 0) * 2 +
             ops.get("CP", 0) + ops.get("CCX", 0) * 7)
    print("    cx 折算 ≈ %d, u 折算 ≈ %d  (实验05 qiskit 转译: cx=29, u=53, 深度=59)"
          % (cx_est, u_est))
    print("    注: 本移植多控旋转用 2×TOFFOLI+2×CRZ 构造, 比 qiskit UnitaryGate")
    print("        转译更重; 数值不可直接比较, 仅供真机时序预估量级参考。")
    # 噪声友好版 (噪声阶段实际使用的等价电路)
    ops_nf = build_full_circuit(solve_dvals(), noise_friendly=True).count_ops()
    cx_nf = (ops_nf.get("CNOT", 0) * 1 + ops_nf.get("CP", 0) * 2 +
             ops_nf.get("CU", 0) * 2 + ops_nf.get("SWAP", 0) * 3)
    print("    噪声友好版 (噪声阶段):", dict(sorted(ops_nf.items())),
          "-> cx 折算 ≈ %d" % cx_nf)


# ============================================================
# 8. OriginIR 导出 (真机提交格式)
# ============================================================
def export_originir(prog, path):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(path))
    print("[7] OriginIR 导出 -> %s" % path)
    originir = prog.originir(precision=15)
    with open(path, "w", encoding="utf-8") as f:
        f.write(originir)
    print("    前 12 行预览:")
    for line in originir.splitlines()[:12]:
        print("      " + line)
    return originir


# ============================================================
# 主流程
# ============================================================
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    assert mode in ("local", "noise", "all", "chip"), \
        "用法: python 07_pyqpanda3_port.py [local|noise|all|chip]"

    print("=" * 70)
    print("[0] 经典参考 + d 值数值求解")
    print("=" * 70)
    print("  y = A_c x =", np.round(y, 4), " (期望 (11,9,9,11)/√404)")
    print("  |y|² = %.4f  ->  P(anc=0) = 404/480 = %.6f"
          % ((np.abs(y) ** 2).sum(), 404 / 480))
    dvals = solve_dvals()
    print("  d (数值解) =", np.round(dvals, 6), " 自检通过 (对角化误差 <1e-12)")
    check("diag(d) = Q·(A_c/4)·Q† 对角化", True)

    print()
    print("=" * 70)
    print("[1] 平台约定校准 (探针电路实测)")
    print("=" * 70)
    qvm, sv_w, keypos = calibrate()
    sv_desc = {q: ("2^%d" % int(round(np.log2(sv_w[q])))) for q in (0, 1, 2)}
    print("  状态向量索引: q0=%s q1=%s q2=%s (低位 q%d)"
          % (sv_desc[0], sv_desc[1], sv_desc[2],
             [q for q in (0, 1, 2) if sv_w[q] == 1][0]))
    print("  counts/prob 键序: 位置[q0]=%d 位置[q1]=%d 位置[q2]=%d"
          % (keypos[0], keypos[1], keypos[2]))
    check("三约定校准完成", True)

    print()
    selfcheck_subcircuits(dvals)

    print()
    print("=" * 70)
    prog, circuit = verify_exact(qvm, keypos, dvals)
    print("=" * 70)

    print()
    ideal_distribution(qvm, prog, keypos)

    if mode in ("noise", "all"):
        print()
        print("=" * 70)
        noise_stage(qvm, dvals, keypos)
        print("=" * 70)

    print()
    resource_ledger(circuit)

    print()
    out_path = "07_pyqpanda3_originir.txt"
    export_originir(prog, out_path)
    if mode == "chip":
        print("""
  [chip 模式] pyqpanda3 0.4.0 的 Python 绑定尚未暴露云 API (pilot_os 为空),
  请将上面导出的 %s 通过本源量子云控制台提交, 或回退到实验 06
  (pyqpanda 3.8.5, 已实现 QCloud / real_chip_measure / get_state_fidelity)。
  真机预期: P(anc=0) < 0.8417 (真实噪声), 后选择保真度 < 1 —— 把实测
  保真度代回实验 05 §5.3 的公式即得该硬件上的墙钟交叉点。""" % out_path)

    print()
    print("=" * 70)
    n_fail = VERDICT.count(False)
    print("判定汇总: %d 项, 通过 %d, 失败 %d" % (len(VERDICT), len(VERDICT) - n_fail, n_fail))
    print("=" * 70)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
