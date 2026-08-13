# -*- coding: utf-8 -*-
"""
实验 06: 在本源量子平台测试 n=4 量子卷积电路 (LCU 块编码)
================================================================
电路: 4 比特
  q0, q1 : 系统比特 (幅值编码输入 x=(3,1,4,2)/sqrt(30))
  q2     : 工作比特 (Toffoli 分解用, 保持 |0>)
  q3     : 辅助比特 (块编码 ancilla, 后选择 |0>)
结构: 态制备 -> QFT2 -> 4 个双控块 B_k (LCU) -> QFT2^dagger -> 测量

期望结果 (理论精确值):
  P(q3=0) = 404/480 = 0.8417
  后选择系统分布: |00>:0.2995  |01>:0.2005  |10>:0.2005  |11>:0.2995
  (标签顺序取决于平台比特序约定, 数值集合 {0.2995, 0.2005, 0.2005, 0.2995} 不变)

用法 (无需参数默认本地验证; 云端需设置环境变量 ORIGIN_QC_TOKEN):
  python 06_origin_quantum.py            # 本地全振幅模拟 (免费, 无需账号)
  python 06_origin_quantum.py cloud      # 本源量子云高性能集群
  python 06_origin_quantum.py noise      # 云端噪声虚拟机 (对照实验05的Aer结果)
  python 06_origin_quantum.py chip       # 真实芯片 (悟空72 / 悟源5号, 需开通算力)
================================================================
"""
import builtins as _builtins
import sys

import numpy as np

from pyqpanda import *

_pysum = _builtins.sum  # pyqpanda 会遮蔽内建 sum

# ---------------- 期望值 ----------------
P_ANC0_TARGET = 404.0 / 480.0          # 0.8417
Y_TARGET = np.array([11, 9, 9, 11], float) / np.sqrt(404)  # anc=0 分量

# ---------------- B 块参数 (pyqpanda 约定实测: CU(a,b,g,d) = e^{ia} RZ(b) RY(g) RZ(d)) ----------------
# RZ(phi) = diag(e^{-i*phi/2}, e^{+i*phi/2}); 每分支相位精确, 不可近似
# 分支(00): B=Z ; (01): B=[[i/2, s],[s, i/2]] ; (10): B=X ; (11): B=[[-i/2, s],[s, -i/2]]
B_PARAMS = {
    0: (np.pi / 2, np.pi, 0.0, 0.0),         # Z
    1: (np.pi / 2, -np.pi / 2, 2 * np.pi / 3, np.pi / 2),   # d=+i/2
    2: (np.pi / 2, 0.0, np.pi, np.pi),        # X
    3: (np.pi / 2, -np.pi / 2, 4 * np.pi / 3, np.pi / 2),   # d=-i/2
}


def state_prep(prog, q):
    """制备 (3,1,4,2)/sqrt(30), qstat 索引 = q0 + 2*q1 (q[0] 为低位, 实测确认)
    RY(q0): q0=1 权重 = (x[1]^2+x[3]^2)/30 = 5/30 -> cos^2 = 5/6
    q0=0 分支: q1 按 x[0]:x[2] = 3:4 分裂;  q0=1 分支: 按 x[1]:x[3] = 1:2"""
    t0 = 2 * np.arccos(np.sqrt(5.0 / 6.0))   # q0
    t00 = 2 * np.arccos(3.0 / 5.0)           # q1 | q0=0: 3:4
    t01 = 2 * np.arccos(1.0 / np.sqrt(5.0))  # q1 | q0=1: 1:2
    prog << RY(q[0], t0)
    # q1 受控于 q0=0: X 翻转控制位
    prog << X(q[0])
    prog << CU(0, 0, t00, 0, q[0], q[1])
    prog << X(q[0])
    # q1 受控于 q0=1
    prog << CU(0, 0, t01, 0, q[0], q[1])


def qft2(prog, q, theta):
    """QFT2 = H(q0) CP(theta) H(q1); theta=+pi/2 为 qiskit 式约定
    注意: pyqpanda 3.8.5 链式 << 疑似重排 CU 位置, 若验证不通过,
    请改逐句构建 (每门单独一行 prog << ...) 见 docs/origin-quantum-guide.md 调试状态"""
    prog << H(q[0])
    prog << CU(theta, 0, 0, 0, q[0], q[1])
    prog << H(q[1])


def diag_block(prog, q, branch, params):
    """双控 B 块: Toffoli -> CU -> Toffoli, 分支 (b0,b1) 为 0 的位先 X 翻转 (逐句构建, 防链式重排)"""
    b0, b1 = branch
    if b0 == 0:
        prog << X(q[0])
    if b1 == 0:
        prog << X(q[1])
    a, b, g, d = params
    prog << Toffoli(q[0], q[1], q[2])
    prog << CU(a, b, g, d, q[2], q[3])
    prog << Toffoli(q[0], q[1], q[2])
    if b1 == 0:
        prog << X(q[1])
    if b0 == 0:
        prog << X(q[0])


def build_prog(q, c, theta, swap_13=False):
    """构建完整线路。swap_13=True 时交换分支 01/11 的 B 块 (对应 QFT 共轭约定)"""
    prog = QProg()
    state_prep(prog, q)
    qft2(prog, q, theta)
    branch_of = {0: 0, 1: (3 if swap_13 else 1), 2: 2, 3: (1 if swap_13 else 3)}
    for m in range(4):
        diag_block(prog, q, (m // 2, m % 2), B_PARAMS[branch_of[m]])
    qft2(prog, q, -theta)
    prog << Measure(q[3], c[3]) << Measure(q[0], c[0]) << Measure(q[1], c[1])
    return prog


def report(name, counts, shots):
    """统计 P(anc=0) 与后选择分布, 与理论对照; counts 为 {比特串: 次数} 字典"""
    print("[%s]" % name)
    p0 = _pysum(v for k, v in counts.items() if k[3] == "0") / shots
    print("  P(anc=0) = %.4f   (理论 0.8417, 偏差 %.3f)" % (p0, abs(p0 - P_ANC0_TARGET)))
    sub = {k[0:3]: v for k, v in counts.items() if k[3] == "0"}
    tot = _pysum(sub.values())
    dist = {k: round(v / tot, 4) for k, v in sub.items()} if tot else {}
    print("  后选择系统分布:", dist)
    print("  理论集合 {0.2995, 0.2005, 0.2005, 0.2995} (标签序取决于平台比特序)")
    return p0


def local_verify():
    """本地全振幅模拟, 自动判定 QFT 相位约定"""
    init_quantum_machine(QMachineType.CPU)
    q = qAlloc_many(4)
    c = cAlloc_many(4)
    for theta, swap in ((np.pi / 2, False), (-np.pi / 2, True)):
        prog = build_prog(q, c, theta, swap)
        result = run_with_configuration(prog, c, 100000)
        p0 = report("LOCAL theta=%+.2f swap13=%s" % (theta, swap), result, 100000)
        if abs(p0 - P_ANC0_TARGET) < 0.01:
            print("  => 约定判定成功: theta=%+.2f, swap13=%s\n" % (theta, swap))
            destroy_quantum_machine()
            return theta, swap
    print("  警告: 两种约定均未命中 0.8417, 请检查 pyqpanda 版本 API 差异")
    destroy_quantum_machine()
    return np.pi / 2, False


def cloud_stage(mode, theta, swap):
    """云端集群 / 噪声 / 真实芯片; token 从环境变量 ORIGIN_QC_TOKEN 读取 (勿硬编码进仓库)"""
    import os
    TOKEN = os.environ.get("ORIGIN_QC_TOKEN", "")
    if not TOKEN:
        print("请设置环境变量: set ORIGIN_QC_TOKEN=<你的 api_token>  (或临时导出, 勿写入仓库文件)")
        return
    qm = QCloud()
    qm.set_configure(72, 72)
    qm.init_qvm(TOKEN)
    q = qm.qAlloc_many(4)
    c = qm.cAlloc_many(4)

    if mode == "noise":
        # 对照实验05: 退极化 p 扫描
        prog = build_prog(q, c, theta, swap)
        for p in (1e-4, 1e-3, 1e-2):
            qm.set_noise_model(NoiseModel.DEPOLARIZING_KRAUS_OPERATOR, [p], [p])
            result = qm.noise_measure(prog, 100000)
            report("NOISE p=%.0e" % p, result, 100000)
    elif mode == "cloud":
        prog = build_prog(q, c, theta, swap)
        result = qm.full_amplitude_measure(prog, 100000)
        report("CLOUD 全振幅", result, 100000)
    elif mode == "chip":
        prog = build_prog(q, c, theta, swap)
        # 悟空 72 比特 (新) 或悟源 5 号 (chip_id=2, 默认)
        # 机时有限时建议 2000 shots: 分布统计误差 ~1%, 足以与理论对照
        SHOTS = int(os.environ.get("ORIGIN_CHIP_SHOTS", "2000"))
        result = qm.real_chip_measure(prog, SHOTS, real_chip_type.origin_72)
        report("CHIP 悟空72 (%d shots)" % SHOTS, result, SHOTS)
        # 可选: 直接拿状态保真度 (平台内置接口)
        # fid = qm.get_state_fidelity(prog, SHOTS, real_chip_type.origin_72)
        # print("  chip 状态保真度:", fid)
    qm.finalize()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    theta, swap = local_verify()          # 本地先判定约定 (免费)
    if mode != "local":
        cloud_stage(mode, theta, swap)
