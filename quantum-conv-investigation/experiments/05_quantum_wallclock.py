# -*- coding: utf-8 -*-
"""
量子侧收尾实验: 把门数账本转成墙钟时间
  1. n=4 精确电路验证 (QFT + LCU 块编码 + QFT†) vs 解析目标态
  2. 电路转译: cx 数 / 深度
  3. 噪声仿真: 退极化错误率 p 下成功概率与输出分布保真度
  4. 解析 T 门成本 @ n=32..65536, 换算墙钟时间, 找交叉点
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate, StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

print("=" * 70)
print("[1] n=4 精确电路验证: c=(1,2,1,0), x=(3,1,4,2) -> y=(11,9,9,11)")
print("=" * 70)

x = np.array([3, 1, 4, 2], float) / np.sqrt(30)
y_target = np.array([11, 9, 9, 11], float) / np.sqrt(404)

qc = QuantumCircuit(3, 3)
# 振幅按 sv 基序直接给出 (StatePreparation 即 sv 序)
qc.append(StatePreparation(x, normalize=False), [0, 1])
# qiskit QFT = Fideal^dagger, 故 diag 需放 d 的反转序: (d0, d3, d2, d1)
qc.append(QFTGate(2), [0, 1])
# 分支块: m=0: d=1 -> Z ; m=1: d=+i/2 ; m=2: d=0 -> X ; m=3: d=-i/2
# B(d) = [[d, s],[s, -d*]], s = sqrt(1-|d|^2)
s = np.sqrt(3.0) / 2
B = {
    0: np.array([[1, 0], [0, -1]], complex),
    1: np.array([[0.5j, s], [s, 0.5j]], complex),
    2: np.array([[0, 1], [1, 0]], complex),
    3: np.array([[-0.5j, s], [s, -0.5j]], complex),
}
for k, ctrl in ((0, 0), (1, 1), (2, 2), (3, 3)):
    qc.append(UnitaryGate(B[k]).control(2, ctrl_state=ctrl), [0, 1, 2])
qc.append(QFTGate(2).inverse(), [0, 1])

sv = Statevector(qc)
amp0 = sv.data[0:4]  # 辅助比特 q2 = 0 的分量
ovl = np.abs(np.vdot(amp0, y_target)) ** 2
print("  |<anc=0 分量 | y_target>|^2 = %.6f  (期望 404/480 = %.4f)"
      % (ovl, 404 / 480))
print("  anc=0 分量 (未归一):", np.round(amp0, 4))
print("  y_target            :", np.round(y_target, 4))

# 测量仿真 (理想, 无噪声)
qc_meas = transpile(qc.copy(), basis_gates=["cx", "u"], optimization_level=3)
qc_meas.measure([0, 1, 2], [0, 1, 2])
sim = AerSimulator()
counts = sim.run(qc_meas, shots=200000).result().get_counts()
# 计数键: 首字符 = 最高索引比特 = q2 (辅助), 后两字符 = (q1, q0)
p0 = sum(v for k, v in counts.items() if k[0] == "0") / 200000
print("  理想采样: P(anc=0) = %.4f (期望 0.8417)" % p0)
sub = {k[1:3]: v for k, v in counts.items() if k[0] == "0"}
tot = sum(sub.values())
print("  后选择系统分布:", {k: round(v / tot, 4) for k, v in sorted(sub.items())})
print("  期望分布      :", {"00": 0.2995, "01": 0.2005, "10": 0.2005, "11": 0.2995})

print()
print("=" * 70)
print("[2] 转译资源: cx 数 / 深度")
print("=" * 70)
qct = transpile(qc, basis_gates=["cx", "u"], optimization_level=3)
ops = qct.count_ops()
print("  cx = %d, u = %d, 深度 = %d" % (ops.get("cx", 0), ops.get("u", 0), qct.depth()))

print()
print("=" * 70)
print("[3] 噪声仿真: 退极化 p 对成功概率与输出保真度的影响")
print("=" * 70)
ideal_dist = np.array([121, 81, 81, 121], float) / 404
for p in (1e-4, 1e-3, 1e-2):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["u"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"])
    qcn = transpile(qc_meas, basis_gates=["cx", "u"], optimization_level=3)
    simn = AerSimulator(noise_model=nm)
    cn = simn.run(qcn, shots=200000).result().get_counts()
    p0n = sum(v for k, v in cn.items() if k[0] == "0") / 200000
    subn = {k[1:3]: v for k, v in cn.items() if k[0] == "0"}
    totn = sum(subn.values())
    if totn > 0:
        qn = np.array([subn.get("00", 0), subn.get("01", 0), subn.get("10", 0), subn.get("11", 0)], float) / totn
        fid = (np.sum(np.sqrt(qn * ideal_dist))) ** 2
    else:
        fid = 0.0
    print("  p=%.0e  P(anc=0)=%.4f (理想 0.8417)  后选择分布保真度 F=%.4f  采样开销 x%.2f"
          % (p, p0n, fid, 0.8417 / max(p0n, 1e-9)))

print()
print("=" * 70)
print("[4] 解析 T 门成本 + 墙钟时间换算 (秩-R LCU, k=2 标量, eps=0.1 -> 20 次查询)")
print("=" * 70)


def T_rotation(m):
    # m 控制比特的多控单比特旋转, 借 1 辅助比特: ~8m T (Barenco 分解)
    return 8 * m + 20


def T_per_query_rankR(n, R):
    q = int(np.log2(n))
    qft = q * (q + 1)  # 2D QFT 控制相位门, 可忽略级
    diag = 2 * R * n * T_rotation(q)
    return qft + diag


def T_per_query_arb(n):
    q = int(np.log2(n))
    return n * n * T_rotation(2 * q)


def classical_MAC(n):
    return 30 * n * n * np.log2(n)


QUERIES = 20
print("  n      经典MAC     经典时间@1ns   量子T(R=2)   量子T(R=1)   量子T(任意核)")
for n in (32, 128, 512, 1024, 4096, 16384, 65536):
    cmac = classical_MAC(n)
    t2 = QUERIES * T_per_query_rankR(n, 2)
    t1 = QUERIES * T_per_query_rankR(n, 1)
    ta = QUERIES * T_per_query_arb(n)
    print("  %-7d %9.1e   %8.4f s    %9.1e  %9.1e  %9.1e"
          % (n, cmac, cmac * 1e-9, t2, t1, ta))

print()
print("  交叉点 (量子 R=2 vs 经典, 量子时间 = T x t_T):")
for tT in (1e-7, 1e-6, 1e-5):
    best = None
    for n in (32, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144):
        qtime = QUERIES * T_per_query_rankR(n, 2) * tT
        ctime = classical_MAC(n) * 1e-9
        if qtime < ctime:
            best = n
            break
    print("  逻辑 T = %.1f us : 交叉点 n ~= %s (图像尺寸 ~= %s)"
          % (tT * 1e6, best if best else ">2.6e5", ("%dx%d" % (int(np.sqrt(best)), int(np.sqrt(best)))) if best else ">512x512"))

print("\n完成。")
