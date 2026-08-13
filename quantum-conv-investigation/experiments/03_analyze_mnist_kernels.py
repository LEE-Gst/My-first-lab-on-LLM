# -*- coding: utf-8 -*-
"""
基于 mnist_kernels.npz (真实训练核) 的补充分析:
  1. conv1 核 rank-R 截断的 Frobenius 相对误差 (R=1,2)
  2. 用实测 R 更新上一轮的优势因子账本
  3. 打印几个核的可视化 (定性检查是否真实边缘/纹理检测器)
"""
import os

import numpy as np

np.set_printoptions(precision=3, suppress=True)
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
d = np.load(os.path.join(_DATA, "mnist_kernels.npz"))
W1 = d["W1"]  # (8,3,3,1)
W2 = d["W2"]  # (16,2,2,8)

print("=" * 70)
print("[1] conv1 核 rank-R 截断的相对误差 (Frobenius)")
print("=" * 70)
err1, err2 = [], []
for k in range(8):
    M = W1[k].reshape(3, 3)
    U, s, Vt = np.linalg.svd(M)
    for R in (1, 2):
        Mr = (U[:, :R] * s[:R]) @ Vt[:R]
        e = np.linalg.norm(M - Mr) / np.linalg.norm(M)
        if R == 1:
            err1.append(e)
        else:
            err2.append(e)
print("  rank-1 截断: 平均相对误差 %.3f (min %.3f / max %.3f)"
      % (np.mean(err1), min(err1), max(err1)))
print("  rank-2 截断: 平均相对误差 %.3f (min %.3f / max %.3f)"
      % (np.mean(err2), min(err2), max(err2)))
print("  => 误差-成本权衡: R=1 丢 %.0f%% 能量, R=2 丢 %.0f%%, R=3 零误差"
      % (100 * np.mean(err1)**2, 100 * np.mean(err2)**2))

print()
print("=" * 70)
print("[2] 实测 R 下的优势因子 (n=32, 每图, 核摊销)")
print("=" * 70)


def cost_classical_fft(n):
    return 30 * n * n * np.log2(n) + n * n


def cost_quantum_rankR(n, R):
    q = int(np.log2(n))
    qft = q * (q + 1)
    diag = R * 2 * n * (4 * q + 2) + 200
    return qft + diag


n = 32
c_fft = cost_classical_fft(n)
print("  经典 2D-FFT 基准: %.0f MAC" % c_fft)
print("  %-28s | R=1(近似18%%)  R=2(近似6%%)  R=3(精确)" % "场景")
scenarios = [
    ("仅运行阶段", 1),
    ("k=2 标量, eps=0.1", 21),
    ("k=2 标量, eps=0.01", 201),
]
for label, mult in scenarios:
    row = []
    for R in (1, 2, 3):
        af = c_fft / (mult * cost_quantum_rankR(n, R))
        row.append("%7.1f" % af)
    print("  %-28s | %s" % (label, "   ".join(row)))

print()
print("=" * 70)
print("[3] 外推 (实测 R=2/3, k=2 标量, eps=0.1)")
print("=" * 70)
print("  n     经典FFT      R=2运行AF   R=2端到端AF   R=3端到端AF")
for nn in (32, 128, 512, 1024):
    cf = cost_classical_fft(nn)
    q2 = cost_quantum_rankR(nn, 2)
    q3 = cost_quantum_rankR(nn, 3)
    print("  %-6d %9.1e   %7.0f x    %7.0f x      %7.0f x"
          % (nn, cf, cf / q2, cf / (21 * q2), cf / (21 * q3)))

print()
print("=" * 70)
print("[4] conv1 训练核可视化 (x20 取整):")
print("=" * 70)
for k in range(8):
    M = W1[k].reshape(3, 3)
    v = np.round(M * 20).astype(int)
    print("  K%d:\n%s" % (k, v))
