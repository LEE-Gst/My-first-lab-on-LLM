# -*- coding: utf-8 -*-
"""
真实小卷积层测量实验
3x3 核, 32x32 输入, 二维循环卷积
测量目标:
  A. 核秩 R (SVD 奇异值谱)  -- 决定量子 LCU 项数
  B. 算子谱集中度 (FFT 域 |C_hat| 能量分布)  -- 决定 diag 加载成本
  C. 三种实现一致性验证 (直接 / FFT / 象限分裂 Cooley-Tukey)
  D. 经典 vs 量子成本账本 (n=32 及外推)
  E. 下游所需 "k 比特" (输出谱集中度 / 摘要统计量)
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(42)

n = 32
m = n // 2

print("=" * 70)
print("实验设置: n=%dx%d 输入, 3x3 核, 二维循环卷积" % (n, n))
print("=" * 70)

# ---------- 1. 输入图像 (1/f 谱合成自然感图像) ----------
fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n))
mag = 1.0 / (1.0 + 100.0 * (fx**2 + fy**2))
noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
X = np.real(np.fft.ifft2(np.fft.fft2(noise) * mag * n**2))
X -= X.min()
X /= X.max()

# ---------- 2. 四个真实核 ----------
kernels = {}
g = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        g[i, j] = np.exp(-((i - 1)**2 + (j - 1)**2) / (2 * 0.7**2))
kernels["Gaussian"] = g / g.sum()

kernels["SobelX"] = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 8.0

gb = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        gb[i, j] = np.exp(-((i - 1)**2 + (j - 1)**2) / 2.0) * np.cos(1.5 * (i - 1))
kernels["Gabor"] = gb

kernels["Random"] = rng.normal(size=(3, 3))


def pad(C):
    Z = np.zeros((n, n))
    Z[0:3, 0:3] = C
    return Z


# ---------- 3. 三种卷积实现 ----------
def conv_direct(X, Cp):
    """直接法 O(n^4): 显式构造双块循环矩阵 B"""
    B = np.zeros((n * n, n * n))
    for i in range(n):
        for j in range(n):
            B[i * n + j, :] = np.roll(
                np.roll(Cp[::-1, ::-1], i + 1, axis=0), j + 1, axis=1
            ).ravel()
    return (B @ X.ravel()).reshape(n, n)


def conv_fft(X, Cp):
    """FFT 法 O(n^2 log n)"""
    return np.real(np.fft.ifft2(np.fft.fft2(Cp) * np.fft.fft2(X)))


w = np.exp(-2j * np.pi / n)
u1 = np.arange(m)[:, None]
v1 = np.arange(m)[None, :]


def fft2_split(M):
    """一层 2D Cooley-Tukey: 四个象限半尺寸 FFT + 蝶形重组"""
    Q = {}
    for a in (0, 1):
        for b in (0, 1):
            tw = np.outer(w ** (a * u1.ravel()), w ** (b * v1.ravel()))
            Q[(a, b)] = tw * np.fft.fft2(M[a::2, b::2])
    F = np.zeros((n, n), complex)
    for u2 in (0, 1):
        for v2 in (0, 1):
            acc = np.zeros((m, m), complex)
            for a in (0, 1):
                for b in (0, 1):
                    acc += ((-1) ** (a * u2 + b * v2)) * Q[(a, b)]
            F[m * u2:m * u2 + m, m * v2:m * v2 + m] = acc
    return F


def ifft2_split(F):
    return np.conj(fft2_split(np.conj(F))) / (n * n)


def conv_split(X, Cp):
    """象限分裂法: 谱域分块计算 + 蝶形重组"""
    Ch = fft2_split(Cp)
    Xh = fft2_split(X)
    return np.real(ifft2_split(Ch * Xh))


# ---------- 4. 验证一致性 ----------
print("\n[验证] 三种实现一致性 (SobelX 核):")
Cp = pad(kernels["SobelX"])
err_split_fft = np.abs(fft2_split(X) - np.fft.fft2(X)).max()
Yd = conv_direct(X, Cp)
Yf = conv_fft(X, Cp)
Ys = conv_split(X, Cp)
print("  |split-FFT - numpy-FFT| max = %.2e" % err_split_fft)
print("  |direct - FFT|          max = %.2e" % np.abs(Yd - Yf).max())
print("  |direct - split|        max = %.2e" % np.abs(Yd - Ys).max())

# ---------- 5. 核秩 R 与算子谱集中度 ----------
print("\n[测量A] 核秩 R (3x3 核 SVD 奇异值) 与谱集中度:")
print("  kernel    sigma1  sigma2  sigma3   R_eff(0.1)  R_eff(0.01)")
for name, K in kernels.items():
    s = np.linalg.svd(K, compute_uv=False)
    r1 = int(np.sum(s > 0.1 * s[0]))
    r2 = int(np.sum(s > 0.01 * s[0]))
    print("  %-8s  %6.3f  %6.3f  %6.3f     %d           %d"
          % (name, s[0], s[1], s[2], r1, r2))

print("\n[测量B] 算子谱集中度 |C_hat| (1024 个谱值, 能量 99%% 所需数量):")
for name, K in kernels.items():
    Ch = np.fft.fft2(pad(K))
    e = np.sort(np.abs(Ch).ravel() ** 2)[::-1]
    cum = np.cumsum(e) / e.sum()
    m99 = int(np.searchsorted(cum, 0.99)) + 1
    print("  %-8s  m99 = %4d / 1024  (%.1f%%)" % (name, m99, 100.0 * m99 / n**2))

# ---------- 6. 输出侧: 下游需要哪 k 个比特 ----------
print("\n[测量C] 输出 Y = C * X 的信息结构 (SobelX):")
Y = conv_fft(X, pad(kernels["SobelX"]))
Yh = np.fft.fft2(Y)
e = np.sort(np.abs(Yh).ravel() ** 2)[::-1]
cum = np.cumsum(e) / e.sum()
m99 = int(np.searchsorted(cum, 0.99)) + 1
m999 = int(np.searchsorted(cum, 0.999)) + 1
p = np.abs(Y).ravel()
p = p / p.sum()
ent = -np.sum(p * np.log(p + 1e-12))
print("  输出谱集中度: 99%% 能量需 %4d/1024 个系数 (%.1f%%)" % (m99, 100.0 * m99 / n**2))
print("  输出谱集中度: 99.9%% 能量需 %4d/1024 个系数 (%.1f%%)" % (m999, 100.0 * m999 / n**2))
print("  输出分布熵: %.3f / %.3f (= log2 1024, 均匀分布基准)" % (ent, np.log(n * n)))
print("  输出统计: mean=%.4f  std=%.4f  min=%.4f  max=%.4f" %
      (Y.mean(), Y.std(), Y.min(), Y.max()))

# ---------- 7. 成本账本 ----------
print("\n[测量D] 成本账本 (n=32, 每处理一张图, 核已摊销):")


def cost_classical_direct(n):
    return n**4


def cost_classical_fft(n):
    return 30 * n * n * np.log2(n) + n * n


def cost_classical_split(n):
    mm = n // 2
    half = 4 * 2 * mm * 5 * mm * np.log2(mm)
    return 2 * half + 16 * n * n + n * n


def cost_quantum_arb(n):
    q = int(np.log2(n))
    qft = q * (q + 1)
    diag = n * n * (4 * q + 2)
    return qft + diag


def cost_quantum_rankR(n, R):
    q = int(np.log2(n))
    qft = q * (q + 1)
    diag = R * 2 * n * (4 * q + 2) + 200
    return qft + diag


c_direct = cost_classical_direct(n)
c_fft = cost_classical_fft(n)
c_split = cost_classical_split(n)
q_arb = cost_quantum_arb(n)

print("  经典 直接法 O(n^4)        : %10.0f MAC" % c_direct)
print("  经典 2D-FFT O(n^2 log n)  : %10.0f MAC" % c_fft)
print("  经典 象限分裂(一层)        : %10.0f MAC  (n=32 时反而多 %.0f%%)"
      % (c_split, 100 * (c_split / c_fft - 1)))
print("  量子 运行(任意核)          : %10.0f 门" % q_arb)
for R in (1, 2, 3):
    print("  量子 运行(秩-%d LCU)        : %10.0f 门" % (R, cost_quantum_rankR(n, R)))

print("\n[测量E] 优势因子 = 经典FFT成本 / 量子端到端成本:")
print("  (端到端 = (查询数+1) x 运行成本, 查询数 = k 个标量 x 1/eps)")
print("  场景                      | R=1     R=2     R=3     任意核")
scenarios = [
    ("仅运行阶段", 0, 0, 0.0),
    ("k=2 标量, eps=0.1", 2, 10, 0.1),
    ("k=2 标量, eps=0.01", 2, 100, 0.01),
    ("完整图像, eps=0.1", n * n, 1, 0.1),
]
for label, k, per, eps in scenarios:
    row = []
    for mode in ("R1", "R2", "R3", "arb"):
        if mode == "arb":
            run = q_arb
        else:
            run = cost_quantum_rankR(n, int(mode[1]))
        queries = k * per
        total = (queries + 1) * run
        af = c_fft / total
        row.append("%6.1f" % af)
    print("  %-28s | %s" % (label, "  ".join(row)))

# ---------- 8. 外推: 交叉点在哪 ----------
print("\n[测量F] 外推 (秩-2 核, k=2 标量读出, eps=0.1):")
print("  n     经典FFT    量子运行    运行AF   端到端AF")
for nn in (32, 128, 512, 1024):
    cf = cost_classical_fft(nn)
    qr = cost_quantum_rankR(nn, 2)
    run_af = cf / qr
    e2e = cf / (21 * qr)
    print("  %-6d %9.1e  %9.1e  %7.0f x  %7.0f x" % (nn, cf, qr, run_af, e2e))

print("\n完成。")
