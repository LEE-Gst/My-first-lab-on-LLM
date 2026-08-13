# -*- coding: utf-8 -*-
"""
MNIST 真实训练实验 (纯 numpy):
  训练两层小卷积网络 (conv1: 8x3x3, conv2: 16x3x3x8, fc)
  提取各层核, 测量: SVD 秩分布 / 可分性 / 谱集中度
目标: 把上一轮 "真实核 R=1~2" 从构造性结论变成统计性结论
"""
import gzip
import os
import urllib.request

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

rng = np.random.default_rng(7)
np.set_printoptions(precision=3, suppress=True)

# ---------- 数据 ----------
TMP = r"C:\Users\31615\AppData\Local\Temp\opencode\mnist"
os.makedirs(TMP, exist_ok=True)
BASE = "https://dataset.bj.bcebos.com/mnist/"
EXPECTED = {
    "train-images-idx3-ubyte.gz": 9912422,
    "train-labels-idx1-ubyte.gz": 28881,
    "t10k-images-idx3-ubyte.gz": 1648877,
    "t10k-labels-idx1-ubyte.gz": 4542,
}
MIRRORS = [
    "https://dataset.bj.bcebos.com/mnist/",
    "https://ghproxy.net/https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/",
    "https://ghfast.top/https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/",
]
FILES = {
    "trX": "train-images-idx3-ubyte.gz",
    "trY": "train-labels-idx1-ubyte.gz",
    "teX": "t10k-images-idx3-ubyte.gz",
    "teY": "t10k-labels-idx1-ubyte.gz",
}


def fetch(name):
    p = os.path.join(TMP, name)
    if os.path.exists(p) and os.path.getsize(p) == EXPECTED[name]:
        return p
    for base in MIRRORS:
        url = base + name
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                with open(p, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
            if os.path.getsize(p) == EXPECTED[name]:
                print("  ok %s <- %s" % (name, base))
                return p
            print("  size mismatch %s (%d != %d)" % (url, os.path.getsize(p), EXPECTED[name]))
            os.remove(p)
        except Exception as e:
            print("  mirror fail %s : %s" % (base, e))
            if os.path.exists(p):
                os.remove(p)
    raise RuntimeError("all mirrors failed for " + name)


def load_img(p):
    with gzip.open(p, "rb") as f:
        d = f.read()
    return np.frombuffer(d, np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32) / 255.0


def load_lbl(p):
    with gzip.open(p, "rb") as f:
        d = f.read()
    return np.frombuffer(d, np.uint8, offset=8)


print("loading MNIST ...")
trX = load_img(fetch(FILES["trX"]))
trY = load_lbl(fetch(FILES["trY"]))
teX = load_img(fetch(FILES["teX"]))
teY = load_lbl(fetch(FILES["teY"]))
print("train", trX.shape, "test", teX.shape)

NTR = 10000
trX, trY = trX[:NTR], trY[:NTR]

# ---------- 网络结构 ----------
K1, K2 = 8, 16
W1 = rng.normal(0, 0.3, (K1, 3, 3, 1)).astype(np.float32)
b1 = np.zeros(K1, np.float32)
W2 = rng.normal(0, 0.2, (K2, 2, 2, K1)).astype(np.float32)
b2 = np.zeros(K2, np.float32)
W3 = rng.normal(0, 0.1, (10, 6 * 6 * K2)).astype(np.float32)
b3 = np.zeros(10, np.float32)


def im2col(X, kh, kw, C):
    N, H, W, _ = X.shape
    sw = sliding_window_view(X, (1, kh, kw, C))
    return sw.reshape(N, (H - kh + 1) * (W - kw + 1), kh * kw * C)


def maxpool(X):
    N, H, W, C = X.shape
    Xr = X.reshape(N, H // 2, 2, W // 2, 2, C)
    tmp = Xr.transpose(0, 1, 3, 5, 2, 4).reshape(N, H // 2, W // 2, C, 4)
    return tmp.max(axis=4), tmp.argmax(axis=4)


def maxpool_back(dout, idx, H, W):
    N, h, w, C = dout.shape
    dX = np.zeros((N, H, W, C), dtype=np.float32)
    dXr = dX.reshape(N, h, 2, w, 2, C)
    a, b = np.unravel_index(idx, (2, 2))
    dXr[np.arange(N)[:, None, None, None],
        np.arange(h)[None, :, None, None], a,
        np.arange(w)[None, None, :, None], b,
        np.arange(C)[None, None, None, :]] = dout
    return dX


def softmax(s):
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def forward(X):
    c1 = im2col(X[..., None], 3, 3, 1) @ W1.reshape(K1, -1).T + b1
    c1 = c1.reshape(-1, 26, 26, K1)
    a1 = np.maximum(c1, 0)
    p1, idx1 = maxpool(a1)
    c2 = im2col(p1, 2, 2, K1) @ W2.reshape(K2, -1).T + b2
    c2 = c2.reshape(-1, 12, 12, K2)
    a2 = np.maximum(c2, 0)
    p2, idx2 = maxpool(a2)
    flat = p2.reshape(-1, 6 * 6 * K2)
    scores = flat @ W3.T + b3
    return scores, (c1, a1, p1, idx1, c2, a2, p2, idx2, flat)


def backward(X, Yoh, probs, cache):
    global W1, b1, W2, b2, W3, b3, vW1, vW2, vW3, vb1, vb2, vb3
    c1, a1, p1, idx1, c2, a2, p2, idx2, flat = cache
    N = X.shape[0]
    d = (probs - Yoh) / N

    dW3 = d.T @ flat
    db3 = d.sum(0)
    dp2 = (d @ W3).reshape(-1, 6, 6, K2)
    da2 = maxpool_back(dp2, idx2, 12, 12)
    dc2 = da2 * (c2 > 0)

    cols2 = im2col(p1, 2, 2, K1)
    dW2 = np.einsum('npi,npk->ik', cols2, dc2.reshape(-1, 144, K2)).reshape(2, 2, K1, K2).transpose(3, 0, 1, 2)
    db2 = dc2.reshape(-1, 144, K2).sum((0, 1))

    dc2pad = np.pad(dc2, ((0, 0), (1, 1), (1, 1), (0, 0)))
    cols2pad = im2col(dc2pad, 2, 2, K2)
    W2f = W2[:, ::-1, ::-1, :].transpose(1, 2, 0, 3).reshape(2 * 2 * K2, K1)
    dp1 = (cols2pad @ W2f).reshape(-1, 13, 13, K1)

    da1 = maxpool_back(dp1, idx1, 26, 26)
    dc1 = da1 * (c1 > 0)
    cols1 = im2col(X[..., None], 3, 3, 1)
    dW1 = np.einsum('npi,npk->ik', cols1, dc1.reshape(-1, 676, K1)).reshape(3, 3, 1, K1).transpose(3, 0, 1, 2)
    db1 = dc1.reshape(-1, 676, K1).sum((0, 1))

    # 动量 SGD
    vW1 = 0.9 * vW1 + dW1; W1 -= lr * vW1; b1 -= lr * db1
    vW2 = 0.9 * vW2 + dW2; W2 -= lr * vW2; b2 -= lr * db2
    vW3 = 0.9 * vW3 + dW3; W3 -= lr * vW3; b3 -= lr * db3


vW1 = np.zeros_like(W1); vW2 = np.zeros_like(W2); vW3 = np.zeros_like(W3)
vb1 = np.zeros_like(b1); vb2 = np.zeros_like(b2); vb3 = np.zeros_like(b3)
vb1 = vb2 = vb3 = None  # 不用动量偏置, 简单起见


def eval_acc(Xs, Ys):
    correct = 0
    for i in range(0, len(Xs), 500):
        Xb = Xs[i:i + 500]
        sc, _ = forward(Xb)
        correct += (sc.argmax(1) == Ys[i:i + 500]).sum()
    return correct / len(Xs)


print("training ...")
BATCH = 64
EPOCHS = 5
for ep in range(EPOCHS):
    lr = 0.1 if ep < 3 else 0.02
    perm = rng.permutation(NTR)
    tot = 0.0
    nb = 0
    for i in range(0, NTR, BATCH):
        idx = perm[i:i + BATCH]
        Xb, Yb = trX[idx], trY[idx]
        Yoh = np.eye(10)[Yb]
        scores, cache = forward(Xb)
        probs = softmax(scores)
        tot += -np.mean(np.log(probs[np.arange(len(idx)), Yb] + 1e-12))
        nb += 1
        backward(Xb, Yoh, probs, cache)
    acc = eval_acc(teX, teY)
    print("  epoch %d  lr=%.2f  loss=%.4f  test_acc=%.4f" % (ep + 1, lr, tot / nb, acc))

# ---------- 保存核 ----------
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
np.savez(os.path.join(_DATA, "mnist_kernels.npz"),
         W1=W1, W2=W2, W3=W3)

# ---------- 测量: 秩分布 ----------
print("\n" + "=" * 70)
print("[测量] conv1 的 8 个 3x3 核 (单通道): SVD 秩 / 可分性")
print("=" * 70)
K1k = W1.reshape(K1, 3, 3)
r1_list = []
for k in range(K1):
    s = np.linalg.svd(K1k[k], compute_uv=False)
    reff1 = int(np.sum(s > 0.1 * s[0]))
    reff2 = int(np.sum(s > 0.01 * s[0]))
    sep = s[0] ** 2 / np.sum(s ** 2)
    r1_list.append((reff1, reff2, sep))
    print("  K%-2d sigma=(%.3f, %.3f, %.3f)  R_eff(0.1)=%d  R_eff(0.01)=%d  可分性=%.2f"
          % (k, s[0], s[1], s[2], reff1, reff2, sep))

print("\n[测量] conv2 的 16 个核 (每核 8 通道 3x3): 整体秩(9x8 矩阵)与逐通道均值")
print("  核  整体秩  整体可分性  通道R_eff均值(0.1)  通道R_eff均值(0.01)")
r2_list = []
for k in range(K2):
    M = W2[k].reshape(4, K1)  # 2*2=4 x 8通道
    s = np.linalg.svd(M, compute_uv=False)
    reff1 = int(np.sum(s > 0.1 * s[0]))
    sep = s[0] ** 2 / np.sum(s ** 2)
    ch_r1, ch_r2 = [], []
    for c in range(K1):
        sc = np.linalg.svd(W2[k, :, :, c], compute_uv=False)
        ch_r1.append(int(np.sum(sc > 0.1 * sc[0])))
        ch_r2.append(int(np.sum(sc > 0.01 * sc[0])))
    r2_list.append((reff1, sep, np.mean(ch_r1), np.mean(ch_r2)))
    print("  K%-2d   %d       %.2f          %.2f                  %.2f"
          % (k, reff1, sep, np.mean(ch_r1), np.mean(ch_r2)))

print("\n[汇总] 统计结论:")
r1_reff = [r[0] for r in r1_list]
print("  conv1: 8 个核 R_eff(0.1) 分布 -> 秩1: %d 个, 秩2: %d 个, 秩3: %d 个"
      % (sum(x == 1 for x in r1_reff), sum(x == 2 for x in r1_reff), sum(x == 3 for x in r1_reff)))
print("  conv1: 平均可分性 sigma1^2/总能量 = %.3f" % np.mean([r[2] for r in r1_list]))
print("  conv2: 16 个核 整体秩 均值 = %.2f, 通道R_eff(0.1)均值 = %.2f"
      % (np.mean([r[0] for r in r2_list]), np.mean([r[2] for r in r2_list])))
print("  conv2: 平均整体可分性 = %.3f" % np.mean([r[1] for r in r2_list]))

# 谱集中度 (与上一轮同口径: 零填充到 32x32)
print("\n[谱集中度] conv1 核零填充到 32x32, |C_hat| 99%% 能量所需系数:")
for k in range(K1):
    Cp = np.zeros((32, 32))
    Cp[0:3, 0:3] = K1k[k]
    Ch = np.fft.fft2(Cp)
    e = np.sort(np.abs(Ch).ravel() ** 2)[::-1]
    cum = np.cumsum(e) / e.sum()
    m99 = int(np.searchsorted(cum, 0.99)) + 1
    print("  K%-2d m99 = %4d / 1024" % (k, m99))
print("  对比上轮合成核: Gaussian 779, Sobel 608, Gabor 652, Random 933")
print("\n完成。")
