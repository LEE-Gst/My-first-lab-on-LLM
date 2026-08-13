# -*- coding: utf-8 -*-
"""
截断实验: 训练核 rank-R 截断在任务层面的真实代价 (MNIST)
  A. 基线 (完整 5 epoch 训练)
  B. rank-2 截断, 全部冻结        -> 纯截断代价
  C. rank-1 截断, 全部冻结
  D. rank-2 截断, 冻结 conv1, 精调下游 3 epoch
  E. rank-2 分解参数化, 全部精调 3 epoch
  F. rank-1 分解参数化, 全部精调 3 epoch
输出: 各方案测试准确率 + 相对基线的损失 + 与量子优势因子的联动
"""
import gzip
import os
import urllib.request

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

rng = np.random.default_rng(7)
np.set_printoptions(precision=3, suppress=True)

TMP = r"C:\Users\31615\AppData\Local\Temp\opencode\mnist"
os.makedirs(TMP, exist_ok=True)
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
                return p
            os.remove(p)
        except Exception:
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


print("loading MNIST (cached) ...")
trX = load_img(fetch("train-images-idx3-ubyte.gz"))
trY = load_lbl(fetch("train-labels-idx1-ubyte.gz"))
teX = load_img(fetch("t10k-images-idx3-ubyte.gz"))
teY = load_lbl(fetch("t10k-labels-idx1-ubyte.gz"))
NTR = 10000
trX, trY = trX[:NTR], trY[:NTR]

K1, K2 = 8, 16


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


class Net:
    def __init__(self):
        self.W1 = rng.normal(0, 0.3, (K1, 3, 3, 1)).astype(np.float32)
        self.b1 = np.zeros(K1, np.float32)
        self.W2 = rng.normal(0, 0.2, (K2, 2, 2, K1)).astype(np.float32)
        self.b2 = np.zeros(K2, np.float32)
        self.W3 = rng.normal(0, 0.1, (10, 6 * 6 * K2)).astype(np.float32)
        self.b3 = np.zeros(10, np.float32)
        self.vW1 = np.zeros_like(self.W1)
        self.vW2 = np.zeros_like(self.W2)
        self.vW3 = np.zeros_like(self.W3)

    def forward(self, X):
        c1 = im2col(X[..., None], 3, 3, 1) @ self.W1.reshape(K1, -1).T + self.b1
        c1 = c1.reshape(-1, 26, 26, K1)
        a1 = np.maximum(c1, 0)
        p1, idx1 = maxpool(a1)
        c2 = im2col(p1, 2, 2, K1) @ self.W2.reshape(K2, -1).T + self.b2
        c2 = c2.reshape(-1, 12, 12, K2)
        a2 = np.maximum(c2, 0)
        p2, idx2 = maxpool(a2)
        flat = p2.reshape(-1, 6 * 6 * K2)
        scores = flat @ self.W3.T + self.b3
        return scores, (c1, a1, p1, idx1, c2, a2, p2, idx2, flat)

    def backward(self, X, Yoh, probs, cache, lr, train1=True, factored=None):
        c1, a1, p1, idx1, c2, a2, p2, idx2, flat = cache
        N = X.shape[0]
        d = (probs - Yoh) / N

        dW3 = d.T @ flat
        db3 = d.sum(0)
        dp2 = (d @ self.W3).reshape(-1, 6, 6, K2)
        da2 = maxpool_back(dp2, idx2, 12, 12)
        dc2 = da2 * (c2 > 0)

        cols2 = im2col(p1, 2, 2, K1)
        dW2 = np.einsum('npi,npk->ik', cols2, dc2.reshape(-1, 144, K2)).reshape(2, 2, K1, K2).transpose(3, 0, 1, 2)
        db2 = dc2.reshape(-1, 144, K2).sum((0, 1))

        dc2pad = np.pad(dc2, ((0, 0), (1, 1), (1, 1), (0, 0)))
        cols2pad = im2col(dc2pad, 2, 2, K2)
        W2f = self.W2[:, ::-1, ::-1, :].transpose(1, 2, 0, 3).reshape(2 * 2 * K2, K1)
        dp1 = (cols2pad @ W2f).reshape(-1, 13, 13, K1)

        da1 = maxpool_back(dp1, idx1, 26, 26)
        dc1 = da1 * (c1 > 0)
        cols1 = im2col(X[..., None], 3, 3, 1)
        dW1 = np.einsum('npi,npk->ik', cols1, dc1.reshape(-1, 676, K1)).reshape(3, 3, 1, K1).transpose(3, 0, 1, 2)
        db1 = dc1.reshape(-1, 676, K1).sum((0, 1))

        self.vW2 = 0.9 * self.vW2 + dW2
        self.W2 -= lr * self.vW2
        self.b2 -= lr * db2
        self.vW3 = 0.9 * self.vW3 + dW3
        self.W3 -= lr * self.vW3
        self.b3 -= lr * db3

        if not train1:
            return
        if factored is None:
            self.vW1 = 0.9 * self.vW1 + dW1
            self.W1 -= lr * self.vW1
            self.b1 -= lr * db1
        else:
            U, V = factored
            for k in range(K1):
                dWk = dW1[k, :, :, 0]
                Uk, Vk = U[k], V[k]
                dU = dWk @ Vk.T
                dV = Uk.T @ dWk
                U[k] = Uk - lr * dU
                V[k] = Vk - lr * dV
                self.W1[k, :, :, 0] = U[k] @ V[k]
            self.b1 -= lr * db1


def eval_acc(net, Xs, Ys):
    correct = 0
    for i in range(0, len(Xs), 500):
        sc, _ = net.forward(Xs[i:i + 500])
        correct += (sc.argmax(1) == Ys[i:i + 500]).sum()
    return correct / len(Xs)


def train_epochs(net, epochs, lrs, train1=True, factored=None, tag=""):
    for ep in range(epochs):
        lr = lrs[ep] if ep < len(lrs) else lrs[-1]
        perm = rng.permutation(NTR)
        tot = 0.0
        nb = 0
        for i in range(0, NTR, 64):
            idx = perm[i:i + 64]
            Xb, Yb = trX[idx], trY[idx]
            Yoh = np.eye(10)[Yb]
            scores, cache = net.forward(Xb)
            probs = softmax(scores)
            tot += -np.mean(np.log(probs[np.arange(len(idx)), Yb] + 1e-12))
            nb += 1
            net.backward(Xb, Yoh, probs, cache, lr, train1=train1, factored=factored)
        acc = eval_acc(net, teX, teY)
        print("  %s epoch %d  lr=%.2f  loss=%.4f  test_acc=%.4f" % (tag, ep + 1, lr, tot / nb, acc))
    return eval_acc(net, teX, teY)


def truncate_W1(W1, R):
    Wt = np.zeros_like(W1)
    for k in range(K1):
        U, s, Vt = np.linalg.svd(W1[k, :, :, 0])
        Wt[k, :, :, 0] = (U[:, :R] * s[:R]) @ Vt[:R]
    return Wt


def factored_init(W1, R):
    U = np.zeros((K1, 3, R), np.float32)
    V = np.zeros((K1, R, 3), np.float32)
    for k in range(K1):
        Uu, s, Vt = np.linalg.svd(W1[k, :, :, 0])
        U[k] = Uu[:, :R] * s[:R]
        V[k] = Vt[:R]
    return U, V


# ================= A. 基线训练 =================
print("\n[A] 基线: 完整训练 5 epoch")
net = Net()
acc_A = train_epochs(net, 5, [0.1, 0.1, 0.1, 0.02, 0.02], tag="base")

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
np.savez(os.path.join(_DATA, "mnist_full.npz"),
         W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2, W3=net.W3, b3=net.b3)

# ================= B/C. 纯截断, 全冻结 =================
print("\n[B/C] 纯截断 (全冻结, 不重训)")
W1_full = net.W1.copy()
acc_B = None
acc_C = None
for R, tag in ((2, "B"), (1, "C")):
    net.W1 = truncate_W1(W1_full, R)
    a = eval_acc(net, teX, teY)
    print("  [%s] rank-%d 截断冻结: test_acc=%.4f (基线 %.4f, 差 %.2f%%)"
          % (tag, R, a, acc_A, 100 * (acc_A - a)))
    if R == 2:
        acc_B = a
    else:
        acc_C = a

# ================= D. 截断 + 冻结 conv1, 精调下游 =================
print("\n[D] rank-2 截断, 冻结 conv1, 精调下游 3 epoch")
net.W1 = truncate_W1(W1_full, 2)
acc_D = train_epochs(net, 3, [0.02, 0.02, 0.01], train1=False, tag="ft-down")

# ================= E/F. 分解参数化, 全部精调 =================
print("\n[E] rank-2 分解参数化, 全部精调 3 epoch")
net.W1 = truncate_W1(W1_full, 2)
U, V = factored_init(net.W1, 2)
acc_E = train_epochs(net, 3, [0.02, 0.02, 0.01], train1=True, factored=(U, V), tag="ft-r2")

print("\n[F] rank-1 分解参数化, 全部精调 3 epoch")
net.W1 = truncate_W1(W1_full, 1)
U, V = factored_init(net.W1, 1)
acc_F = train_epochs(net, 3, [0.02, 0.02, 0.01], train1=True, factored=(U, V), tag="ft-r1")

# ================= 汇总 =================
print("\n" + "=" * 70)
print("[汇总] rank 截断在任务层面的代价 (基线 %.4f)" % acc_A)
print("=" * 70)
rows = [
    ("A 基线 (R=3 完整)", acc_A, "-"),
    ("B R=2 冻结", acc_B, 100 * (acc_A - acc_B)),
    ("C R=1 冻结", acc_C, 100 * (acc_A - acc_C)),
    ("D R=2 冻结conv1+精调下游", acc_D, 100 * (acc_A - acc_D)),
    ("E R=2 分解全精调", acc_E, 100 * (acc_A - acc_E)),
    ("F R=1 分解全精调", acc_F, 100 * (acc_A - acc_F)),
]
for name, a, diff in rows:
    print("  %-26s acc=%.4f  差 %s" % (name, a, "-" if diff == "-" else "%.2f%%" % diff))

print("\n完成。")
