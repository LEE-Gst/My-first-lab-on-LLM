# -*- coding: utf-8 -*-
"""A2: 从已保存的 JSON 生成草稿用 5 张图。

数据取自仓内 results/ 子目录，输出到仓内 figures/。
无论从哪里调用本脚本都能工作（基于 __file__ 定位仓库根）。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RES = REPO_ROOT / "results"

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 9


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------- Fig1: M0 基线（qwen + R1） ----------------
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    qwen = load(RES / "m0" / "results_baseline.json")
    r1 = load(RES / "m0" / "r1_baseline.json")
    # n=50 corrected result for the B10 CoT L=50 outlier
    corr_path = RES / "m0" / "r1_b10_l50_cot_n50.json"
    corr_b10_cot = None
    if corr_path.exists():
        corr = load(corr_path)
        if corr.get("cell") == "B10_cot_L50" and corr["result"]["acc"] is not None:
            corr_b10_cot = corr["result"]["acc"]
    for ax, name in zip(axes, ["B6", "B10"]):
        e = qwen["automata"][name]
        Ls = sorted(int(k) for k in e["accuracy"])
        accs = [e["accuracy"][str(L)]["acc"] for L in Ls]
        ax.plot(Ls, accs, "o-", label="Qwen2.5-7B direct", color="#1f77b4")
        # R1
        if name in r1["automata"]:
            re_ = r1["automata"][name]["modes"]
            rd, rc = [], []
            for key, res in re_.items():
                mode, Ls_ = key.split("_L")
                if res["acc"] is not None:
                    (rd if mode == "direct" else rc).append((int(Ls_), res["acc"]))
            # replace the survivor-bias outlier with the n=50 corrected value
            if name == "B10" and corr_b10_cot is not None:
                rc = [(L, corr_b10_cot if L == 50 else a) for (L, a) in rc]
            for pts, lbl, col in [(sorted(rd), "R1-7B direct", "#d62728"), (sorted(rc), "R1-7B CoT", "#ff7f0e")]:
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], "s--", label=lbl, color=col)
        ax.axhline(0.5, ls=":", c="gray")
        ax.set_title(f"{name} (k={e['k']})")
        ax.set_xlabel("input length L")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("verdict accuracy")
    axes[0].legend(fontsize=8)
    fig.suptitle("Fig1. 7B models on balanced DFAs: B10 CoT L=50 corrected by n=50 rerun")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_m0_baseline.png")
    plt.close(fig)


# ---------------- Fig2: Phase1 课程化 ----------------
def fig2():
    m = load(RES / "phase1" / "curriculum_ceiling_metrics.json")
    new_acc, pool_acc = [], []
    labels = []
    for rec in m:
        if rec["step"] != "final":
            continue
        k, L = rec["k"], rec["L"]
        key = f"newk{k}L{L}"
        new_acc.append(rec[key]["acc"])
        pool_acc.append(rec["pool_acc"])
        labels.append(f"S{rec['stage']}\nk{k},L{L}")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, pool_acc, w, label="pool memorization", color="#8b949e")
    ax.bar(x + w / 2, new_acc, w, label="new-DFA generalization", color="#2ea043")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("full-trajectory accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Fig2. Phase 1 curriculum: k=3 generalizes, k≥4 cliff")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_curriculum.png")
    plt.close(fig)


# ---------------- Fig3: Executor vs 控制 Transformer ----------------
def fig3():
    ex = load(RES / "phase2" / "dfa_executor_final_extended_eval.json")
    ctrl = load(RES / "phase2" / "control_transformer_eval.json")
    ex_map = {(r["k"], r["L"]): r["acc"] for r in ex}
    ctrl_map = {(r["k"], r["L"]): r["acc"] for r in ctrl}
    ks = [3, 6, 10, 20, 32]
    Ls = [50, 100, 200, 500, 1000]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, mp, title in [(axes[0], ex_map, "Neural DFA Executor (232K)"),
                          (axes[1], ctrl_map, "Control Transformer (3.4M)")]:
        grid = np.full((len(ks), len(Ls)), np.nan)
        for i, k in enumerate(ks):
            for j, L in enumerate(Ls):
                if (k, L) in mp:
                    grid[i, j] = mp[(k, L)]
        ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(Ls)))
        ax.set_xticklabels(Ls, fontsize=8)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels(ks, fontsize=8)
        ax.set_xlabel("L")
        ax.set_title(title, fontsize=9)
        for i in range(len(ks)):
            for j in range(len(Ls)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=7)
    axes[0].set_ylabel("k")
    fig.suptitle("Fig3. Full-trajectory accuracy on NEW random DFAs (green=perfect)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_executor_vs_transformer.png")
    plt.close(fig)


# ---------------- Fig4: NL 中间层 ----------------
def fig4():
    v1 = load(RES / "nl" / "nl_middleware_report.json")
    v2 = load(RES / "nl" / "nl_middleware_report_v2.json")
    keys = ["T1_L8", "T1_L16", "T1_L32", "T2_L8", "T2_L16", "T2_L32", "T3_L8", "T3_L16", "T3_L32"]
    A = [v1[k]["A"] for k in keys]
    Cv1 = [v1[k]["C"] for k in keys]
    Cv2 = [v2[k]["C"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(keys))
    w = 0.26
    ax.bar(x - w, A, w, label="pure LLM (direct/CoT)", color="#d62728")
    ax.bar(x, Cv1, w, label="middleware v1 (parse+exec)", color="#8b949e")
    ax.bar(x + w, Cv2, w, label="middleware v2 (+coder/repair)", color="#2ea043")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ") for k in keys], fontsize=7, rotation=30)
    ax.set_ylabel("full-trajectory accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Fig4. NL middleware: LLM parses, executor computes (pure LLM = 0 everywhere)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_nl_middleware.png")
    plt.close(fig)


# ---------------- Fig5: 外置层 ----------------
def fig5():
    td = load(RES / "external" / "tape_deep_report.json")
    bits, sizes, accs = [], [], []
    for key, v in td["C1"].items():
        b = int(key.split("_")[0][1:])
        bits.append(b)
        sizes.append(v["file_bytes"])
        accs.append(v["acc"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(bits))
    colors = {32: "#8b949e", 8: "#1f77b4", 4: "#2ea043"}
    ax = axes[0]
    ax.bar(x, sizes, color=[colors[b] for b in bits])
    ax.set_xticks(x)
    ax.set_xticklabels([f"b{b}" for b in bits], fontsize=7)
    ax.set_ylabel("tape file size (bytes, log)")
    ax.set_yscale("log")
    ax.set_title("L=20000 checkpoint tape: size vs quantization")
    ax.grid(alpha=0.3, axis="y")
    for i, (sz, ac) in enumerate(zip(sizes, accs)):
        ax.text(i, sz * 1.15, "OK" if ac else "FAIL", ha="center", fontsize=7,
                color="#2ea043" if ac else "#d62728")
    ax = axes[1]
    ext = load(RES / "external" / "external_layer_report.json")
    labels = ["replay\n(1000 steps)", "memoization\n(100 queries)", "segmented\nL=20000"]
    vals = [ext["A"]["speedup"], ext["C"]["speedup"], ext["B"]["full_time_s"] / ext["B"]["seg_time_s"]]
    ax.bar(range(3), vals, color=["#9467bd", "#ff7f0e", "#1f77b4"])
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("speedup (x, log)")
    ax.set_title("External tape benefits")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.2, f"{v:.1f}x" if v < 1000 else f"{v/1000:.0f}kx", ha="center", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Fig5. External activation tape: 4-bit lossless, replayable, memoizable")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_external_tape.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    print("fig1 done")
    fig2()
    print("fig2 done")
    fig3()
    print("fig3 done")
    fig4()
    print("fig4 done")
    fig5()
    print("fig5 done")
