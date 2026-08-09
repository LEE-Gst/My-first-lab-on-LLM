```
首先声明，基本代码都是AI写的，不保真嗷
```

# Project Introduction — NSM / DFA Probe Experiments

> 用 DFA 作探针，实证"纯概率模型不能做算法推理"，并给出可行补全路径：
> **非线性递归 + 显式记忆 + 过程监督 + 外置纸带**。
> 232K 参数执行器在结构追踪上击败 7B Transformer，并推广到递归程序与真实 Python 源码执行。

## TL;DR

| 主张 | 证据 |
|---|---|
| 7B Transformer（含 R1 推理模型）在 DFA 追踪上 α_struct ≈ 0 | `docs/reports/M0_结果摘要.md`、`results/m0/` |
| 3.3M GRU 跨 DFA 课程化在 k=4 出现悬崖，**40M 同样崩塌** | `results/gru_scale/` + `Phase1-课程化天花板报告.md` |
| **232K 神经执行器**：k≤32、L≤1000 全部 100% 全轨迹精确 | `checkpoints/dfa_executor_k32_final.pt` (909KB) |
| 同编码同监督下 3.4M 因果 Transformer 完全失败 | `checkpoints/control_transformer_final.pt` + `Phase2-Executor报告.md` |
| LLM 解析 + Executor 执行：C≡D（与算法逐位等价） | `docs/reports/NL中间层实验报告.md` |
| 外置纸带：4bit 量化无损、纯文件重放、断点恢复、记忆化 9× | `docs/reports/外置层实验报告.md` + `tapes/` |
| 浓度门控 H3 证伪：朴素 GRU 已足够 | `checkpoints/c2_*_final.pt` + `C2-NSM门控消融报告.md` |
| 部署误报解剖：13 条纸带"38.5% 不一致"系评审误数（实为 61.5%）| `tapes/` + `评审验证报告.md` |

## 仓库结构

```
.
├── nsm/                       # 9 核心模块（数据/模型/执行器/评测）
│   ├── data.py                # 平衡 DFA 生成、字节/结构化编码、课程化采样
│   ├── model.py               # GRULM / NSMByteLM / TraceTransformer
│   ├── dfa_executor.py        # 232K 神经 DFA 执行器（核心）
│   ├── program_executor.py    # RISC-lite 寄存器机执行器
│   ├── recursive_executor.py  # 递归程序执行器（+值栈/返回栈）
│   ├── py_subset_compiler.py  # Python 子集 ast 编译器 + 习语识别
│   ├── py_random_gen.py       # 随机 Python 程序生成器
│   ├── eval.py                # M0 三分量评测协议（α_struct）
│   └── __init__.py
├── scripts/                   # 8 训练/评测/服务脚本
│   ├── train_executor.py
│   ├── train_curriculum.py
│   ├── train_programs.py
│   ├── train_recursive.py
│   ├── train_control_transformer.py
│   ├── nl_middleware.py
│   ├── nsm_server.py          # Ollama 兼容服务（解析+执行+纸带）
│   └── make_figures.py
├── docs/                      # 完整文档
│   ├── blog_draft.md          # 中文主稿（12 节）
│   ├── blog_draft_en.md       # 英文完整版
│   ├── final_summary.md       # 终审总结
│   └── reports/               # 13 份分阶段报告
├── figures/                   # 5 张主图
├── results/                   # JSON 结果（按阶段分目录）
├── tapes/                     # 18 条激活纸带（§9 误报解剖原始素材）
└── checkpoints/               # 12 个核心模型（LFS 追踪，~245MB）
```

## 关键结果速览

### 架构对比

| 方法 | 参数量 | k=3 | k=6 | k=10 | k=20 | 长度泛化 |
|---|---|---|---|---|---|---|
| Qwen2.5-7B direct | 7B | — | 0.53 | 0.47 | — | 随 L 衰减 |
| Qwen2.5-7B CoT | 7B | — | 0.40 | 0.73 | — | 随 L 衰减 |
| DeepSeek-R1-7B direct | 7B | — | 0.67 | 0.56 | — | 0.83/0.71@L100* |
| 3.4M Transformer（控制组）| 3.4M | 0.17 | 0.02 | 0.01 | — | 无 |
| 3.3M GRU 课程化 | 3.3M | **0.93** | ~0 | ~0 | ~0 | k=3: 65%@L200 |
| **40M GRU（§3 容量对照）** | 40M | ~0 | ~0 | ~0 | ~0 | 无 |
| **232K Neural Executor** | **0.23M** | **1.00** | **1.00** | **1.00** | **1.00** | **100%@L1000** |

*R1 数据为小样本（n=10-12），方向性结论。

### 核心消融

| 假设 | 判定 |
|---|---|
| H1 Transformer 缺串行深度，需要递归 | ✅ 证实（M0 + 控制 Transformer） |
| H2 需要非线性状态更新 | ✅ 证实（线性化臂崩溃） |
| H3 浓度门控/多时间尺度分区有增益 | ❌ **证伪**：GRU 更优且快 40 倍 |
| H4 需要显式记忆/外部结构 | ✅ 证实且超预期（Executor + 外置纸带） |

## 快速复现

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练 DFA Executor（约 25 分钟，RTX 4060）
python scripts/train_executor.py

# 3. 启动 Ollama 兼容服务（解析+执行+纸带）
python scripts/nsm_server.py

# 4. 重生成 5 张主图
python scripts/make_figures.py
```

> **重要**：仓内 `checkpoints/` 包含 12 个核心 checkpoint（LFS 追踪），足够复现 §3-§9 所有数字结论。
> 30M NSM 提案版原始模型（456MB）**未包含**——它已在 §8 中被证伪，需要复现者直接重训。

## 数据 / Checkpoint / 大文件分布

| 类别 | 在仓内 | 在外（HF Hub） |
|---|---|---|
| 代码 + 文档 + 图 | ✅ | — |
| 12 个核心 .pt（~245MB） | ✅ via LFS | — |
| 30M NSM 提案版（nsm30m_poc/*，456MB）| ❌ | 复现训练脚本（25 分钟）|
| GRU scale 11M/25M 中间档（138MB）| ❌ | scale 对比只需 3M + 40M 两端 |
| Smoke / 训练中 step*.pt | ❌ | 训练过程存档，无科学价值 |

```
  还没干，大文件没传，还没有看包脸怎么个事
```

## 完整文档阅读顺序

1. `docs/blog_draft.md`（中文 12 节，最快建立全景）
2. `docs/final_summary.md`（一页终审总结 + 7 条核心结论）
3. `docs/reports/` 13 份分阶段报告（每个实验模块一份）
4. `docs/reports/评审验证报告.md`（§9 误报解剖，独立可读）

## 引用

```bibtex
@misc{project-introduction-nsm-dfa,
  title  = {NSM / DFA Probe: A 232K Neural Executor Beats 7B Transformers at Structured State Tracking},
  author = {LEE-Gst},
  year   = {2026},
  url    = {https://github.com/LEE-Gst/project-introduction}
}
```

## 许可

MIT License — 详见 `LICENSE`。
