# C1 报告：RISC 程序执行探针

## 结论

**Executor 架构从 DFA 成功推广到带控制流和循环的 RISC 程序**：1.14M 参数的 ProgramExecutor 对未见过的随机程序（算术/前跳控制流/计数循环，n_instr=10~16）实现 **100% 全轨迹精确执行**（pc + 8 寄存器，每步都对）。

## 任务与架构

**RISC-lite**：8 寄存器、模 16 算术、输入串直接初始化寄存器。
指令集：mov / inc / dec / add / sub / seti / jmp / jz / jnz / halt。

**架构**（Neural DFA Executor 的泛化）：
- 程序 spec 结构化编码：每条指令 = opcode token + 操作数 token，从 token id 确定性解析 (op, reg_a, reg_b/imm)
- **取指**：pc 槽注意力（键 = 纯 pc 嵌入），CE 过程监督对齐
- **寄存器文件**：(n_reg, d) 显式槽位，值用 val_emb 锚定
- **语义计算**：MLP([op_emb, pc_emb, val(reg_a), val(reg_b), val(imm)]) → 写回值 + 下一 pc
  （RISC 指令语义上下文无关，无需指令内容表示参与）
- **写回/跳转**：STE（straight-through estimator）——前向取纯嵌入锚点，梯度走软路

## 训练关键：三重教师强制 + 计划采样

v1→v4 迭代中发现的三个暴露偏差，全部用教师强制 + 退火解决：
1. **取指槽**：教师强制 → 注意力 CE 收敛到 0
2. **寄存器历史**：读操作数用真值历史（否则一次写错全盘皆错）
3. **pc 表示**：p 用真值 pc 嵌入（否则 pc 错后监督矛盾）

计划采样：前 50% 步全教师强制，50-80% 线性退火到 0，最后 20% 全自主滚动练习。

## 训练课程

| 阶段 | 内容 | 结果 |
|---|---|---|
| arith | 纯算术程序（n=6-12） | exact=1.000 |
| cf | +前跳条件跳 | exact=1.000 |
| loop | +计数循环 | exact=1.000 |
| **mix** | **三者混合 + n=6-16** | **全部 1.000** |

分阶段顺序训练出现**灾难性遗忘**（loop 训完后 cf 从 1.00→0.48）和长程序泛化差（n=14 掉到 0）；
混合训练 + 扩大 n_instr 覆盖后两者同时解决。

## 最终评测（未见过程序，n=100/配置）

| 程序类型 | n=10 | n=14 | n=16 |
|---|---|---|---|
| arith | 1.000 | 1.000 | 1.000 |
| cf | 1.000 | 1.000 | 1.000 |
| loop | 1.000 | 1.000 | 1.000 |

## 工程教训

1. **`run_program` max_steps 不退出 bug**（T=300 vs 实际 12）导致 25 倍减速——轨迹生成要在停机后及时截断。
2. **逐元素 GPU 张量构造**（2304 次 H2D/批）是隐形性能杀手——一律 CPU 构建完一次性传输。
3. **DFA Executor 的成功要素可迁移**：内容锚定（嵌入查表）+ 过程监督 + 结构化解析，这套组合在程序执行上同样有效。

## 产物

- `nsm/program_executor.py`：模型 + 数据 + 训练 + 评测
- `run_program_stages.py`（分阶段）、`run_program_mix.py`（混合）
- checkpoint：`checkpoints/program_executor_{arith,cf,loop,mix}/final.pt`（最佳：mix）
- 数据：`checkpoints/program_executor_mix/final_eval.json`
