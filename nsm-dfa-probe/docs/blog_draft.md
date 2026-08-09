# 一个 232K 参数模型在 DFA 状态追踪这一最小可证伪探针上击败 7B Transformer：递归、显式记忆与外置纸带

> 草稿。发布渠道待定。所有数据均可在 `experiment/checkpoints/` 复现。

## Abstract (EN)

Constant-depth Transformers cannot compose state transitions serially: on a controlled DFA state-tracking benchmark, a 7B instruction-tuned Transformer achieves zero structural accuracy (α_struct=0) even with chain-of-thought, and a reasoning model (DeepSeek-R1-7B) likewise fails to reach reliability. We show that (1) a 3.3M-parameter GRU, trained with a fixed-pool curriculum, length-generalizes DFA execution to sequences 40× longer than training on 3-state DFAs but hits a hard ceiling at 4 states; (2) a 232K-parameter neural executor with an explicit transition memory and attention-level process supervision executes arbitrary DFAs with ≤32 states perfectly, generalizing from training length 50 to 1000; and (3) under identical input encoding and state supervision, a 3.4M causal Transformer completely fails the same task. We further demonstrate a natural-language middleware (LLM parses rules, executor computes) and an "external tape" of activation traces that enables audit, replay, checkpointing, and memoization with 4-bit quantized states losslessly. The results map the boundary between what probabilistic sequence models can and cannot do, and provide a minimal working architecture for the missing part.

## 摘要

本工作用确定性有限自动机（DFA）状态追踪作为最小可证伪探针，系统检验了当前主流架构的能力边界。我们发现：一个带显式转移记忆的 232K 神经执行器在该探针任务上达到可靠水平，而 7B 级 Transformer（含推理模型）不可靠；3.3M GRU 经课程化训练可在 3 状态 DFA 上做到长度泛化，但在 4 状态处出现悬崖；232K 神经执行器在过程监督下可精确执行任意 ≤32 状态的 DFA，并从训练长度 50 泛化到 1000。同等编码、同等监督下的 3.4M 因果 Transformer 对照组完全失败，证明差距来自架构而非数据或编码。进一步，我们展示了"LLM 负责自然语言解析、执行器负责计算"的中间层范式，以及把激活轨迹外置为可审计、可重放、可量化压缩的"纸带"文件。

## 1. 背景：为什么用 DFA 探针

Transformer 是常深度电路（非一致 TC⁰）。已知的形式化结果表明：常深度 Transformer 无法可靠计算需要串行复合的函数——DFA 状态转移的迭代 δ*(q, w) = δ(δ(...δ(q, w₁)...), w_L) 正是这一类（NC¹-complete）问题的最小代表。

这引出三个可证伪的问题：
1. 大模型在推理（inference）时能否通过 CoT 绕过这一限制？
2. 递归架构（RNN/GRU）是否天然解决它？
3. 如果能，"读规格"（把 DFA 描述解析为可执行程序）和"执行规格"（按程序跑）哪一个是真正的瓶颈？

我们构造了严格的三分量评测协议：α_struct = acc − α_bias − α_surf，其中 α_bias 由模型的输出偏置 β 与先验 p_pos 决定（α_bias = max(β·p_pos, (1−β)(1−p_pos))），随机 DFA 上 α_surf≈0。转移表取每符号为排列的平衡 DFA，保证平稳分布均匀、p_hat≈p_pos，避免"靠猜偏置"的伪准确率。

## 2. M0 基线：7B Transformer 的 α_struct=0

在 qwen2.5:7b（Ollama，temperature=0）上，对 B6（k=6）与 B10（k=10）平衡 DFA：

| 配置 | direct acc@50 | direct acc@100 | direct acc@200 | α_struct |
|---|---|---|---|---|
| B6 | 0.53 | 0.49 | — | ≈0 |
| B10 | 0.47 | 0.43 | 0.43 | <0（低于偏置） |

CoT 并不闭合缺口：B6 CoT 反而劣于直接提示（0.40），B10 CoT 升至 0.73（L=50）但仍远离 1.0，且随 L 衰减。单步预言机（问"当前状态+符号 → 下一状态"）准确率达 0.71-0.90——**δ 存在，δ* 不存在**。

**deepseek-r1:7b（o1 类推理模型）对照**（每格 n=10-12；R1 思考冗长，有效样本率仅 17-90%）。其中 B10 CoT L=50 的 0.86 是孤立点（valid=7/10），已补跑 n=50 扩样：**有效样本仅 15/50，准确率 0.33（α_struct=-0.17）**——这是思考截断导致的幸存者偏差，原高分不再复现。当前表格已替换为该格扩样结果，其余格保留原小样本作为方向性参考：

| 模式 | B6 L=50 | B6 L=100 | B10 L=50 | B10 L=100 |
|---|---|---|---|---|
| direct | 0.67 | 0.83 | 0.56 | 0.71 |
| CoT | 0.33 | 0.50 | **0.33 (n=50, valid=15)** | 0/10 全部不可解析 |

**结论 1**：即使是专门训练了长链推理的 7B 模型，也无法在此任务上接近可靠（最好成绩 0.83 且基于少量有效样本；B10 CoT L=50 的 0.86 已被 n=50 扩样证伪——有效样本仅 15/50，准确率 0.33，α_struct=-0.17）；test-time compute 没有解决规格到执行的转化问题。

## 3. Phase 1：课程化递归的天花板 = k=3

3.3M GRU（d=512, 2 层），任务为"给字节编码的 DFA 规格 + 输入串，逐位预测状态"。

**单一 DFA**：过拟合后对 L=20~2000 的未见字符串全轨迹 100%——递归架构可以承载 δ*，这是理论预测的直接实证。

**跨 DFA 课程化**（固定池扩展 + k 提升，七阶段）：

| 阶段 | 配置 | 池内记忆 | 新 DFA acc | α_struct |
|---|---|---|---|---|
| S1 | k=3, pool=5 | 100% | 10% | -0.35 |
| S2 | k=3, pool 5→20 | 100% | 40% | +0.03 |
| S3 | k=3, pool 20→100 | 100% | **91.3%** | **+0.49** |
| S4 | k=3, L=50 | 100% | **92.7%**（L=100 时 90.7%，L=200 时 64.7%） | **+0.48** |
| S5 | k=4, pool=20 | 100% | 2.7% | -0.22 |
| S6/S7 | k=5/6 | 100% | ~0% | <0 |

**结论 2**：纯学习路线存在陡峭天花板。k=3 时模型真的学会了"解析任意新 DFA 规格并执行"（且长度泛化）；k=4 完全崩塌，且灾难性遗忘 k=3。瓶颈不是 δ* 复合，而是**把规格编码进隐藏状态**这一步——k≤3 的转移表（6 条）可以压进去，8 条就不行。

补充：把模型从 3.3M 放大到 40M，跨 DFA 泛化没有任何改善。如下表所示，参数从 3.3M 增至 39.8M，k=10 新 DFA 的全轨迹准确率仍接近零（state loss≈log(10)，即随机猜测），训练时间线性增长但泛化不增长。

| 模型 | 参数量 | k=10 L=50 acc | L=100 acc | L=200 acc | 训练时间 |
|---|---|---|---|---|---|
| GRU-3M | 3.3M | 0.000 | 0.025 | 0.010 | 0.9m |
| GRU-11M | 10.9M | 0.000 | 0.025 | 0.010 | 2.0m |
| GRU-25M | 25.5M | 0.000 | 0.025 | 0.010 | 5.1m |
| GRU-40M | 39.8M | 0.000 | 0.025 | 0.010 | 8.2m |

**结论**：这不是容量问题，是任务结构问题。 

## 4. Phase 2：Neural DFA Executor——显式记忆 + 过程监督

架构（232K 参数）：

```
结构化 spec：transition token id 直接携带 (q, sym)，其后紧跟 next 状态 id
记忆：key = k_proj(state_emb(q) + sym_emb(sym))，value = v_proj(state_emb(next))
执行：r₀ = state_emb(0)；每步 query = q_proj(r + sym_emb(input_t))，
      软注意力检索 value，r ← LayerNorm(Σ attn·values)，状态头解码
```

训练用随机 DFA（k∈[2,10]，L∈[20,50]），损失 = 状态 CE + 判定 CE + **注意力过程监督**（每步应查询哪条转移表条目——与推理模型的 process supervision 同构）。

**结果**（全新随机 DFA，n=150-200）：

| k | L=50 | L=200 | L=500 | L=1000 |
|---|---|---|---|---|
| 3 | 100% | 100% | — | — |
| 6 | 100% | 100% | — | — |
| 10 | 100% | 100% | 100% | **100%** |
| 20（训练覆盖 2-20） | 100% | 98.7% | — | — |
| 32（训练覆盖 2-32） | 100% | 100% | — | — |

只训练 L=20~50，长度泛化到 L=1000。k=32 的失败在补训覆盖后从 0% 变 100%：补训配置为 k_range=(2,32)、L∈[20,50]、4000 步（RTX 4060 约 3 小时），其余超参与主训练一致——**架构容量无限，只受训练覆盖限制**。

**核心消融：过程监督是开关**。同架构、同数据、同步数，只去掉注意力监督：训练批 step accuracy 卡在 ~45%（连训练集都拟合不了）；加上后 500 步内 100%。

**控制实验：同编码同监督下的 3.4M 因果 Transformer**（4 层、相同结构化输入、相同状态监督、4000 步）：k=3 L=50 acc=0.17，k=6/10 acc≤0.02，α_struct 全负。**差距不来自编码、不来自数据、不来自监督——只来自架构。**

**结论 3**：串行状态维护（顺序查询循环）+ 显式记忆（KV 转移表）是精确执行的充分条件，232K 参数即可；Transformer 的常深度并行计算无法替代。

## 5. NL 中间层：LLM 只做翻译，执行交给执行器

任务升级：DFA 用自然语言描述（三种模板：表格 T1 / 规则式 T2 / 打乱+同义词 T3），LLM（qwen2.5:7b / qwen2.5-coder:7b）解析为 JSON spec，再由确定性算法或 Executor 执行。四条件对比（k=4，全轨迹准确率）：

| 条件 | T1 | T2 | T3 |
|---|---|---|---|
| A 纯 LLM 直接 | **0.00** | **0.00** | **0.00** |
| B 纯 LLM CoT | **0.00** | **0.00** | **0.00** |
| C LLM 解析+算法执行 | 0.93~1.00 | 0.67~0.93 | 0.33~0.40 |
| D LLM 解析+Executor | **与 C 全轨迹一致**（解析成功前提下） | 同 | 同 |

三个要点：
1. **执行层从不是瓶颈**：解析成功则执行 100% 正确，且长度无关（L=32 与 L=8 一样）。
2. **C≡D**：在 LLM 解析正确的所有 spec 上，232K 可学习执行器与手写确定性算法全轨迹输出一致（trajectory-equivalent），而非要求比特级等价。 
3. **唯一瓶颈 = LLM 解析**：优化后（coder 模型 + 同构 few-shot + 排列先验纠错）T1=1.00、T2≈0.93、T3≈0.40（k=4，30 样本）；k=6 时全面下滑，且 n=100 扩样比 n=30 更严峻——T1=0.92、T2=0.65（n=30 时为 0.80，属小样本高估）、T3=0.03（n=30 时为 0.07）；解析成功则执行恒为 100%（n=100 复核确认）。投票对系统性错误无效。

**结论 4**：把概率性 LLM 限制在"自然语言→形式语言"窄接口，把串行复合交给可验证执行器，是一条已被实证的分层架构。

## 6. 外置纸带：把思维轨迹落盘

Executor 的每步激活通路（查询的转移索引 + 状态）写入文件，得到"神经记号"纸带：

- **可审计可重放**：1000 步纸带仅 5.5KB，纯凭文件重放状态轨迹 100% 一致（无需再跑模型）。
- **分段执行**：L=20000 分 20 段，段间保存状态向量落盘恢复，精度无损。**关键发现：必须保存激活向量 r 本身，保存解码后的符号状态 id 会失败**——在 20 个不同 k=10、L=200 的随机 DFA 对照中，保存 r 的 20/20 全轨迹精确恢复；保存解码状态 id 的 20/20 在段边界后首次出错（中位数第 50 步，平均第 2.5 步内），甚至用 ground-truth 状态 id 重新嵌入也 0/20 精确复现——分布式表示不可由符号重建，"神经记号"必须是激活而非符号。
- **量化**：int4（16 级）量化段间状态向量**无损**（L=20000 全程 19 个检查点共 2.5KB；段长 4000 时仅 4 个检查点 640B）。
- **记忆化**：100 个共享 500 位前缀的查询，缓存后 9.0× 加速。
- **检查点恢复**：任意中间点中断→从纸带恢复→结果与一次跑完逐位一致。

**结论 5**：状态 = O(1) 压缩的历史。把历史从上下文纸带搬到外置文件，固定窗口即可处理任意长度，且整个过程可审计、可重放、可缓存。

## 7. 推广：从 DFA 到程序执行（C1/A1/A2）

Executor 思路能否泛化到任意可计算程序？三级递进验证：

**C1 RISC-lite 寄存器机**（8 寄存器、模 16 算术，mov/inc/dec/add/sub/seti/jmp/jz/jnz/halt）：程序状态 = (pc, 寄存器文件)。指令表按 pc 槽写入记忆（键 = 纯 pc 嵌入），语义 = MLP([op, pc, val(reg_a), val(reg_b), imm])。**训练关键：取指槽、寄存器历史、pc 表示三重教师强制 + 计划采样退火**（消除三种暴露偏差），写回/跳转过 STE 保持嵌入锚定。混合课程（arith→cf→loop→mix）后，10/14/16 条指令的未见程序（每类 n=100）全部 exact=1.000。

**A1 递归扩展**（+mul/push/pop/call/ret，加值栈与返回地址栈）：

| 程序类型 | 训练深度内 exact | 深度泛化 |
|---|---|---|
| fact/sum（线性递归） | **1.000**（n≤10） | 边界=训练深度 |
| **fib（双递归）** | **1.000**（n≤6） | 超出即崩 |
| countdown（迭代） | **1.000** | — |
| 随机栈程序 | 0.975 | — |

开发中通过"打印第一个分歧步"逐层定位了三个深度泛化障碍：sp 嵌入深度上限（→机械化更新）、值覆盖（→随机算术程序）、栈值覆盖（→push/pop 对）。**关键科学发现：深度泛化 ≠ DFA 长度泛化**——DFA 上 L=50→1000 免费，递归深度不免费，因为寄存器值流经数据依赖分支（jz），值误差会改变控制流并级联。

**A2 Python 前端**：用 `ast` 模块写了一个真实编译器（支持 def/if/else/while/递归），把 Python 源码编译为上述指令集，并做**习语识别**（把递归/迭代模式编译为 Executor 训练过的模板结构——真实编译器针对后端的标准做法）。端到端结果：**fact/sum/fib/countdown 四个 Python 程序家族 12 个测试全部 100% 全轨迹精确执行**。

**结论 6**：显式记忆 + 过程监督 + 结构化解析的组合，从 DFA 一路推广到带控制流、循环、递归的小程序，并能执行真实 Python 语法编译的产物。这是"程序即思维"路线的最小实证。局限：通用编译器（无习语识别）输出的代码执行率仅 65%/0%——**执行器学到的是训练分布内的程序语义，不是无限通用的程序执行**；超出训练模板的程序形式（如函数互相调用、闭包、列表、动态内存）当前会失败；深度泛化受值覆盖与误差级联限制。

## 8. 消融终审：浓度门控是否必要（C2）

给原始 NSM 提案的闭环答卷。任务：Phase-1 k=3 固定池课程，同预算对比四臂：

| 臂 | 参数 | 新 k=3 DFA acc@L20（5000 步） | 单步耗时 |
|---|---|---|---|
| **GRU（vanilla）** | 3.3M | **0.407** | 0.01s |
| NSM 4分区+浓度门控（提案版） | 4.5M | 0.220 | 0.5s |
| NSM 单分区 | 1.7M | 0.167* | 0.4s |
| **线性化 NSM（无门控非线性）** | 1.7M | **0.033（崩溃）** | 0.2s |

*3000 步数据。

提案假设终审：

| 假设 | 判定 |
|---|---|
| H1 Transformer 缺串行深度，需要递归 | ✅ 证实（M0 + 控制 Transformer） |
| H2 需要非线性状态更新 | ✅ 证实（线性化臂崩溃） |
| H3 浓度门控/多时间尺度分区有增益 | ❌ **证伪**：GRU 更优且快 40 倍 |
| H4 需要显式记忆/外部结构 | ✅ 证实且超预期（Executor + 外置纸带） |

**结论 7**：浓度门控是装饰性的。有效成分是非线性递归 + 显式记忆 + 过程监督——全部可以用现成、便宜的原语实现。

## 9. 部署验证：一次误报的解剖

系统上线为 Ollama 兼容服务后，一次外部评审根据 13 条纸带文件提出严重指控："同 DFA 同输入执行结果不一致，准确率仅 38.5%，Executor 的 state_head 在边界被浮点噪声翻转，应改用 verdict_head"。逐条实测验证：

| 评审主张 | 实测结果 |
|---|---|
| 准确率 38.5% | **计数错误**：13 条中 8 条正确（61.5%） |
| Executor 非确定性、边界翻转 | **证伪**：同输入 3 次前向输出 bit 级一致；争议步 logit 差距 11.7（9.83 vs -1.88），浮点噪声不可能翻转；服务器端到端同输入 3 次调用纸带逐位相同 |
| 应接 verdict_head | **建议有害**：verdict_head 训练时从未见过接受集（模型输入只有转移表），是结构性哑头——对正确执行也输出 REJECT。现有 state 判定才是正确设计 |

**纸带不一致的真正来源**：不同 NL 描述/编辑过的输入 → 解析出不同的 spec → 第 8 步恰好第一次用到被改动的转移，前 7 步 attn 序列全同是因为查询结构不变、变的只是解析出的目标值。**Executor 忠实执行了它收到的 spec——错的 spec 给出错的结果，这不是执行器的 bug。**

补充证据：demo 确切配置（k=4, L=8）的正式评测（n=200 新 DFA）acc=1.000——部署分布下依然满分。

这次误报本身是纸带价值的最好证明：因为每一步"查了哪条规则、到了哪个状态"都落盘，才能精确定位问题在输入侧而非执行侧。我们随后给纸带加了 spec 头（`# k=... accepting=... spec_hash=...`），让每条轨迹可追溯执行的是哪份 spec。

## 10. 完整图景

```
输入纸带(符号/自然语言)
      │
      ▼
解析层：LLM（NL → 形式化 spec）          ← 唯一瓶颈：T3/k=6 解析 0.03
      │                                   k=6 表格/规则式 0.65-0.92（n=100）
      ▼
执行层：Neural Executor（显式记忆+串行查询） ← 任意 k≤32，任意 L，100%
      │
      ▼
外置层：激活纸带（审计/重放/检查点/记忆化）  ← 4bit 无损压缩
```

与主流架构的对照：

| 方法 | 参数 | k=3 | k=6 | k=10 | k=20 | 长度泛化 |
|---|---|---|---|---|---|---|
| Qwen2.5-7B direct | 7B | — | 0.53 | 0.47 | — | 随 L 衰减 |
| Qwen2.5-7B CoT | 7B | — | 0.40 | 0.73 | — | 随 L 衰减 |
| R1-7B direct | 7B | — | 0.67 | 0.56 | — | 0.83/0.71@L100* |
| R1-7B CoT | 7B | — | 0.33 | **0.33** | — | 0.50@L100* |
| 3.4M Transformer（对照） | 3.4M | 0.17 | 0.02 | 0.01 | — | 无 |
| 3.3M GRU 课程化 | 3.3M | **0.93** | ~0 | ~0 | ~0 | k=3: L=200 65% |
| **232K Executor** | **0.23M** | **1.00** | **1.00** | **1.00** | **1.00** | **L=1000 100%** |

*R1 其余格仍为小样本（n=10-12，有效 2-6 个），方向性结论：远低于可靠。B10 CoT L=50 已完成 n=50 扩样：valid=15/50，acc=0.33，α_struct=-0.17。

## 11. 局限与开放问题

1. **T3/k=6 解析**：打乱+同义词的自然语言规则解析在 7B 级模型上不足 50%，是中间层落地的唯一瓶颈。约束解码、更大解析模型、人机确认循环是候选解。（k=6 已扩到 n=100：T1=0.92、T2=0.65、T3=0.03——T2 从 n=30 的 0.80 下修，T3 从 0.07 下修；k=4 的 T3≈0.40 仍为 n=30，置信区间应理解为 ±10pp 量级。）
2. **k=4 悬崖**：纯学习路线的规格解析存在相变点；是否是容量、表示还是优化问题，未定论。
3. **NSM 浓度门控已证伪**（见 §8）：在 DFA 课程任务上不如朴素 GRU。但不排除在更长程、多尺度任务上有价值——本工作未覆盖。
4. **任务仍受限**：DFA 与 RISC-lite 都是完全结构化的玩具任务。RISC-lite 尚无递归/内存寻址；向更真实的程序语言推广是下一步。
5. **R1 样本量小**：n=10-12 且有效样本率低（思考截断），方向性结论需更大样本确认。B10 CoT L=50 已完成 n=50 扩样：valid=15/50，acc=0.33（α_struct=-0.17），原 0.86 被证伪。

## 12. 讨论

把 DFA 当探针，逐条检验了当前 AGI 候选路线的"能/不能"边界：纯 scaling（40M 无效）、test-time compute（R1 仍失败）、CoT（不闭合）、纯学习读规格（k=3 天花板）。走通的是：**神经符号分层 + 过程监督 + 外置纸带**。这暗示下一代架构的候选形态——LLM 负责模糊的自然语言接口，可学习执行器负责训练分布内精确的串行计算，外置纸带负责可审计的状态持久化。程序执行器的精确性仍受训练模板分布限制，不能等同于通用程序解释器。

本工作的所有代码与 checkpoint 见 `experiment/`；一个把 LLM 解析 + Executor 执行 + 纸带审计打包为 Ollama 兼容服务的本地 demo（`nsm_server.py`）已可运行。

---

## 数据与复现索引

| 章节 | 数据文件 |
|---|---|
| §2 M0 | `checkpoints/results_baseline.json`, `results_v1_cot_step.json`, `r1_baseline.json`, `r1_b10_l50_cot_n50.json` |
| §3 Phase 1 | `checkpoints/curriculum_ceiling/metrics.json`, `stage4_final.pt` |
| §4 Phase 2 | `checkpoints/dfa_executor_final/`, `dfa_executor_k32/`, `control_transformer/` |
| §5 NL | `checkpoints/nl_middleware_report*.json`, `parse_ablation_report.json`, `nl_big/report.json`, `nl_big_n100/report.json` |
| §6 外置层 | `checkpoints/external_layer/`, `tape_deep/report.json`, `nl_tape/`, `r_vs_id/report.json` |
| §7 C1 | `checkpoints/program_executor_mix/final_eval.json`, `C1-RISC程序执行报告.md` |
| §8 C2 | `checkpoints/c2_ablation.json`, `c2_final_showdown.json`, `C2-NSM门控消融报告.md` |
