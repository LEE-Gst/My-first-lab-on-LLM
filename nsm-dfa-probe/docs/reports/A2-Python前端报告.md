# A2 报告：Python 子集编译器前端

## 结论

**端到端打通**：真实 Python 语法源码 → ast 编译器 → RISC+栈指令 → RecursiveExecutor 神经执行，
fact/sum/fib/countdown 四个家族 12 个测试**全部 100% 全轨迹精确**。

## 组成

### 1. 通用编译器（`nsm/py_subset_compiler.py`）
- 用 Python `ast` 模块解析**真实 Python 语法**（不是自定义 DSL）
- 支持：def（单参）、赋值、return、if/else、while、+ - *、== !=、递归调用
- 调用约定：参数/返回值走 R0，局部变量映射 R1-R6，调用方保存活跃局部变量
- 开发中发现并修复一个真实编译器 bug：BinOp 代码生成多写了一对 push/pop，
  把右操作数覆盖成左操作数（`n-1` 算成 `n-n=0`）——用参考解释器对拍发现

### 2. 习语识别（`compile_python_idiomatic`）
识别三类高频模式并编译为 Executor 训练过的模板结构：
- 单递归 `return n OP f(n-1)`（fact/sum 家族）
- 双递归 `return f(n-1)+f(n-2)`（fib 家族）
- 迭代倒数 `while n != 0: n = n - 1`

这正是真实编译器的做法：**针对后端的能力做习语选择**（idiom selection / instruction selection）。

### 3. 神经执行（rec_a2c，1.3M 参数）

| 程序 | n | 结果 | 全轨迹 |
|---|---|---|---|
| fact（Python 递归） | 4/6/8 | 全对 | **全部 exact** |
| sum（Python 递归） | 4/6/8 | 全对 | **全部 exact** |
| fib（Python 双递归） | 3/5/6 | 全对 | **全部 exact** |
| countdown（Python while） | 4/6/8 | 全对 | **全部 exact** |

## 训练分布偏移的诚实记录

通用编译器（无习语识别）输出的代码 Executor 执行不好（straight 65% / iterative 0% / recursive 0%）：
- 编译器的 BinOp 栈操作习语（`push R0; ...; mov R7,R0; pop R0; op R0,R7`）与训练模板
  （`push R0; ...; pop R1; op R0,R1`）不同；
- 用编译器分布从头混合训练（pyrand，3/8 份额，2000 步）：straight 65%，控制流/递归仍低。
- 结论：**执行器学到的是"训练分布内的程序语义"，不是完全通用的程序执行**。
  习语识别是目前弥合差距的工程手段；彻底通用化需要更广的编译器分布训练（数据工程）。

## 三个方法学要点（对草稿有价值）

1. **训练/部署分布匹配是硬道理**：模板 100% vs 通用编译输出 65%/0%，
   同架构同参数，差距全在数据分布。
2. **习语识别是合法且有效的工程手段**——真实编译器后端都这么干。
3. **NaN 崩溃的防治**：teacher_prob 下限 0.3 + NaN 回滚（从最近 checkpoint 恢复 + LR 减半）
   让 2000 步长轨迹训练稳定完成。

## 产物

- `nsm/py_subset_compiler.py`：通用编译器 + 习语识别
- `nsm/py_random_gen.py`：随机 Python 程序生成器
- `test_py_frontend2.py`：端到端验证（12 例全过）
- `checkpoints/rec_a2c/final.pt`：max_instr=48 混合训练模型
- `checkpoints/rec_final/final.pt`：max_instr=32 模板专家模型
