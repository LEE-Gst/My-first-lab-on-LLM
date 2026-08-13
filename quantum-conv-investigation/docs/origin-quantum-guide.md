# 本源量子平台测试指南

> 在**本源量子**(Origin Quantum)平台上复现本项目的 n=4 量子卷积电路(实验 05 的 LCU 块编码线路),并与 Qiskit/Aer 的结果直接对照。

---

## 零、本地调试状态交接(2026-08-13,pyqpanda 3.8.5 实测)

**已验证 ✅**:
- 便携 Python 3.11(临时目录)+ pyqpanda 3.8.5 可用(USTC 镜像安装);
- 门全部可用:RY / H / X / Measure / Toffoli / CU;
- **CU 约定已实测**:`CU(a,b,g,d, qA, qB)` = 控制为 qA、U4 作用于 qB,且
  `U4(a,b,g,d) = e^{ia}·RZ(b)·RY(g)·RZ(d)`,`RZ(φ)=diag(e^{-iφ/2}, e^{+iφ/2})`;
- **状态向量索引约定**:qstat 索引 = q0 + 2·q1(q[0] 为低位),与测量键序一致;
- B 块参数已按上述约定重算并手工验证矩阵(Z、X、d=±i/2 四个块全部核对无误);
- Toffoli 双控分解 + X 翻转分支逻辑经状态向量确认工作正常;
- 链式 `<<` **不是**问题(顺序诊断 V1=V2=V5 完全相同)。

**已修复 ✅(最新)**:
- **态制备角度已修正**:原角度把 q0/q1 角色搞反(实测状态为 (3,4,1,2) 而非
  (3,1,4,2))。新角度(已写入脚本):`t0 = 2·arccos(√(5/6)) ≈ 0.8411`、
  `t00 = 2·arccos(3/5) ≈ 1.8546`、`t01 = 2·arccos(1/√5) ≈ 2.2143`,
  逐项验算后状态精确为 (3,1,4,2)/√30。

**未解决 ❌(接手后从这里继续)**:
- **QFT2 的构造**:实测平台的 `H(q0)+CU(π/2,0,0,0,q0,q1)+H(q1)` 矩阵为
  `Q = (1/2)·i^{k0}·(−1)^{k·j}`(即 S·H⊗H,相位落在 q0 输出行上),**不是 DFT**,
  因此计算基下的对角阶段无法对角化 A_c(数值解 D̃ = Q·(A/4)·Q† 非对角,残差 0.5);
- **修复方向**:用已验证原语构造真正的 DFT 或换路:
  (a) CP(θ) 改用显式分解 `RZ(q0,−θ/2) + RZ(q1,−θ/2) + CNOT(q0,q1) + RZ(q1,θ/2) + CNOT(q0,q1)`
      (全局相位除外),再搭 H+CP+H 型 QFT2;
  (b) 或直接用数值方法:先测出自己构造的 QFT2 矩阵 Q',验证 D̃ = Q'·(A/4)·Q'† 为对角,
      再按对角元素布置 B 块(与 qiskit 调试同法);
  (c) 完成后本地验证应命中 P(anc=0) = 0.8417,再进入云端/噪声/芯片阶段。

**机时预算(60 秒)**:
- 本地验证:0 机时(免费);
- 云端全振幅 + 噪声扫描(4 比特线路):<1 秒;
- 真实芯片:建议 `ORIGIN_CHIP_SHOTS=2000`(默认已设),单任务量级数秒;
- **60 秒足够完成全部四步**,大头只在真实芯片提交,一次即可。

---

## 一、为什么本源量子适合这个实验

| 需求 | 本源量子平台 |
|------|-------------|
| 3-4 比特小线路 | 悟源 5 号(6 比特)、悟空(72 比特)均可跑 |
| 免费验证 | pyqpanda **本地全振幅模拟器**(无需账号) |
| 噪声对照 | 云端噪声虚拟机 `noise_measure` + `set_noise_model`,与实验 05 的 Aer 退极化模型**同口径** |
| 真实芯片 | `real_chip_measure` / `get_state_fidelity` 直接给测量分布和态保真度 |

---

## 二、准备(按需三选一)

1. **本地验证(免费,无账号)**:只需 `pip install pyqpanda`;
2. **云端模拟(需账号)**:注册 [本源量子云](https://qcloud.originqc.com.cn/),**工作台 → 个人账号中心**获取 `api_token`;
3. **真实芯片(需开通算力)**:同上,并确保已开通计算服务(未开通会报 `un-activate products or lack of computing power`)。

```bash
pip install pyqpanda
```

## 三、电路说明(4 比特)

```
q0, q1 : 系统比特 (幅值编码 x = (3,1,4,2)/√30)
q2     : 工作比特 (Toffoli 分解用, 保持 |0⟩)
q3     : 辅助比特 (块编码 ancilla, 后选择 |0⟩)

态制备 → QFT2 → 4 个双控 B 块 (LCU) → QFT2† → 测量
```

双控 B 块的标准分解(仅用原生门,规避多控门 API 差异):

```
[q0,q1 按分支取反] → Toffoli(q0,q1,q2) → CU(α,β,γ,δ)(q2→q3) → Toffoli(q0,q1,q2) → [还原]
```

B 块参数(ZYZ 分解 $B = e^{i\delta}RZ(\alpha)RY(\gamma)RZ(\beta)$,分支相位精确保留):

| 分支 | B | (α, β, γ, δ) |
|------|---|--------------|
| 00 | Z | (π, 0, 0, π/2) |
| 01 | [[i/2, s],[s, i/2]] | (−π/2, π/2, 2π/3, π/2) |
| 10 | X | (0, π, π, π/2) |
| 11 | [[−i/2, s],[s, −i/2]] | (π/2, 3π/2, 2π/3, π/2) |

## 四、四步测试

### 第 1 步:本地验证(自动判定约定)

```bash
python experiments/06_origin_quantum.py
```

脚本自动尝试两种 QFT 相位约定(θ=±π/2),打印判定结果。**判定标准**:

- `P(anc=0) ≈ 0.8417`(= 404/480,理论精确值);
- 后选择系统分布集合 = {0.2995, 0.2005, 0.2005, 0.2995}(标签顺序取决于平台比特序,数值集合不变)。

### 第 2 步:云端全振幅

```bash
# 先替换脚本内 TOKEN = "YOUR_API_TOKEN"
python experiments/06_origin_quantum.py cloud
```

### 第 3 步:云端噪声模拟(对照实验 05)

```bash
python experiments/06_origin_quantum.py noise
```

| 退极化率 p | 期望 P(anc=0)(Aer 实测) | 期望后选择保真度 |
|-----------|------------------------|-----------------|
| 10⁻⁴ | ≈0.8385 | ≈1.000 |
| 10⁻³ | ≈0.8222 | ≈0.9999 |
| 10⁻² | ≈0.6895 | ≈0.9983 |

> pyqpanda 噪声模型枚举:`NoiseModel.DEPOLARIZING_KRAUS_OPERATOR` / `BIT_PHASE_FLIP_OPRATOR` / `DECOHERENCE_KRAUS_OPERATOR` 等。

### 第 4 步:真实芯片

```bash
python experiments/06_origin_quantum.py chip
```

- 默认提交**悟空 72 比特**(`real_chip_type.origin_72`),也可用悟源 5 号(`chip_id=2`,默认);
- 脚本内置 `is_mapping=True, is_optimization=True`(默认开启)处理拓扑映射与门融合;
- 可选:调用 `get_state_fidelity(prog, shot, chip_id)` 直接获取状态保真度,与本实验理论态比较。

## 五、常见错误对照(官方文档)

| 报错 | 含义 |
|------|------|
| `server connection failed` | 服务器连接失败 |
| `api key error` | api_token 异常,去官网确认 |
| `un-activate products or lack of computing power` | 未开通产品或算力不足 |
| `build system error` | 编译系统出错 |
| `exceeding maximum timing sequence` | 量子程序时序过长(Toffoli 分解后深度过大时出现,可尝试 3 比特版或减少分解) |
| `unknown task status` | 其他任务状态异常 |

## 六、结果解读与下一步

1. **本地/云端全振幅**:若偏差 >0.01,先怀疑 pyqpanda 版本 API 差异(脚本已内置两种约定自检),再核对 `CU` 的参数顺序;
2. **噪声模拟 vs 实验 05**:同一退极化率下两平台结果应接近——本源噪声虚拟机用 Kraus 算子,与 Aer 的退极化通道同族;
3. **真实芯片**:预期 P(anc=0) < 0.8417(真实噪声),后选择分布保真度 < 1——**这正是实验 05 墙钟账本里"逻辑 T 门时钟"那一列的物理来源**:把真实芯片的保真度代回实验 05 的公式,即可得到该硬件上真实的交叉点;
4. **扩展**:本线路是"QFT + LCU + 后选择"的最小闭环。验证通过后,可批量提交(batch_real_chip_measure,上限 200)扫描不同核 c 的卷积结果,或加入振幅估计回路测试"只取 k 个标量"的读出成本。

---

*相关文件:`experiments/06_origin_quantum.py`;理论对照:`experiments/05_quantum_wallclock.py`(实验五);平台文档:pyqpanda 官方教程「本源量子云服务」章节。*
