# Checkpoint 索引

12 个核心模型按角色分组。**所有 \*.pt\ 已通过 \.gitattributes\ 配置 LFS 追踪**。

## ★ 主结果
- \dfa_executor_k32_final.pt\ (909KB) — 232K 执行器，k≤32 / L≤1000 全部 100% 精确（§4 核心）
- \dfa_executor_final_final.pt\ (909KB) — 同上，备版
- \program_executor_mix_final.pt\ (4.4MB) — C1 RISC-lite 寄存器机执行器
- \ec_final_final.pt\ (5.0MB) — A1/A2 递归程序执行器

## §3 课程化
- \curriculum_ceiling_stage4_final.pt\ (12.7MB) — Phase 1 k=3 课程最佳
- \gru_scale_gru-3m_final.pt\ (12.7MB) — GRU 3M 基线
- \gru_scale_gru-40m_final.pt\ (152MB) — GRU 40M（§3 容量对照关键证据）

## §4 控制组
- \control_transformer_final.pt\ (13.9MB) — 3.4M 因果 Transformer 同编码同监督

## §8 C2 消融
- \c2_gru_final.pt\ (12.7MB) — vanilla GRU（参考基线）
- \c2_nsm_g1_final.pt\ (6.5MB) — NSM 单分区
- \c2_nsm_g4_conc_final.pt\ (17.2MB) — NSM 4 分区+浓度门控（提案版）
- \c2_linearnsm_final.pt\ (6.5MB) — 线性化 NSM（无门控非线性）

## 不在仓内的 checkpoint（可重训）
- \
sm30m_poc/*\ (456MB) — 30M NSM 提案完整版，§8 已证伪，重训约 8 分钟
- \gru_scale_gru-11m/25m_final.pt\ (138MB) — 11M/25M 中间档，scaling 对比只需 3M/40M 两端
- \smoke*/\、gru_poc/、opt2*/、opt3/、opt4/、	race_*/、ec_*/ 中间档 — 训练过程存档
