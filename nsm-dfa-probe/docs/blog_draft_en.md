# A 232K-Parameter Model Beats 7B Transformers at Structured State Tracking: Recursion, Explicit Memory, and an External Tape under the DFA Probe

> Draft. Chinese main version: `blog_draft.md`. All data reproducible under `experiment/checkpoints/`.

## Abstract

Constant-depth Transformers cannot compose state transitions serially: on a controlled DFA state-tracking benchmark, a 7B instruction-tuned Transformer achieves zero structural accuracy (α_struct=0) even with chain-of-thought, and a reasoning model (DeepSeek-R1-7B) likewise fails to reach reliability. We show that (1) a 3.3M-parameter GRU, trained with a fixed-pool curriculum, length-generalizes DFA execution to sequences 40× longer than training on 3-state DFAs but hits a hard ceiling at 4 states; (2) a 232K-parameter neural executor with an explicit transition memory and attention-level process supervision executes arbitrary DFAs with ≤32 states perfectly, generalizing from training length 50 to 1000; (3) under identical input encoding and state supervision, a 3.4M causal Transformer completely fails the same task; (4) the same executor design executes register-machine programs with control flow and recursion, including double recursion, at 100% trace accuracy within the trained depth envelope — and, via an ast-based compiler with idiom selection, executes real Python source end-to-end. We further demonstrate a natural-language middleware (LLM parses rules, executor computes) and an "external tape" of activation traces enabling audit, replay, checkpointing, and memoization with 4-bit quantized states losslessly.

## 1. Why DFA as a Probe

Constant-depth Transformers live in (non-uniform) TC⁰. Iterated DFA transition δ*(q, w) is a minimal serial-composition task (NC¹-flavored). Three falsifiable questions: (1) Can CoT bypass this at inference time? (2) Do recurrent architectures solve it natively? (3) If so, is the bottleneck "reading the spec" or "executing the spec"?

Protocol: α_struct = acc − α_bias − α_surf, with balanced DFAs (permutation transitions) so the stationary distribution is uniform and p_hat ≈ p_pos — no free accuracy from output bias.

## 2. M0 Baseline: α_struct = 0 for 7B Transformers

Qwen2.5-7B (Ollama, temperature 0), balanced DFAs B6 (k=6) and B10 (k=10):

| config | direct acc@50 | acc@100 | acc@200 | α_struct |
|---|---|---|---|---|
| B6 | 0.53 | 0.49 | — | ≈0 |
| B10 | 0.47 | 0.43 | 0.43 | <0 |

CoT does not close the gap (B6 CoT 0.40; B10 CoT 0.73@L=50, decaying with L). The single-step oracle ("current state + symbol → next state") reaches 0.71-0.90: **δ exists, δ* does not**.

DeepSeek-R1-7B (o1-class reasoning model; small n, verbose-thinking truncation lowers valid rate). The B10 CoT L=50 score of 0.86 was an isolated outlier (valid=7/10); at L=100 it collapses to unparseable. We reran this cell with n=50: **only 15/50 responses were parseable, accuracy 0.33 (α_struct=-0.17)** — the original high score does not reproduce and is a survivor-bias artifact. The table below replaces that cell with the n=50 result; other cells retain the original small-sample values as directional reference:

| mode | B6 L=50 | B6 L=100 | B10 L=50 | B10 L=100 |
|---|---|---|---|---|
| direct | 0.67 | 0.83 | 0.56 | 0.71 |
| CoT | 0.33 | 0.50 | **0.33 (n=50, valid=15)** | 0/10 unparseable |

**Conclusion 1**: even a reasoning-trained 7B model is nowhere near reliable. The B10 CoT L=50 0.86 has been falsified by the n=50 rerun: only 15/50 parseable, accuracy 0.33, α_struct=-0.17. Test-time compute does not solve spec→execution conversion.

## 3. Phase 1: The Curriculum Ceiling is k=3

A 3.3M GRU (d=512, 2 layers), task: byte-encoded DFA spec + input string → per-step state prediction.

**Single DFA**: 100% full-trajectory accuracy on unseen strings up to L=2000 — recurrent architectures do carry δ*.

**Cross-DFA curriculum** (fixed pool, expanding; then k up):

| stage | config | pool memorization | new-DFA acc | α_struct |
|---|---|---|---|---|
| S1 | k=3, pool=5 | 100% | 10% | -0.35 |
| S2 | k=3, pool→20 | 100% | 40% | +0.03 |
| S3 | k=3, pool→100 | 100% | **91.3%** | **+0.49** |
| S4 | k=3, L=50 | 100% | **92.7%** (90.7% @L=100, 64.7% @L=200) | **+0.48** |
| S5 | k=4 | 100% | 2.7% | -0.22 |
| S6/S7 | k=5/6 | 100% | ~0% | <0 |

**Conclusion 2**: pure learning has a sharp ceiling. At k=3 the model genuinely learns "parse an arbitrary new DFA spec and execute it" (with length generalization); k=4 collapses, with catastrophic forgetting of k=3. The bottleneck is not δ* composition but **compressing the spec into the hidden state**. Scaling 3.3M→40M changes nothing. The table below shows that from 3.3M to 39.8M parameters, full-trajectory accuracy on new k=10 DFAs stays near zero (state loss≈log(10), i.e., random guessing), while training time grows linearly.

| model | params | k=10 L=50 acc | L=100 acc | L=200 acc | train time |
|---|---|---|---|---|---|
| GRU-3M | 3.3M | 0.000 | 0.025 | 0.010 | 0.9m |
| GRU-11M | 10.9M | 0.000 | 0.025 | 0.010 | 2.0m |
| GRU-25M | 25.5M | 0.000 | 0.025 | 0.010 | 5.1m |
| GRU-40M | 39.8M | 0.000 | 0.025 | 0.010 | 8.2m |

**Conclusion**: it is task structure, not capacity.

## 4. Phase 2: Neural DFA Executor — Explicit Memory + Process Supervision

Architecture (232K params): structured spec encoding (transition token id carries (q,sym); the next token is the next state id) → memory key = k_proj(state_emb(q)+sym_emb(sym)), value = v_proj(state_emb(next)) → sequential soft retrieval with r₀ = state_emb(0), query = q_proj(r + sym_emb(input)), r ← LayerNorm(Σ attn·values).

Trained on random DFAs (k∈[2,10], L∈[20,50]) with state CE + verdict CE + **attention-level process supervision** (which table entry each step should query — the analogue of process supervision in reasoning models).

**Results** (new random DFAs): k=3/6/10 at L=50-1000 → **100%** (α up to +0.75); k=20 (trained to 20) → 100%/98.7%; k=32 (trained to 32) → 100%. Trained only at L≤50. The k=32 failure was resolved by retraining with k_range=(2,32), L∈[20,50], 4000 steps (~3h on RTX 4060), all other hyperparameters unchanged. Coverage-limited, not capacity-limited.

**Core ablation — process supervision is a switch**: identical architecture/data/steps without attention supervision fail to fit even the training batch (step accuracy ~45%); with it, 100% within 500 steps.

**Control**: a 3.4M causal Transformer (4 layers, same structured input, same state supervision, 4000 steps) gets k=3 L=50 acc=0.17, k=6/10 ≤0.02, α all negative. The gap is architecture, not encoding, data, or supervision.

**Conclusion 3**: serial state maintenance (sequential query loop) + explicit memory (KV transition table) suffice for exact execution with 232K parameters; constant-depth parallel computation cannot substitute.

## 5. NL Middleware: LLM Translates, Executor Computes

DFA described in natural language (3 templates: table T1 / rule T2 / shuffled+synonyms T3). LLM (qwen2.5:7b / qwen2.5-coder:7b) parses to JSON spec; a deterministic algorithm or the Executor runs it (k=4, full-trajectory accuracy):

| condition | T1 | T2 | T3 |
|---|---|---|---|
| A pure LLM direct | **0.00** | **0.00** | **0.00** |
| B pure LLM CoT | **0.00** | **0.00** | **0.00** |
| C LLM parse + algorithm | 0.93~1.00 | 0.67~0.93 | 0.33~0.40 |
| D LLM parse + Executor | **identical to C** | | |

Three points: (1) execution is never the bottleneck — length-invariant; (2) C≡D: on every spec the LLM parsed correctly, the 232K learned executor is trajectory-equivalent to the handwritten algorithm; (3) the only bottleneck is LLM parsing: after optimization (coder model + isomorphic few-shot + permutation-prior repair) T1=1.00, T2≈0.93, T3≈0.40 (k=4, n=30); at k=6 everything degrades, and the n=100 rerun is harsher than n=30 — T1=0.92, T2=0.65 (0.80 at n=30, a small-sample overestimate), T3=0.03 (0.07 at n=30); given a correct parse, execution stays 100% (reconfirmed at n=100). Voting does not fix systematic errors.

**Conclusion 4**: confining the probabilistic LLM to the narrow "NL → formal language" interface and handing serial composition to a verifiable executor is an empirically validated layered architecture.

## 6. The External Tape: Thought Traces on Disk

Each execution step writes its activation trace (queried transition index + state) to a file:

- **Auditable & replayable**: 1000 steps = 5.5KB; replay from file alone reproduces the state trajectory 100% — no model rerun needed.
- **Segmented execution**: L=20000 in 20 segments, activation vector written to disk at segment boundaries, restored losslessly. Key finding: **the activation vector r itself must be stored — the decoded symbolic state id fails**. On 20 random k=10 DFAs with L=200, saving r recovers the full trajectory exactly in 20/20 cases; saving the decoded state id fails in 20/20 (first error at the segment boundary, median step 50), and even re-embedding the ground-truth state id recovers 0/20 exactly. Distributed representations cannot be rebuilt from symbols; "neural notation" must be activation, not symbol.
- **Quantization**: 4-bit (16-level) quantized state vectors are **lossless** (L=20000: 19 checkpoints, 2.5KB total; with segment length 4000: 4 checkpoints, 640B).
- **Memoization**: 100 queries sharing a 500-symbol prefix → 9.0× speedup.
- **Checkpoint/resume**: interrupt anywhere, resume from tape, bit-identical continuation.

**Conclusion 5**: state = O(1)-compressed history. Moving history off the context tape to an external file yields fixed-window processing of arbitrary length, with auditability, replay, and caching.

## 7. Generalization: From DFAs to Program Execution (C1/A1/A2)

**C1 — RISC-lite register machine** (8 registers, mod-16 arithmetic; mov/inc/dec/add/sub/seti/jmp/jz/jnz/halt): instruction memory keyed by pure pc embeddings; semantics MLP over (op, pc, val(reg_a), val(reg_b), imm). Training recipe: **triple teacher forcing (fetch slot, register history, pc) + scheduled sampling**, STE for writes/jumps. After a mixed curriculum (arith → control flow → loops → mix), unseen programs of 10/14/16 instructions (100 each): **exact trace = 1.000 everywhere**.

**A1 — recursion** (+mul/push/pop/call/ret; value stack + return stack; stack pointer updated **mechanically** from op syntax — not learned — which is what makes depth handling possible):

| program | in-distribution exact | depth generalization |
|---|---|---|
| fact/sum (linear recursion) | **1.000** (n≤10) | boundary = trained depth |
| **fib (double recursion)** | **1.000** (n≤6) | fails beyond |
| countdown (iteration) | **1.000** | — |
| random stack programs | 0.975 | — |

Three depth-generalization obstacles were found by printing the first divergent step: sp-embedding depth cap (→ mechanical sp), value coverage (→ random arithmetic programs), stack-value coverage (→ push/pop pairs). **Key scientific finding: recursion-depth generalization is not free, unlike DFA length generalization** — register values flow through data-dependent branches (jz), so value errors alter control flow and cascade. The failure boundary tracks the training depth and moves outward as training deepens.

**A2 — Python frontend**: a real ast-based compiler (def / if-else / while / recursion / + - *) compiles Python source to the instruction set, with **idiom selection** targeting executor-trained template structures (standard compiler-backend practice). End-to-end: **fact / sum / fib / countdown Python program families — 12 tests, all 100% exact-trace execution by the neural executor**.

**Conclusion 6**: explicit memory + process supervision + structured parsing generalizes from DFAs to programs with control flow, loops, and recursion, and executes compiled real-Python source. Limitation: a general compiler without idiom selection gets only 65%/0% — the executor learns program semantics within the training distribution, not unbounded universal execution; program forms outside the training templates, such as mutual recursion, closures, lists, or dynamic memory, currently fail.

## 8. Ablation Verdict: Is Concentration Gating Necessary? (C2)

Closing the loop on the original NSM proposal. Task: Phase-1 k=3 fixed-pool curriculum, equal budget:

| arm | params | new-k3 acc@L20 (5000 steps) | s/step |
|---|---|---|---|
| **vanilla GRU** | 3.3M | **0.407** | 0.01s |
| NSM 4-partition + concentration gating (proposal) | 4.5M | 0.220 | 0.5s |
| NSM single-partition | 1.7M | 0.167* | 0.4s |
| **linearized NSM (no gating nonlinearity)** | 1.7M | **0.033 (collapsed)** | 0.2s |

*3000 steps.

| hypothesis | verdict |
|---|---|
| H1 Transformers lack serial depth; recurrence needed | ✅ confirmed |
| H2 nonlinear state update needed | ✅ confirmed (linearized arm collapses) |
| H3 concentration gating / multi-timescale partitions help | ❌ **falsified**: GRU is better and 40× faster |
| H4 explicit memory / external structure needed | ✅ confirmed beyond expectation (Executor + tape) |

**Conclusion 7**: concentration gating is decorative. The effective ingredients are nonlinear recurrence + explicit memory + process supervision — all available as cheap off-the-shelf primitives.

## 9. Deployment Verification: Anatomy of a False Alarm

After the system shipped as an Ollama-compatible service, an external review of 13 tape files alleged: "same DFA, same input, inconsistent execution; accuracy only 38.5%; state_head flips at the boundary under floating-point noise; should use verdict_head." Measured verdicts:

| claim | measured |
|---|---|
| 38.5% accuracy | **miscounted**: 8/13 correct = 61.5% |
| Executor non-determinism / boundary flip | **falsified**: bit-identical outputs across 3 reruns; contested-step logit margin 11.7 (9.83 vs -1.88); 3 end-to-end server calls → byte-identical tapes |
| should use verdict_head | **harmful**: verdict_head never saw accepting sets during training (model input contains only the transition table) — it is structurally a dummy head and answers REJECT on correct executions. The state-based verdict is the correct design |

The true source of divergence: different NL descriptions / edited inputs → differently parsed specs → step 8 is the first step using the altered transition; identical attention sequences for steps 1-7 because the query structure is unchanged — only the parsed target value differs. **The Executor faithfully executed the spec it received — a wrong spec yields a wrong result; that is not an executor bug.**

Additional evidence: at the demo's exact configuration (k=4, L=8), formal evaluation on 200 new DFAs: acc=1.000 — still perfect in the deployment distribution.

The false alarm is itself the best demonstration of the tape's value: because every step's "which rule was queried, which state was reached" is on disk, the problem could be localized to the input side rather than the execution side. We then added a spec header to each tape (`# k=... accepting=... spec_hash=...`), making every trace attributable to the exact spec it executed.

## 10. The Full Picture

```
input tape (symbols / natural language)
      │
      ▼
parse layer: LLM (NL → formal spec)          ← only bottleneck: T3/k=6 parsing 0.03
      │                                        k=6 table/rule format 0.65-0.92 (n=100)
      ▼
execution layer: Neural Executor (explicit memory + serial queries) ← any k≤32, any L, 100%
      │
      ▼
external layer: activation tape (audit / replay / checkpoint / memoize) ← 4-bit lossless
```

| method | params | k=3 | k=6 | k=10 | k=20 | length gen |
|---|---|---|---|---|---|---|
| Qwen2.5-7B direct | 7B | — | 0.53 | 0.47 | — | decays |
| Qwen2.5-7B CoT | 7B | — | 0.40 | 0.73 | — | decays |
| R1-7B direct | 7B | — | 0.67 | 0.56 | — | 0.83/0.71@L100* |
| R1-7B CoT | 7B | — | 0.33 | **0.33** | — | unparseable@L100 |
| 3.4M Transformer (control) | 3.4M | 0.17 | 0.02 | 0.01 | — | none |
| 3.3M GRU curriculum | 3.3M | **0.93** | ~0 | ~0 | ~0 | k=3: 65%@L200 |
| **232K Executor** | **0.23M** | **1.00** | **1.00** | **1.00** | **1.00** | **100%@L1000** |

*Other R1 cells remain small-sample (n=10-12, low valid rate); directional only. The B10 CoT L=50 n=50 rerun is complete: valid=15/50, acc=0.33, α_struct=-0.17.

## 11. Limitations and Open Questions

1. **T3/k=6 parsing**: shuffled+synonym NL rule parsing is below 50% at 7B scale — the only bottleneck for the middleware. Constrained decoding, larger parsers, or human-in-the-loop confirmation are candidates. (k=6 now at n=100: T1=0.92, T2=0.65, T3=0.03 — T2 revised down from 0.80 at n=30, T3 from 0.07; only k=4 T3≈0.40 remains n=30, interpret as roughly ±10pp.)
2. **k=4 cliff**: pure-learning spec parsing has a phase transition; whether capacity, representation, or optimization is undecided.
3. **Concentration gating falsified** (§8) on the DFA curriculum; longer-range multi-timescale tasks not covered.
4. **Tasks remain structured toys**: RISC-lite has no recursion-depth generalization, no mutual recursion, no multi-arg functions; mod-16 values, stack 64, ≤48 instructions.
5. **R1 sample size small** (n=10-12 with low valid rate from thinking truncation). The B10 CoT L=50 n=50 rerun is complete: valid=15/50, acc=0.33 (α_struct=-0.17), falsifying the original 0.86 outlier.

## 12. Discussion

Using DFA as a probe, we tested the "can / cannot" boundaries of current AGI candidate routes one by one: pure scaling (no effect at 40M), test-time compute (R1 still fails), CoT (doesn't close), pure-learning spec reading (k=3 ceiling). What works: **neural-symbolic layering + process supervision + external tape**. This suggests a candidate shape for next-generation architectures — LLMs handle the fuzzy natural-language interface, learnable executors handle exact serial computation within the training distribution, and an external tape provides auditable state persistence. The program executor is not a universal interpreter; its precision is bounded by the distribution of training templates.

## Data & Reproduction Index

| section | data |
|---|---|
| §2 M0 | `checkpoints/results_baseline.json`, `results_v1_cot_step.json`, `r1_baseline.json`, `r1_b10_l50_cot_n50.json` |
| §3 Phase 1 | `checkpoints/curriculum_ceiling/metrics.json`, `stage4_final.pt` |
| §4 Phase 2 | `checkpoints/dfa_executor_final/`, `dfa_executor_k32/`, `control_transformer/` |
| §5 NL | `checkpoints/nl_middleware_report*.json`, `parse_ablation_report.json`, `nl_big/report.json`, `nl_big_n100/report.json` |
| §6 tape | `checkpoints/external_layer/`, `tape_deep/report.json`, `nl_tape/`, `r_vs_id/report.json` |
| §7 programs | `checkpoints/program_executor_mix/`, `rec_final/`, `rec_a2c/`, `C1-RISC程序执行报告.md`, `A1-递归程序执行报告.md`, `A2-Python前端报告.md` |
| §8 C2 | `checkpoints/c2_ablation.json`, `c2_final_showdown.json`, `C2-NSM门控消融报告.md` |
| §9 deployment | `评审验证报告.md`, `checkpoints/tapes/` |
