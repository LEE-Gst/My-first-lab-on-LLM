# -*- coding: utf-8 -*-
"""DFA 数据管线：排列转移平衡 DFA + 轨迹监督 + 课程化采样。"""
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch


# 字节编码约定（与 ASCII 不冲突）
BYTE_START = 200
BYTE_SEP = 201
BYTE_END = 202
BYTE_SYM0 = 64   # symbol 0 编码为 byte 64
STATE_OFFSET = 128  # state q 编码为 byte 128+q
VERDICT_ACCEPT = 192
VERDICT_REJECT = 193

# 结构化编码：把 (q, sym) 转移用一个专用 transition token 表示
TRANSITION_OFFSET = 70  # transition token = 70 + q*m + sym


def make_balanced_dfa(k: int, m: int, f_size: int, seed: Optional[int] = None):
    """每符号为排列转移的 DFA：平稳分布均匀，p_hat ≈ f_size/k。"""
    rng = random.Random(seed)
    delta = {}
    for sym in range(m):
        perm = list(range(k))
        rng.shuffle(perm)
        for q in range(k):
            delta[(q, sym)] = perm[q]
    states = list(range(k))
    if f_size > k:
        f_size = k
    accepting = set(rng.sample(states, f_size))
    return {"k": k, "m": m, "start": 0, "accepting": sorted(accepting), "delta": delta}


def run_dfa(dfa, word):
    q = dfa["start"]
    for sym in word:
        q = dfa["delta"][(q, sym)]
    return q in set(dfa["accepting"])


def dfa_trajectory(dfa, word):
    """返回 (states, verdict)：states[t] 是读完 word[:t] 后的状态。"""
    states = [dfa["start"]]
    q = dfa["start"]
    for sym in word:
        q = dfa["delta"][(q, sym)]
        states.append(q)
    # states 长度 = L+1；我们关心的是输入每个字节后的状态 states[1:]
    return states[1:], q in set(dfa["accepting"])


def encode_dfa_spec(dfa) -> List[int]:
    """把 DFA 规格编码成字节列表。

    格式：[START, k, m, |F|, accepting..., transitions..., SEP]
    transitions: 对每个 (q,sym) 输出 q, sym_byte, next_q
    """
    k, m = dfa["k"], dfa["m"]
    spec = [BYTE_START, k, m, len(dfa["accepting"])]
    spec.extend(dfa["accepting"])
    for q in range(k):
        for sym in range(m):
            nxt = dfa["delta"][(q, sym)]
            spec.extend([q, BYTE_SYM0 + sym, nxt])
    spec.append(BYTE_SEP)
    return spec


def encode_sample(dfa, word) -> Tuple[List[int], List[int], int]:
    """编码一个训练样本。

    Returns:
        tokens: [spec bytes, input bytes, END]
        state_labels: 每个输入字节对应的真实状态（与 input bytes 对齐）
        verdict_label: 1=ACCEPT, 0=REJECT
    """
    spec = encode_dfa_spec(dfa)
    states, verdict = dfa_trajectory(dfa, word)
    input_bytes = [BYTE_SYM0 + sym for sym in word]
    tokens = spec + input_bytes + [BYTE_END]
    state_labels = states
    return tokens, state_labels, 1 if verdict else 0


def encode_structured_dfa_spec(dfa) -> List[int]:
    """结构化 DFA 规格编码：转移用独立 token，形如 [START,k,m,|F|,F...,T(q,sym),next_q,...,SEP]。

    transition token = TRANSITION_OFFSET + q*m + sym，明确区分"查询哪个转移"和"目标状态"。
    """
    k, m = dfa["k"], dfa["m"]
    spec = [BYTE_START, k, m, len(dfa["accepting"])]
    spec.extend(dfa["accepting"])  # 直接用状态值作为 token
    for q in range(k):
        for sym in range(m):
            nxt = dfa["delta"][(q, sym)]
            spec.extend([TRANSITION_OFFSET + q * m + sym, nxt])
    spec.append(BYTE_SEP)
    return spec


def encode_structured_sample(dfa, word) -> Tuple[List[int], List[int], int]:
    """结构化训练样本。"""
    spec = encode_structured_dfa_spec(dfa)
    states, verdict = dfa_trajectory(dfa, word)
    input_bytes = [BYTE_SYM0 + sym for sym in word]
    tokens = spec + input_bytes + [BYTE_END]
    return tokens, states, 1 if verdict else 0


def encode_trace_sample(dfa, word) -> List[int]:
    """交错格式：[spec, input_1, state_1, input_2, state_2, ..., input_L, state_L, verdict]。

    TraceTransformer 因果 LM 训练用。
    """
    spec = encode_dfa_spec(dfa)
    states, verdict = dfa_trajectory(dfa, word)
    seq = list(spec)
    for sym, state in zip(word, states):
        seq.append(BYTE_SYM0 + sym)
        seq.append(STATE_OFFSET + state)
    seq.append(VERDICT_ACCEPT if verdict else VERDICT_REJECT)
    return seq


def encode_trace_prompt(dfa, word) -> List[int]:
    """交错 prompt：[spec, input_1, input_2, ..., input_L]（模型将自回归补 state_i）。"""
    spec = encode_dfa_spec(dfa)
    seq = list(spec)
    for sym in word:
        seq.append(BYTE_SYM0 + sym)
    return seq


def state_from_token(t: int) -> Optional[int]:
    if STATE_OFFSET <= t < STATE_OFFSET + 64:
        return t - STATE_OFFSET
    return None


def verdict_from_token(t: int) -> Optional[bool]:
    if t == VERDICT_ACCEPT:
        return True
    if t == VERDICT_REJECT:
        return False
    return None


@dataclass
class DFADataConfig:
    m: int = 2                    # 字母表大小
    k_range: Tuple[int, int] = (5, 32)     # 状态数范围
    L_range: Tuple[int, int] = (20, 200)   # 字符串长度范围
    batch_size: int = 16
    max_states: int = 64
    curriculum: bool = True       # 是否启用课程化
    seed: Optional[int] = None


class DFACurriculumSampler:
    """根据训练状态动态调整 DFA 难度。"""

    def __init__(self, cfg: DFADataConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        # 当前难度窗口
        self.k_min, self.k_max = cfg.k_range[0], min(cfg.k_range[0] + 5, cfg.k_range[1])
        self.L_min, self.L_max = cfg.L_range[0], min(cfg.L_range[0] + 20, cfg.L_range[1])
        self.stats = []  # (reward_mean, reward_std) 最近若干窗口

    def sample_dfa_and_word(self):
        k = self.rng.randint(self.k_min, self.k_max)
        L = self.rng.randint(self.L_min, self.L_max)
        # 接受状态数：尽量让 p_pos ≈ 0.5（k 偶数则一半，奇数则 (k±1)/2 随机）
        f_size = k // 2
        if k % 2 == 1 and self.rng.random() < 0.5:
            f_size = (k + 1) // 2
        dfa = make_balanced_dfa(k, self.cfg.m, f_size, seed=self.rng.randint(0, 2**31))
        word = tuple(self.rng.randrange(self.cfg.m) for _ in range(L))
        return dfa, word

    def update(self, reward_mean: float, reward_std: float):
        """根据上一轮奖励统计调整难度。"""
        self.stats.append((reward_mean, reward_std))
        if len(self.stats) > 10:
            self.stats.pop(0)
        if not self.cfg.curriculum:
            return
        # 如果平均 reward 高且方差合理 → 增加难度
        if reward_mean > 0.85 and reward_std > 0.05:
            self.L_max = min(self.L_max + 10, self.cfg.L_range[1])
            if self.L_max - self.L_min > 30:
                self.L_min = min(self.L_min + 5, self.cfg.L_range[0] + 20)
            if reward_mean > 0.95:
                self.k_max = min(self.k_max + 2, self.cfg.k_range[1])
        # 如果方差太低（全成功/全失败）→ 降低难度或增加多样性
        if reward_std < 0.02:
            self.L_max = max(self.L_max - 10, self.cfg.L_range[0] + 10)
            self.k_max = max(self.k_max - 2, self.cfg.k_range[0] + 2)

    def get_difficulty(self):
        return {"k": (self.k_min, self.k_max), "L": (self.L_min, self.L_max)}


class DFAIterDataset:
    """生成式 DFA 数据集，与 DataLoader 不同，它 infinite yield 批次。"""

    def __init__(self, cfg: DFADataConfig):
        self.cfg = cfg
        self.sampler = DFACurriculumSampler(cfg)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            batch_tokens = []
            batch_state_labels = []
            batch_verdicts = []
            max_len = 0
            samples = []
            for _ in range(self.cfg.batch_size):
                dfa, word = self.sampler.sample_dfa_and_word()
                tokens, states, verdict = encode_sample(dfa, word)
                samples.append((tokens, states, verdict))
                max_len = max(max_len, len(tokens))

            # padding to max_len
            pad_id = 0
            for tokens, states, verdict in samples:
                pad_len = max_len - len(tokens)
                batch_tokens.append(tokens + [pad_id] * pad_len)
                spec_len_i = len(tokens) - len(states) - 1
                state_seq = [-100] * spec_len_i + states + [-100]
                state_seq = state_seq + [-100] * pad_len
                batch_state_labels.append(state_seq)
                batch_verdicts.append(verdict)

            yield {
                "input_ids": torch.tensor(batch_tokens, dtype=torch.long),
                "state_labels": torch.tensor(batch_state_labels, dtype=torch.long),
                "verdict_labels": torch.tensor(batch_verdicts, dtype=torch.long),
            }

    def update_curriculum(self, reward_mean: float, reward_std: float):
        self.sampler.update(reward_mean, reward_std)

class DFAMetaSampler:
    """每批采样少量 DFA，每个 DFA 生成多个词。让模型先'记住'一个 DFA 再泛化。"""

    def __init__(self, cfg: DFADataConfig, dfas_per_batch: int = 4, words_per_dfa: int = 8):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.dfas_per_batch = dfas_per_batch
        self.words_per_dfa = words_per_dfa

    def sample_dfa_and_words(self):
        k = self.rng.randint(self.cfg.k_range[0], self.cfg.k_range[1])
        L = self.rng.randint(self.cfg.L_range[0], self.cfg.L_range[1])
        f_size = k // 2
        if k % 2 == 1 and self.rng.random() < 0.5:
            f_size = (k + 1) // 2
        dfa = make_balanced_dfa(k, self.cfg.m, f_size, seed=self.rng.randint(0, 2**31))
        words = [tuple(self.rng.randrange(self.cfg.m) for _ in range(L)) for _ in range(self.words_per_dfa)]
        return dfa, words


class DFAMetaIterDataset:
    """元学习式数据集：每批 N 个 DFA × M 个词。"""

    def __init__(self, cfg: DFADataConfig, dfas_per_batch: int = 4, words_per_dfa: int = 8):
        self.cfg = cfg
        self.sampler = DFAMetaSampler(cfg, dfas_per_batch, words_per_dfa)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            samples = []
            max_len = 0
            for _ in range(self.sampler.dfas_per_batch):
                dfa, words = self.sampler.sample_dfa_and_words()
                for word in words:
                    tokens, states, verdict = encode_sample(dfa, word)
                    samples.append((tokens, states, verdict))
                    max_len = max(max_len, len(tokens))

            pad_id = 0
            batch_tokens = []
            batch_state_labels = []
            batch_verdicts = []
            for tokens, states, verdict in samples:
                pad_len = max_len - len(tokens)
                batch_tokens.append(tokens + [pad_id] * pad_len)
                spec_len_i = len(tokens) - len(states) - 1
                state_seq = [-100] * spec_len_i + states + [-100]
                state_seq = state_seq + [-100] * pad_len
                batch_state_labels.append(state_seq)
                batch_verdicts.append(verdict)

            yield {
                "input_ids": torch.tensor(batch_tokens, dtype=torch.long),
                "state_labels": torch.tensor(batch_state_labels, dtype=torch.long),
                "verdict_labels": torch.tensor(batch_verdicts, dtype=torch.long),
            }

    def update_curriculum(self, reward_mean: float, reward_std: float):
        pass


class DFAStructuredIterDataset:
    """结构化编码版 DFAIterDataset。"""

    def __init__(self, cfg: DFADataConfig):
        self.cfg = cfg
        self.sampler = DFACurriculumSampler(cfg)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            samples = []
            max_len = 0
            for _ in range(self.cfg.batch_size):
                dfa, word = self.sampler.sample_dfa_and_word()
                tokens, states, verdict = encode_structured_sample(dfa, word)
                samples.append((tokens, states, verdict))
                max_len = max(max_len, len(tokens))

            pad_id = 0
            batch_tokens, batch_state_labels, batch_verdicts = [], [], []
            for tokens, states, verdict in samples:
                pad_len = max_len - len(tokens)
                batch_tokens.append(tokens + [pad_id] * pad_len)
                spec_len_i = len(tokens) - len(states) - 1
                state_seq = [-100] * spec_len_i + states + [-100]
                state_seq = state_seq + [-100] * pad_len
                batch_state_labels.append(state_seq)
                batch_verdicts.append(verdict)

            yield {
                "input_ids": torch.tensor(batch_tokens, dtype=torch.long),
                "state_labels": torch.tensor(batch_state_labels, dtype=torch.long),
                "verdict_labels": torch.tensor(batch_verdicts, dtype=torch.long),
            }

    def update_curriculum(self, reward_mean: float, reward_std: float):
        self.sampler.update(reward_mean, reward_std)


class DFAStructuredMetaIterDataset:
    """结构化编码版 DFAMetaIterDataset。"""

    def __init__(self, cfg: DFADataConfig, dfas_per_batch: int = 4, words_per_dfa: int = 8):
        self.cfg = cfg
        self.dfas_per_batch = dfas_per_batch
        self.words_per_dfa = words_per_dfa
        self.rng = random.Random(cfg.seed)

    def _sample_dfa_and_words(self):
        k = self.rng.randint(self.cfg.k_range[0], self.cfg.k_range[1])
        L = self.rng.randint(self.cfg.L_range[0], self.cfg.L_range[1])
        f_size = k // 2
        if k % 2 == 1 and self.rng.random() < 0.5:
            f_size = (k + 1) // 2
        dfa = make_balanced_dfa(k, self.cfg.m, f_size, seed=self.rng.randint(0, 2**31))
        words = [tuple(self.rng.randrange(self.cfg.m) for _ in range(L)) for _ in range(self.words_per_dfa)]
        return dfa, words

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            samples = []
            max_len = 0
            for _ in range(self.dfas_per_batch):
                dfa, words = self._sample_dfa_and_words()
                for word in words:
                    tokens, states, verdict = encode_structured_sample(dfa, word)
                    samples.append((tokens, states, verdict))
                    max_len = max(max_len, len(tokens))

            pad_id = 0
            batch_tokens, batch_state_labels, batch_verdicts = [], [], []
            for tokens, states, verdict in samples:
                pad_len = max_len - len(tokens)
                batch_tokens.append(tokens + [pad_id] * pad_len)
                spec_len_i = len(tokens) - len(states) - 1
                state_seq = [-100] * spec_len_i + states + [-100]
                state_seq = state_seq + [-100] * pad_len
                batch_state_labels.append(state_seq)
                batch_verdicts.append(verdict)

            yield {
                "input_ids": torch.tensor(batch_tokens, dtype=torch.long),
                "state_labels": torch.tensor(batch_state_labels, dtype=torch.long),
                "verdict_labels": torch.tensor(batch_verdicts, dtype=torch.long),
            }

    def update_curriculum(self, reward_mean: float, reward_std: float):
        pass


class DFATraceIterDataset:
    """TraceTransformer 用：交错 input/state 因果 LM 数据集。"""

    def __init__(self, cfg: DFADataConfig):
        self.cfg = cfg
        self.sampler = DFACurriculumSampler(cfg)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            seqs = []
            max_len = 0
            for _ in range(self.cfg.batch_size):
                dfa, word = self.sampler.sample_dfa_and_word()
                seq = encode_trace_sample(dfa, word)
                seqs.append(seq)
                max_len = max(max_len, len(seq))

            pad_id = 0
            batch = []
            for seq in seqs:
                batch.append(seq + [pad_id] * (max_len - len(seq)))

            ids = torch.tensor(batch, dtype=torch.long)
            labels = torch.full_like(ids, -100)
            labels[:, :-1] = ids[:, 1:]
            yield {"input_ids": ids, "labels": labels}

    def update_curriculum(self, reward_mean: float, reward_std: float):
        self.sampler.update(reward_mean, reward_std)


