# -*- coding: utf-8 -*-
"""NSM (Neuronal State Machine) 字节级语言模型。

核心设计：
- 多层非线性递归，每层维护 G 个状态分区（"组织"）
- 每个分区有独立的 GRU 风格更新门 / 复位门（满足 D1：非线性转移）
- 全局池化产生"浓度"门控 c^(g)，控制各分区更新速率（多时间尺度记忆）
- 顺序处理提供 O(L) 串行深度，绕开 Transformer 的 TC^0 上限
- 预留 chunked_scan 接口，后续可换硬件对齐实现
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NSMConfig:
    vocab_size: int = 256          # 字节词表
    d_model: int = 512             # 每分区维度
    n_layers: int = 8              # NSM 层数
    n_partitions: int = 4          # G：状态分区数（"组织数"）
    mlp_ratio: float = 2.67        # FFN/MLP 扩展比
    dropout: float = 0.0
    max_states: int = 64           # state_head 输出上限
    use_concentration: bool = True # 是否启用浓度门控（H4 消融）
    use_nonlinearity: bool = True  # 是否启用 GRU 门控非线性（H2 消融）
    chunk_size: int = 128          # 硬件对齐块大小（NSA 启发）


@dataclass
class TraceTransformerConfig:
    """TraceTransformer：用 CoT 式生成状态轨迹的因果 Transformer。"""
    vocab_size: int = 256
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    max_seq_len: int = 512
    dropout: float = 0.0


class TraceTransformer(nn.Module):
    """因果 Transformer，输入 [spec, input, SEP] 后自回归生成 [state_1...state_L, verdict]。"""

    def __init__(self, cfg: TraceTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=int(cfg.d_model * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embedding.weight  # tied

        self.register_buffer('causal_mask',
                             nn.Transformer.generate_square_subsequent_mask(cfg.max_seq_len))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        for name, p in self.named_parameters():
            if 'embedding' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)
        mask = self.causal_mask[:L, :L]
        x = self.encoder(x, mask=mask, is_causal=True)
        logits = self.head(x)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class NSMCell(nn.Module):
    """单个 NSM 递归步：处理一个输入字节，更新 G 个分区状态。"""

    def __init__(self, cfg: NSMConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        G = cfg.n_partitions
        self.d = d
        self.G = G

        # 每个分区独立的输入/隐藏门控参数 (G, d, d) + (G, d)
        self.W_ir = nn.Parameter(torch.empty(G, d, d))
        self.b_ir = nn.Parameter(torch.empty(G, d))
        self.W_hr = nn.Parameter(torch.empty(G, d, d))
        self.b_hr = nn.Parameter(torch.empty(G, d))
        self.W_iz = nn.Parameter(torch.empty(G, d, d))
        self.b_iz = nn.Parameter(torch.empty(G, d))
        self.W_hz = nn.Parameter(torch.empty(G, d, d))
        self.b_hz = nn.Parameter(torch.empty(G, d))
        self.W_ih = nn.Parameter(torch.empty(G, d, d))
        self.b_ih = nn.Parameter(torch.empty(G, d))
        self.W_hh = nn.Parameter(torch.empty(G, d, d))
        self.b_hh = nn.Parameter(torch.empty(G, d))

        # 浓度门控：慢变量 MLP，从全局池化状态产生 G 个标量
        if cfg.use_concentration:
            self.conc_mlp = nn.Sequential(
                nn.Linear(d, max(16, d // 4)),
                nn.SiLU(),
                nn.Linear(max(16, d // 4), G),
                nn.Sigmoid(),
            )
        else:
            self.conc_mlp = None

        # 输出投影：把 G 个分区拼起来 -> d，再可选 FFN
        self.out_proj = nn.Linear(G * d, d, bias=False)
        hidden = int(d * cfg.mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg.dropout)

        self._init_weights()

    def _init_weights(self):
        # 递归隐藏权重正交初始化
        for W in [self.W_hr, self.W_hz, self.W_hh]:
            for g in range(self.G):
                nn.init.orthogonal_(W[g])
                with torch.no_grad():
                    W[g] *= 0.9
        # 输入权重 xavier
        for W in [self.W_ir, self.W_iz, self.W_ih]:
            nn.init.xavier_uniform_(W, gain=math.sqrt(2.0))
        # 偏置
        for b in [self.b_ir, self.b_hr, self.b_iz, self.b_hz, self.b_ih, self.b_hh]:
            nn.init.zeros_(b)
        # 更新门偏置初始为 -1：开始时偏向保留旧状态（长记忆）
        nn.init.constant_(self.b_hz, -1.0)
        # 输出/FFN
        nn.init.xavier_uniform_(self.out_proj.weight)
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _gate(self, x: torch.Tensor, h: torch.Tensor,
              W_ix: nn.Parameter, b_ix: nn.Parameter,
              W_ih: nn.Parameter, b_ih: nn.Parameter,
              act=torch.sigmoid) -> torch.Tensor:
        """计算门控，x: (B, d), h: (B, G, d) -> gate: (B, G, d)"""
        gx = torch.einsum('bd,gde->bge', x, W_ix) + b_ix   # (B, G, d)
        gh = torch.einsum('bgd,gde->bge', h, W_ih) + b_ih  # (B, G, d)
        return act(gx + gh)

    def step(self, x: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """单步前向。

        Args:
            x: (B, d) 当前输入表示
            h_prev: (B, G, d) 上一时刻状态
        Returns:
            out: (B, d) 当前步输出表示
            h_new: (B, G, d) 新状态
        """
        cfg = self.cfg
        B, G, d = h_prev.shape

        if cfg.use_nonlinearity:
            r = self._gate(x, h_prev, self.W_ir, self.b_ir, self.W_hr, self.b_hr, torch.sigmoid)
            z = self._gate(x, h_prev, self.W_iz, self.b_iz, self.W_hz, self.b_hz, torch.sigmoid)
            # candidate
            gx_h = torch.einsum('bd,gde->bge', x, self.W_ih) + self.b_ih
            h_reset = h_prev * r
            gh_h = torch.einsum('bgd,gde->bge', h_reset, self.W_hh) + self.b_hh
            tilde = torch.tanh(gx_h + gh_h)
            h = (1 - z) * h_prev + z * tilde
        else:
            # 线性化消融（真线性：无 tanh 候选，测试非线性是否必要）
            gx_h = torch.einsum('bd,gde->bge', x, self.W_ih) + self.b_ih
            gh_h = torch.einsum('bgd,gde->bge', h_prev, self.W_hh) + self.b_hh
            h = 0.9 * h_prev + 0.1 * (gx_h + gh_h)

        # 浓度门控
        if cfg.use_concentration and self.conc_mlp is not None:
            pooled = h_prev.mean(dim=1)              # (B, d)
            c = self.conc_mlp(pooled)                # (B, G)
            c = c.unsqueeze(-1)                      # (B, G, 1)
            h_new = (1 - c) * h_prev + c * h
        else:
            h_new = h

        # 输出表示
        h_cat = h_new.reshape(B, G * d)              # (B, G*d)
        out = self.out_proj(h_cat)                   # (B, d)
        out = self.norm1(out)
        out2 = self.ffn(out)
        out = self.norm2(out + self.dropout(out2))
        return out, h_new

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """按序列顺序执行。

        Args:
            x: (B, L, d) 输入序列
            h0: (B, G, d) 初始状态，默认全 0
        Returns:
            outs: (B, L, d) 每步输出
            h_final: (B, G, d) 最终状态
        """
        B, L, d = x.shape
        device = x.device
        if h0 is None:
            h = torch.zeros(B, self.G, d, device=device, dtype=x.dtype)
        else:
            h = h0

        outs = []
        for t in range(L):
            out, h = self.step(x[:, t], h)
            outs.append(out)
        outs = torch.stack(outs, dim=1)              # (B, L, d)
        return outs, h


class NSMByteLM(nn.Module):
    """完整字节级 NSM 语言模型，带多任务输出头。"""

    def __init__(self, cfg: NSMConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([NSMCell(cfg) for _ in range(cfg.n_layers)])

        # 任务头
        self.next_byte_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.state_head = nn.Linear(cfg.d_model, cfg.max_states, bias=False)
        self.verdict_head = nn.Linear(cfg.d_model, 2, bias=False)  # ACCEPT/REJECT

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.next_byte_head.weight = self.embedding.weight  # tied weights

    def forward(self, input_ids: torch.Tensor, h0: Optional[List[torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Args:
            input_ids: (B, L) byte ids
            h0: 可选的各层初始状态列表
        Returns:
            byte_logits: (B, L, 256)
            state_logits: (B, L, max_states)
            verdict_logits: (B, L, 2)
            h_finals: 各层最终状态列表
        """
        x = self.embedding(input_ids)                 # (B, L, d)
        h_finals = []
        for i, layer in enumerate(self.layers):
            h0_i = h0[i] if h0 is not None else None
            x, h_f = layer(x, h0_i)
            h_finals.append(h_f)

        byte_logits = self.next_byte_head(x)
        state_logits = self.state_head(x)
        verdict_logits = self.verdict_head(x)
        return byte_logits, state_logits, verdict_logits, h_finals

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class GRULM(nn.Module):
    """简单 GRU 字节语言模型：用 PyTorch 优化过的 GRU 提供串行深度。"""

    def __init__(self, cfg: NSMConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.gru = nn.GRU(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.next_byte_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.state_head = nn.Linear(cfg.d_model, cfg.max_states, bias=False)
        self.verdict_head = nn.Linear(cfg.d_model, 2, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.next_byte_head.weight = self.embedding.weight
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
                with torch.no_grad():
                    p *= 0.95
            elif 'bias' in name:
                nn.init.zeros_(p)

    def forward(self, input_ids, h0=None):
        x = self.embedding(input_ids)
        h, h_n = self.gru(x, h0)
        h = self.norm(h)
        return self.next_byte_head(h), self.state_head(h), self.verdict_head(h), h_n

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

