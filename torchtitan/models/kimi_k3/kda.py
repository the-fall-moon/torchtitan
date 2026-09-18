# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Delta Attention modules for Kimi K3."""

from dataclasses import dataclass

import torch
from fla.modules import ShortConvolution as _FLA_ShortConvolution
from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_causal_conv1d
from fla.modules.fused_norm_gate import rms_norm_gated as _fla_rms_norm_gated
from torch import nn

from torchtitan.models.common import Linear
from torchtitan.models.kimi_k3.npu_kda import npu_chunk_kda as _npu_chunk_kda
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.protocols.module import Module

# Shape suffixes:
# T = packed tokens, D = model dimension, H = heads,
# K = key head dimension, V = value head dimension, C = projection channels.


class KimiRMSNormGated(Module):
    """Per-head RMSNorm followed by a sigmoid output gate."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return _fla_rms_norm_gated(
            x,
            gate,
            self.weight,
            None,
            activation="sigmoid",
            eps=self.eps,
        )


class KimiShortConvolution(_FLA_ShortConvolution, Module):
    """KDA short causal convolution backed by FLA's fused kernel.

    Mirrors the released Kimi K3 HF model, which builds FLA's
    ``ShortConvolution`` per q/k/v projection. Inputs are the packed-token
    2D layout ``(T, D)``; the Triton kernel runs on NPU via fla's
    triton_ascend backend.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        kernel_size: int
        activation: str = "silu"

    def __init__(self, config: Config):
        super().__init__(
            hidden_size=config.hidden_size,
            kernel_size=config.kernel_size,
            activation=config.activation,
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        y_TD, _ = _fla_causal_conv1d(
            x=x_TD.unsqueeze(0),
            weight=self.weight.squeeze(1),
            activation=self.activation,
            backend=self.backend,
        )
        return y_TD.squeeze(0), None


class KimiKDAKernel(Module):
    """Stateless dispatch to FLA's chunked KDA kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        lower_bound: float | None = -5.0

    def __init__(self, config: Config):
        super().__init__()
        self.lower_bound = config.lower_bound
        if self.lower_bound is not None and not (-5.0 <= self.lower_bound < 0.0):
            raise ValueError("KDA lower_bound must be in the safe range [-5, 0).")

    def forward(
        self,
        q_BLHK: torch.Tensor,
        k_BLHK: torch.Tensor,
        v_BLHV: torch.Tensor,
        gate_BLHK: torch.Tensor,
        beta_BLH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
    ) -> torch.Tensor:
        out_BLHV, _ = _npu_chunk_kda(
            q_BLHK,
            k_BLHK,
            v_BLHV,
            gate_BLHK,
            beta_BLH,
            A_log=A_log_H,
            dt_bias=dt_bias_HK.reshape(-1),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self.lower_bound is not None,
            lower_bound=self.lower_bound,
        )
        return out_BLHV


class KimiDeltaAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int
        head_dim: int
        conv_kernel_size: int
        q_proj: Linear.Config
        k_proj: Linear.Config
        v_proj: Linear.Config
        q_conv: KimiShortConvolution.Config
        k_conv: KimiShortConvolution.Config
        v_conv: KimiShortConvolution.Config
        forget_a: Linear.Config
        forget_b: Linear.Config
        beta: Linear.Config
        output_gate: Linear.Config
        kernel: Module.Config
        output_norm: KimiRMSNormGated.Config
        output_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.conv_kernel_size = config.conv_kernel_size

        self.q_proj = config.q_proj.build()
        self.k_proj = config.k_proj.build()
        self.v_proj = config.v_proj.build()
        self.q_conv = config.q_conv.build()
        self.k_conv = config.k_conv.build()
        self.v_conv = config.v_conv.build()
        self.forget_a = config.forget_a.build()
        self.forget_b = config.forget_b.build()
        self.beta = config.beta.build()
        self.output_gate = config.output_gate.build()
        self.kernel = config.kernel.build()
        self.output_norm = config.output_norm.build()
        self.output_proj = config.output_proj.build()

        self.A_log = nn.Parameter(torch.empty(config.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(config.num_heads, config.head_dim))

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if attention_masks is not None:
            raise NotImplementedError(
                "Kimi K3 reference KDA does not support packed-document masks."
            )

        num_tokens = x_TD.shape[0]
        q_THK = self.q_conv(self.q_proj(x_TD))[0].view(
            num_tokens, self.num_heads, self.head_dim
        )
        k_THK = self.k_conv(self.k_proj(x_TD))[0].view(
            num_tokens, self.num_heads, self.head_dim
        )
        v_THV = self.v_conv(self.v_proj(x_TD))[0].view(
            num_tokens, self.num_heads, self.head_dim
        )
        forget_THK = self.forget_b(self.forget_a(x_TD)).view(
            num_tokens, self.num_heads, self.head_dim
        )
        beta_TH = self.beta(x_TD).float()

        out_THV = self.kernel(
            q_THK.unsqueeze(0),
            k_THK.unsqueeze(0),
            v_THV.unsqueeze(0),
            forget_THK.unsqueeze(0),
            beta_TH.unsqueeze(0),
            self.A_log,
            self.dt_bias,
        ).squeeze(0)
        output_gate_THV = self.output_gate(x_TD).view(
            num_tokens, self.num_heads, self.head_dim
        )
        out_THV = self.output_norm(out_THV, output_gate_THV)
        return self.output_proj(out_THV.reshape(num_tokens, -1))
