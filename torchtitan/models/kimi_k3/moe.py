# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SiTU feed-forward and latent MoE modules for Kimi K3."""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor

from torchtitan.models.common import Linear
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe import GroupedExperts, MoE
from torchtitan.models.common.nn_modules import RMSNorm

# Shape suffixes:
# T = packed tokens, D = model dimension, E = experts,
# F = expert hidden dimension, R = routed tokens, K = selected experts per token.


def _situ_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """Kimi's SiTU-GLU activation, evaluated in FP32."""
    input_dtype = gate.dtype
    gate = gate.float()
    up = up.float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(input_dtype)


class KimiFeedForward(FeedForward):
    """FeedForward with Kimi's SiTU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        beta: float = 1.0
        linear_beta: float | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.beta = config.beta
        self.linear_beta = config.linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(
            _situ_glu(self.w1(x), self.w3(x), self.beta, self.linear_beta),
        )


class _NPUGMM(torch.autograd.Function):
    """npu_grouped_matmul behind an explicit autograd.Function.

    A bare torch.ops.npu.npu_grouped_matmul call records gradients through
    autograd's fallback machinery, which is not replay-stable under
    torch.utils.checkpoint on Ascend ("aten.lift_fresh ... not found in
    storage" during activation-checkpoint backward). Wrapping the op in an
    explicit Function with a hand-written backward makes the checkpoint
    replay well-defined: forward calls the kernel, backward splits the
    incoming gradient along the per-expert offsets (the reverse of the
    forward's concatenation).
    """

    @staticmethod
    def forward(ctx, A, B_t, offs):
        import torch_npu  # noqa: F401  (lazy: absent on GPU/CPU-only boxes)

        weights = [B_t[i] for i in range(B_t.shape[0])]
        outs = torch_npu.npu_grouped_matmul(
            [A],
            weights,
            group_list=offs.tolist(),
            group_type=0,
            split_item=0,
        )
        ctx.offs = offs
        ctx.save_for_backward(A, B_t)
        return torch.cat(outs, dim=0)

    @staticmethod
    def backward(ctx, grad_out):
        # out_i = A[offs_i:offs_{i+1}] @ B_t[i]  (B_t is [E, K, N])
        # dA = cat_i(grad_out_i @ B_t[i].T) along rows
        # dB_t[i] = A[offs_i:offs_{i+1}].T @ grad_out_i
        A, B_t = ctx.saved_tensors
        offs = [int(x) for x in ctx.offs.tolist()]
        dA_parts = []
        dB_t_parts = []
        begin = 0
        for i, end in enumerate(offs):
            block = grad_out[begin:end]  # [m_i, N]
            a_block = A[begin:end]  # [m_i, K]
            dA_parts.append(block @ B_t[i].T)
            dB_t_parts.append(a_block.T @ block)
            begin = end
        return torch.cat(dA_parts, dim=0), torch.stack(dB_t_parts, dim=0), None


_npu_gmm = _NPUGMM.apply


class KimiGroupedExperts(GroupedExperts):
    """``common/moe.py::GroupedExperts`` with Kimi's SiTU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        beta: float = 1.0
        linear_beta: float | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.beta = config.beta
        self.linear_beta = config.linear_beta

    def _grouped_mm(self, *, A, B_t, offs):
        """Grouped matmul of ``A @ B_t`` with per-expert token offsets.

        Uses torch_npu's ``npu_grouped_matmul`` (the "npu_gmm" kernel) on
        Ascend: torch._grouped_mm is CUDA-only. ``B_t`` is ``[E, K, N]`` --
        split per expert into the list the op expects; ``offs`` is the
        cumsum of per-expert token counts, passed as a host-side list.
        """
        if hasattr(torch, "npu") and torch.npu.is_available():
            return _npu_gmm(A, B_t, offs)
        return super()._grouped_mm(A=A, B_t=B_t, offs=offs)

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1_EFD, DTensor):
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)

        gate_RF = self._grouped_mm(
            A=x_RD.bfloat16(),
            B_t=w1_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )
        up_RF = self._grouped_mm(
            A=x_RD.bfloat16(),
            B_t=w3_EFD.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )

        h_RF = _situ_glu(gate_RF, up_RF, self.beta, self.linear_beta)

        return self._grouped_mm(
            A=h_RF,
            B_t=w2_EDF.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        ).type_as(x_RD)


class KimiLatentMoE(MoE):
    """``common/moe.py::MoE`` with Kimi's latent routed-expert path."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        routed_down: Linear.Config
        routed_norm: RMSNorm.Config
        routed_up: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.routed_down = config.routed_down.build()
        self.routed_norm = config.routed_norm.build()
        self.routed_up = config.routed_up.build()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        weights_TK, expert_ids_TK, scores_TE = self.router(x_TD, self.expert_bias_E)
        routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
            -1, expert_ids_TK, True
        )
        num_tokens_per_expert_E = routing_map_TE.sum(dim=0)
        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(num_tokens_per_expert_E)

        routed_TD = self.routed_experts(
            self.routed_down(x_TD),
            weights_TK,
            expert_ids_TK,
            num_tokens_per_expert_E,
        )
        out_TD = self.routed_up(self.routed_norm(routed_TD))
        if self.shared_experts is not None:
            out_TD = out_TD + self.shared_experts(x_TD)
        return out_TD
