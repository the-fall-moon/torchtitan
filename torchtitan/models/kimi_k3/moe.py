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


try:
    import cann_ops_nn  # noqa: F401  (ops-nn situ_glu; absent outside Ascend)

    _CANN_OPS_NN_AVAILABLE = True
except ImportError:
    _CANN_OPS_NN_AVAILABLE = False


class KimiSituGLU(torch.autograd.Function):
    """ops-nn AscendC SiTU-GLU behind an explicit autograd.Function.

    The fused kernel takes the concatenated [R, 2F] input (gate on the left)
    and computes beta * tanh(gate / beta) * sigmoid(gate) * up in one pass,
    ~2.4-4x faster than the eager fp32 composition at kimi_k3 shapes (see
    the migration package's FUSED_OPS_PERF_LOG.md section 3.5). The wrapper
    follows MindSpeed-MM's SituGLUFunction calling convention.
    """

    @staticmethod
    def forward(ctx, x, dim, beta, linear_beta, activate_left):
        ctx.dim = dim
        ctx.beta = beta
        ctx.linear_beta = linear_beta
        ctx.activate_left = activate_left
        ctx.save_for_backward(x)
        return cann_ops_nn.situ_glu(
            x,
            dim=dim,
            beta=beta,
            linear_beta=linear_beta,
            activate_left=activate_left,
        )

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = cann_ops_nn.situ_glu_grad(
            grad_output,
            x,
            dim=ctx.dim,
            beta=ctx.beta,
            linear_beta=ctx.linear_beta,
            activate_left=ctx.activate_left,
        )
        return grad_x, None, None, None, None


def _situ_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """Kimi's SiTU-GLU activation, evaluated in FP32."""
    if _CANN_OPS_NN_AVAILABLE and gate.device.type == "npu":
        x = torch.cat((gate, up), dim=-1)
        return KimiSituGLU.apply(x, -1, beta, linear_beta or 0.0, True)
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


class KimiNPUGMM(torch.autograd.Function):
    """npu_grouped_matmul behind an explicit autograd.Function.

    A bare torch.ops.npu.npu_grouped_matmul call silently drops gradients:
    the op registers only an NPU forward kernel (no Autograd key), yet its
    outputs still report requires_grad=True, so backward "succeeds" while
    the inputs receive no gradients at all. Wrapping the op in an explicit
    Function with a hand-written backward makes the gradient flow real (and
    replay-stable under torch.utils.checkpoint, unlike autograd's fallback
    machinery which fails with "aten.lift_fresh ... not found in storage").

    The calling convention follows MindSpeed-MM's GmmFunction (the fastest
    layout measured on CANN 9.2.0, see the migration package's
    FUSED_OPS_PERF_LOG.md section 3.1): a single [E, K, N] weight tensor
    with a device-side int64 per-expert token-count tensor
    (``group_list_type=1``). ``split_item=2`` returns one pre-concatenated
    [R, N] output (no torch.cat), and the backward runs as two more calls --
    dgrad with the transposed weight, wgrad as a split-K grouped matmul
    (``split_item=3, group_type=2``) producing [E, K, N] directly. This
    avoids both the 2E-call python loop of the first implementation and the
    host int-array group_list, which aclnn caps at 128 experts (the released
    K3 has 896). ``offs`` (cumulative ends, int32) is converted to counts
    on device; no D2H sync is involved.
    """

    @staticmethod
    def forward(ctx, A, B_t, offs):
        import torch_npu  # noqa: F401  (lazy: absent on GPU/CPU-only boxes)

        counts_int64 = torch.diff(offs, prepend=offs.new_zeros(1)).to(torch.int64)
        out = torch_npu.npu_grouped_matmul(
            [A],
            [B_t],
            bias=None,
            group_list=counts_int64,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]
        ctx.save_for_backward(A, B_t)
        ctx.counts_int64 = counts_int64
        return out

    @staticmethod
    def backward(ctx, grad_out):
        import torch_npu  # noqa: F401  (kernel calls below need it)

        # out_i = A[b_i:e_i] @ B_t[i]  (B_t is [E, K, N])
        A, B_t = ctx.saved_tensors
        counts_int64 = ctx.counts_int64

        # dA = grad_out @ B_t[i].T per expert, concatenated: same split-M
        # call as the forward with the weight transposed in place.
        dA = torch_npu.npu_grouped_matmul(
            [grad_out],
            [B_t.transpose(-2, -1)],
            bias=None,
            group_list=counts_int64,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

        # dB_t[i] = A[b_i:e_i].T @ grad_out[b_i:e_i] as a split-K grouped
        # matmul: x laid out [K, R], weight [R, N], output [E, K, N].
        dB_t = torch_npu.npu_grouped_matmul(
            [A.t()],
            [grad_out],
            bias=None,
            group_list=counts_int64,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]
        return dA, dB_t, None


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
            return KimiNPUGMM.apply(A, B_t, offs)
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
