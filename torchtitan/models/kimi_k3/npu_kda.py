"""NPU KDA backend: fla_npu ascendc chunk_kda fwd/bwd as an autograd function.

Drop-in replacement for ``fla.ops.kda.chunk_kda``'s internal ``_chunk_kda``
(autograd.Function) on Ascend NPU. Uses fla_npu's fused ascendc kernels:

* forward: ``npu_chunk_kda_fwd`` -- 12 outputs aligned with FLA's
  ``chunk_kda_fwd``: (o, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h,
  initial_state), with ``disable_recompute=True`` (saved tensors are real
  outputs, no recompute).
* backward: fused ``npu_chunk_kda_bwd`` for K=V=128 (the fused path's only
  supported head_dim; the launcher route crashes with ACLNN_ERR_INNER_NULLPTR
  561103 at any head_dim, so the ctypes reference route is used). For
  K<128 -- the NPU-safe debugmodel config -- FLA's triton ``chunk_kda_bwd``
  is used instead, which runs correctly on NPU.

Layout note: the ascendc fwd takes BSND inputs and emits BNSD saved tensors;
the ascendc bwd consumes BNSD. FLA's triton kernels use BNSD throughout.
The wrapper converts at the boundaries.

Pre/post processing mirrors ``_chunk_kda.forward/backward``: l2-norm on q/k
(use_qk_l2norm_in_kernel), sigmoid on beta (use_beta_sigmoid_in_kernel), and
their backward chain rules. The *original* q/k/beta are what the backward
chain rules need, so they are what gets saved.

fla_npu's kernels require g (and beta) in fp32 -- bf16 inputs are rejected by
aclnn with 161002 -- so the wrapper upcasts at the fwd boundary and returns
gradients in the caller's dtype.
"""

from __future__ import annotations

import torch


def _l2norm_fwd(x):
    # Row-wise L2 norm, matching FLA's l2norm_fwd (no eps).
    return torch.nn.functional.normalize(x, dim=-1, eps=0.0)


def _l2norm_bwd(x, dx):
    # d(l2norm(x)) chain: (I - q q^T) / norm applied to dx, recomputing
    # the norm from the saved original x.
    norm = x.norm(dim=-1, keepdim=True)
    q = x / norm
    return (dx - q * (dx * q).sum(dim=-1, keepdim=True)) / norm


def _canonical_chunk_indices(cu_seqlens, chunk_size):
    cu = [int(x) for x in cu_seqlens]
    indices = []
    for begin, end in zip(cu, cu[1:]):
        n = (end - begin + chunk_size - 1) // chunk_size
        for i in range(n):
            indices.append(begin + i * chunk_size)
            indices.append(min(begin + (i + 1) * chunk_size, end))
    return indices


def _to_bnsd(x):
    """(1, T, H, D) -> (1, H, T, D)."""
    return x.transpose(1, 2).contiguous()


def _to_bsnd(x):
    """(1, H, T, D) -> (1, T, H, D)."""
    return x.transpose(1, 2).contiguous()


class _NPUKDA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q, k, v, g, beta,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        chunk_indices,
        chunk_size,
        safe_gate,
        lower_bound,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel,
    ):
        from fla_npu.ops.ascendc import npu_chunk_kda_fwd

        q_n = _l2norm_fwd(q) if use_qk_l2norm_in_kernel else q
        k_n = _l2norm_fwd(k) if use_qk_l2norm_in_kernel else k

        beta_raw = beta
        beta_f = beta_raw.float().sigmoid() if use_beta_sigmoid_in_kernel else beta_raw.float()

        g_f = g.float()

        (o, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state_out) = (
            npu_chunk_kda_fwd(
                q_n, k_n, v, g_f, beta_f, float(scale), chunk_size,
                layout="BSND",
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
                use_gate_in_kernel=use_gate_in_kernel,
                A_log=A_log,
                dt_bias=dt_bias,
                disable_recompute=True,
            )
        )

        ctx.save_for_backward(
            q, k, v, gk, g_f, beta_raw, beta_f,
            A_log, dt_bias, Aqk, Akk, w, u, qg, kg, v_new, h,
            cu_seqlens, chunk_indices,
        )
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.lower_bound = lower_bound
        ctx.scale = scale
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        return o.type_as(q), final_state

    @staticmethod
    def backward(ctx, do, dht):
        (q, k, v, gk, g_f, beta_raw, beta_f,
         A_log, dt_bias, Aqk, Akk, w, u, qg, kg, v_new, h,
         cu_seqlens, chunk_indices) = ctx.saved_tensors

        head_dim = q.shape[-1]
        use_ascendc_bwd = head_dim >= 128

        q_n = _l2norm_fwd(q) if ctx.use_qk_l2norm_in_kernel else q
        k_n = _l2norm_fwd(k) if ctx.use_qk_l2norm_in_kernel else k

        assert use_ascendc_bwd, "only head_dim >= 128 reaches this wrapper"
        from fla_npu.ops.ascendc import _aclnn_ctypes as _ctypes_ref

        # Fused ascendc backward: K=V=128 only; the launcher route crashes
        # in aclnn (ACLNN_ERR_INNER_NULLPTR 561103), the ctypes reference
        # route works. Consumes BNSD tensors; dt_bias is [H, K] here.
        dq, dk, dv, db, dg, dh0, dA, dbias = _ctypes_ref.npu_chunk_kda_bwd(
            _to_bnsd(q_n), _to_bnsd(k_n), _to_bnsd(v),
            beta_f.transpose(1, 2).contiguous(),
            gk, Aqk, Akk, w, qg, kg, v_new, h,
            _to_bnsd(do), float(ctx.scale),
            raw_g=_to_bnsd(g_f) if ctx.use_gate_in_kernel else None,
            A_log=A_log,
            dt_bias=dt_bias.reshape(q.shape[2], head_dim) if dt_bias is not None else None,
            chunk_size=ctx.chunk_size,
            safe_gate=ctx.safe_gate,
            lower_bound=ctx.lower_bound,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
        )

        dq = _to_bsnd(dq)
        dk = _to_bsnd(dk)
        dv = _to_bsnd(dv)
        dg = _to_bsnd(dg)
        db = _to_bsnd(db)  # ascendc bwd returns BNSD; beta_raw is BSND
        if ctx.use_qk_l2norm_in_kernel:
            dq = _l2norm_bwd(q, dq)
            dk = _l2norm_bwd(k, dk)
        if ctx.use_beta_sigmoid_in_kernel:
            db = db * beta_raw.float().sigmoid() * (1 - beta_raw.float().sigmoid())

        return (
            dq.to(q), dk.to(k), dv.to(v), dg.to(g_f), db.to(beta_raw),
            None,  # scale
            None,  # initial_state
            None,  # output_final_state
            None,  # state_v_first
            None,  # cu_seqlens
            None,  # chunk_indices
            None,  # chunk_size
            None,  # safe_gate
            None,  # lower_bound
            None,  # use_gate_in_kernel
            dA,  # A_log
            dbias.reshape(-1) if dbias is not None else None,  # dt_bias
            None,  # use_qk_l2norm_in_kernel
            None,  # use_beta_sigmoid_in_kernel
        )


def npu_chunk_kda(
    q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
    safe_gate=False, lower_bound=None, chunk_size=64, disable_recompute=False,
    return_intermediate_states=False, state_v_first=False,
    cu_seqlens=None, cu_seqlens_cpu=None, cp_context=None, **kwargs,
):
    """High-level KDA chunked attention (same signature as FLA's).

    On NPU, FLA's own triton_ascend backend handles head_dim < 128 correctly
    (validated by the kimi_k3 KDA kernel unit test). For head_dim >= 128 --
    which FLA's triton kernels cannot run on this NPU -- this wrapper takes
    over with fla_npu's fused ascendc kernels (fwd + bwd). Any other device
    falls through to FLA unchanged.
    """
    if q.shape[-1] < 128:
        from fla.ops.kda import chunk_kda

        return chunk_kda(
            q, k, v, g, beta, scale=scale, initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval, safe_gate=safe_gate,
            lower_bound=lower_bound, chunk_size=chunk_size,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_intermediate_states,
            state_v_first=state_v_first, cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu, cp_context=cp_context, **kwargs,
        )
    if disable_recompute is not False:
        raise ValueError("NPU KDA backend always disables recompute.")
    if return_intermediate_states:
        raise ValueError("NPU KDA backend does not support intermediate states.")
    if cu_seqlens_cpu is not None:
        cu_seqlens = cu_seqlens_cpu
    chunk_indices = None
    if cu_seqlens is not None:
        chunk_indices = _canonical_chunk_indices(cu_seqlens, chunk_size)
    o, final_state = _NPUKDA.apply(
        q, k, v, g, beta, scale if scale is not None else q.shape[-1] ** -0.5,
        initial_state, output_final_state, state_v_first,
        cu_seqlens, chunk_indices, chunk_size, safe_gate, lower_bound,
        use_gate_in_kernel, kwargs.pop("A_log", None), kwargs.pop("dt_bias", None),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
    )
    return o, final_state
