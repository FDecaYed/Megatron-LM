# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused tests for the shared-kernel SBHD CSA integration."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import csa as csa_module
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels, dsa_kernels
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttention,
    unfused_compressed_sparse_attn,
)


class _LayoutTensor:
    """Tensor-like object for layout-only eligibility checks."""

    def __init__(self, shape, dtype, *, device="cuda:0", is_cuda=True):
        self.shape = torch.Size(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = is_cuda

    def size(self, dim):
        return self.shape[dim]


def test_shared_topk_metadata_compacts_stably_and_tracks_empty_rows():
    topk = torch.tensor([[[3, -1, 1, 2], [-1, -1, -1, -1], [4, 0, -1, 2]]], dtype=torch.int64)

    compacted, lengths = dsa_cudnn_kernels._compact_valid_topk_indices(topk)

    assert compacted.tolist() == [[[3, 1, 2, -1], [-1, -1, -1, -1], [4, 0, 2, -1]]]
    assert lengths.dtype == torch.int32
    assert lengths.tolist() == [[3, 0, 3]]

    prepared, prepared_lengths = dsa_cudnn_kernels._prepare_attention_topk_indices(topk, sk=5)
    assert prepared.dtype == torch.int32
    assert prepared.tolist() == [[[1, 2, 3, -1], [-1, -1, -1, -1], [0, 2, 4, -1]]]
    assert torch.equal(prepared_lengths, lengths)


def test_shared_sparse_attention_eligibility_is_explicit(monkeypatch):
    query = _LayoutTensor((8, 1, 64, 512), torch.bfloat16)
    key = _LayoutTensor((10, 1, 1, 512), torch.bfloat16)
    topk = _LayoutTensor((1, 8, 20), torch.int32)
    sink = _LayoutTensor((64,), torch.float32)
    monkeypatch.setattr(dsa_cudnn_kernels, "_flash_mla_supports_head_count", lambda _query: True)

    assert dsa_cudnn_kernels.supports_fused_absorbed_sparse_attention(
        query, key, topk, 512, attn_sink=sink
    )
    assert not dsa_cudnn_kernels.supports_fused_absorbed_sparse_attention(
        query, key, topk, 128, attn_sink=sink
    )
    assert not dsa_cudnn_kernels.supports_fused_absorbed_sparse_attention(
        query, _LayoutTensor((10, 1, 1, 128), torch.bfloat16), topk, 512, attn_sink=sink
    )
    assert not dsa_cudnn_kernels.supports_fused_absorbed_sparse_attention(
        query,
        key,
        _LayoutTensor((1, 8, 20), torch.int32, is_cuda=False, device="cpu"),
        512,
        attn_sink=sink,
    )

    monkeypatch.setattr(dsa_cudnn_kernels, "_flash_mla_supports_head_count", lambda _query: False)
    assert not dsa_cudnn_kernels.supports_fused_absorbed_sparse_attention(
        query, key, topk, 512, attn_sink=sink
    )


@pytest.mark.parametrize("value_dim", [16, 64, 128])
def test_csa_toy_value_dims_fall_back_before_backend_dispatch(value_dim):
    csa = SimpleNamespace(
        use_fused_kernels=True,
        v_head_dim=value_dim,
        config=SimpleNamespace(),
        attn_sink=torch.zeros(2, dtype=torch.float32),
        softmax_scale=value_dim**-0.5,
    )
    query = torch.zeros(4, 1, 2, value_dim, dtype=torch.bfloat16)
    kv_full = torch.zeros(4, 1, value_dim, dtype=torch.bfloat16)
    topk = torch.zeros(1, 4, 1, dtype=torch.int32)

    with patch.object(
        csa_module.dsa_kernels,
        "supports_fused_absorbed_sparse_attention",
        side_effect=AssertionError("toy layouts must not reach the fused backend"),
    ):
        output = CompressedSparseAttention._try_fused_sparse_attention(csa, query, kv_full, topk)

    assert output is None


def test_csa_shared_dispatch_threads_the_learnable_sink():
    query = Mock(ndim=4, is_cuda=True, dtype=torch.bfloat16)
    query.size.side_effect = lambda dim: (2, 1, 64, 512)[dim]
    kv_full = Mock(ndim=3, is_cuda=True, dtype=torch.bfloat16)
    kv_full.size.side_effect = lambda dim: (3, 1, 512)[dim]
    key = Mock()
    kv_full.unsqueeze.return_value = key
    topk = Mock(ndim=3, is_cuda=True)
    sink = Mock()
    sink_fp32 = Mock()
    sink.float.return_value = sink_fp32
    expected = object()
    csa = SimpleNamespace(
        use_fused_kernels=True,
        v_head_dim=512,
        config=object(),
        attn_sink=sink,
        softmax_scale=512**-0.5,
    )

    with (
        patch.object(
            csa_module.dsa_kernels, "supports_fused_absorbed_sparse_attention", return_value=True
        ) as supports,
        patch.object(
            csa_module.dsa_kernels, "run_fused_absorbed_sparse_attention", return_value=expected
        ) as run,
    ):
        actual = CompressedSparseAttention._try_fused_sparse_attention(csa, query, kv_full, topk)

    assert actual is expected
    supports.assert_called_once()
    assert supports.call_args.kwargs["attn_sink"] is sink_fp32
    assert run.call_args.kwargs["attn_sink"] is sink_fp32


def test_backend_neutral_sparse_hook_dispatches_sink_and_eligibility(monkeypatch):
    sink = torch.zeros(1, dtype=torch.float32)
    expected = torch.ones(1)
    seen = {}

    def supports(*args, **kwargs):
        seen["supports"] = (args, kwargs)
        return True

    def run(*args, **kwargs):
        seen["run"] = (args, kwargs)
        return expected

    backend = SimpleNamespace(
        supports_fused_absorbed_sparse_attention=supports, run_fused_absorbed_sparse_attention=run
    )
    monkeypatch.setattr(dsa_kernels, "_load_backend", lambda _config: backend)
    config = SimpleNamespace(dsa_kernel_backend="cudnn")
    query = torch.zeros(1, 1, 1, 1)
    key = torch.zeros(1, 1, 1, 1)
    topk = torch.zeros(1, 1, 1, dtype=torch.int32)

    assert dsa_kernels.supports_fused_absorbed_sparse_attention(
        config, query, key, topk, 512, attn_sink=sink
    )
    actual = dsa_kernels.run_fused_absorbed_sparse_attention(
        config, query, key, topk, 1.0, 512, attn_sink=sink
    )

    assert actual is expected
    assert seen["supports"][1]["attn_sink"] is sink
    assert seen["run"][1]["attn_sink"] is sink


def test_shared_sparse_backward_returns_sink_gradient(monkeypatch):
    num_rows, num_heads, qk_dim, value_dim = 2, 3, 4, 5
    q_flat = torch.zeros(num_rows, num_heads, qk_dim)
    kv_flat = torch.zeros(4, qk_dim)
    out_flat = torch.zeros(num_rows, num_heads, value_dim)
    sink = torch.zeros(num_heads)
    expected_sink_grad = torch.arange(num_heads, dtype=torch.float32)

    class FakeDSA:
        @staticmethod
        def sparse_attention_backward_wrapper(*args, **kwargs):
            del kwargs
            return {
                "dq": torch.ones_like(args[0]),
                "dkv": torch.ones_like(args[1]),
                "d_sink": expected_sink_grad,
            }

    monkeypatch.setattr(dsa_cudnn_kernels, "_cudnn_dsa", FakeDSA())
    grad_query, grad_kv, grad_sink = dsa_cudnn_kernels._run_sparse_attention_backward(
        q_flat=q_flat,
        kv_flat=kv_flat,
        attn_sink=sink,
        global_idxs=torch.zeros(num_rows, 1, dtype=torch.int32),
        out_flat=out_flat,
        lse=torch.zeros(num_rows, num_heads),
        topk_length=torch.ones(num_rows, dtype=torch.int32),
        softmax_scale=0.5,
        sq=num_rows,
        b=1,
        num_heads=num_heads,
        d=qk_dim,
        skv=4,
        grad_output=torch.ones(num_rows, 1, num_heads, value_dim),
        all_rows_nonempty=True,
    )

    assert grad_query.shape == (num_rows, 1, num_heads, qk_dim)
    assert grad_kv.shape == (4, 1, qk_dim)
    assert torch.equal(grad_sink, expected_sink_grad)


def test_inference_topk_uses_compressed_causal_bounds_and_native_fallback():
    q = torch.ones(8, 1, 2, 4, dtype=torch.bfloat16)
    k = torch.ones(2, 1, 4, dtype=torch.bfloat16)
    weights = torch.ones(8, 1, 2, dtype=torch.bfloat16)
    positions = torch.arange(1, 9).unsqueeze(1)
    causal_mask = torch.zeros(1, 8, 2)
    fused_indices = torch.zeros(1, 8, 2, dtype=torch.int32)
    native_indices = torch.ones(1, 8, 2, dtype=torch.int64)
    csa = SimpleNamespace(
        config=object(),
        compress_ratio=4,
        indexer=SimpleNamespace(index_topk=8, softmax_scale=0.25),
        _can_use_fused_indexer_topk=lambda *_args: True,
    )

    with patch.object(
        csa_module.dsa_kernels,
        "run_fused_qk_topk",
        return_value=(fused_indices, torch.ones(1, 8, dtype=torch.int32)),
    ) as run:
        actual = CompressedSparseAttention._select_inference_compressed_topk(
            csa, q, k, weights, causal_mask, positions
        )

    assert actual is fused_indices
    assert run.call_args.args[5].tolist() == [0] * 8
    assert run.call_args.args[6].tolist() == [0, 0, 0, 1, 1, 1, 1, 2]

    with (
        patch.object(csa_module.dsa_kernels, "run_fused_qk_topk", return_value=None),
        patch.object(
            csa_module, "fused_qk_topk_naive", return_value=(torch.empty(0), native_indices)
        ) as native,
    ):
        actual = CompressedSparseAttention._select_inference_compressed_topk(
            csa, q, k, weights, causal_mask, positions
        )

    assert actual is native_indices
    torch.testing.assert_close(native.call_args.args[2], weights * 0.25)
    assert native.call_args.kwargs["mask"] is causal_mask


def test_ratio4_training_keeps_native_full_denominator_loss_before_fused_attention():
    sq, batch, heads, dim = 4, 1, 2, 16
    query = torch.randn(sq, batch, heads, dim)
    key = torch.randn(sq, batch, 1, dim)
    compressed_kv = torch.randn(1, batch, dim)
    q_indexer = torch.randn(sq, batch, 2, 4)
    k_indexer = torch.randn(1, batch, 4)
    weights_indexer = torch.randn(sq, batch, 2)
    selected = torch.zeros(batch, sq, 1, dtype=torch.int64)
    indexer_loss = torch.tensor(0.25)
    non_compressed_lse = torch.randn(batch, heads, sq)
    fused_output = torch.randn(sq, batch, heads, dim)
    final_output = fused_output.reshape(sq, batch, -1)
    indexer = SimpleNamespace(
        index_topk=8,
        softmax_scale=0.5,
        pg_collection=object(),
        forward_before_topk=Mock(return_value=(q_indexer, k_indexer, weights_indexer)),
    )
    csa = SimpleNamespace(
        compressor=Mock(return_value=compressed_kv),
        compress_ratio=4,
        window_size=2,
        indexer=indexer,
        training=True,
        config=SimpleNamespace(
            dsa_indexer_loss_coeff=0.1,
            dsa_indexer_use_sparse_loss=False,
            calculate_per_token_loss=False,
            num_layers=1,
            mtp_num_layers=0,
        ),
        layer_number=1,
        attn_sink=torch.zeros(heads, dtype=torch.float32),
        softmax_scale=dim**-0.5,
        _try_fused_sparse_attention=Mock(return_value=fused_output),
    )

    with (
        patch.object(csa_module, "nvtx_range_push"),
        patch.object(csa_module, "nvtx_range_pop"),
        patch.object(
            csa_module, "_compute_unfused_csa_non_compressed_lse", return_value=non_compressed_lse
        ),
        patch.object(
            csa_module.FusedDSAIndexerLoss, "apply", return_value=(selected, indexer_loss)
        ) as native_loss,
        patch.object(csa_module.DSAIndexerLossLoggingHelper, "save_loss_to_tracker"),
        patch.object(
            csa_module.DSAIndexerLossAutoScaler, "apply", return_value=final_output
        ) as attach_loss,
    ):
        actual = CompressedSparseAttention.forward(
            csa,
            query,
            key,
            value=key,
            attention_mask=None,
            x=torch.randn(sq, batch, 8),
            qr=torch.randn(sq, batch, 4),
        )

    assert actual is final_output
    native_loss.assert_called_once()
    assert native_loss.call_args.args[16] is True
    assert native_loss.call_args.args[17] is non_compressed_lse
    csa._try_fused_sparse_attention.assert_called_once()
    attach_loss.assert_called_once()
    torch.testing.assert_close(attach_loss.call_args.args[0], final_output)
    assert attach_loss.call_args.args[1] is indexer_loss


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_real_shared_fused_csa_matches_native_forward_backward_and_loss():
    """Compare shared FlashMLA/cuDNN attention with the native CSA reference."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("fused CSA requires SM90+")
    try:
        from cudnn import DSA  # noqa: F401
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError:
        pytest.skip("fused CSA dependencies are unavailable")

    torch.manual_seed(1234)
    sq, sk, batch, heads, dim = 8, 8, 1, 64, 512
    scale = dim**-0.5
    topk = torch.full((batch, sq, sk), -1, dtype=torch.int32, device="cuda")
    for row in range(1, sq):
        topk[:, row, : row + 1] = torch.arange(row + 1, device="cuda", dtype=torch.int32)

    fused_query = (
        torch.randn(sq, batch, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.02
    ).requires_grad_()
    fused_kv = (
        torch.randn(sk, batch, dim, device="cuda", dtype=torch.bfloat16) * 0.02
    ).requires_grad_()
    fused_sink = torch.zeros(heads, device="cuda", dtype=torch.float32, requires_grad=True)
    native_query = fused_query.detach().clone().requires_grad_()
    native_kv = fused_kv.detach().clone().requires_grad_()
    native_sink = fused_sink.detach().clone().requires_grad_()
    grad = torch.randn(sq, batch, heads, dim, device="cuda", dtype=torch.float32)

    fused_output = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention(
        fused_query, fused_kv.unsqueeze(2), topk, scale, dim, attn_sink=fused_sink
    )
    assert fused_output is not None
    native_output = unfused_compressed_sparse_attn(
        native_query, native_kv, native_sink, topk, scale
    ).reshape(sq, batch, heads, dim)

    torch.testing.assert_close(fused_output.float(), native_output.float(), rtol=5e-2, atol=5e-2)
    fused_loss = (fused_output.float() * grad).sum()
    native_loss = (native_output.float() * grad).sum()
    torch.testing.assert_close(fused_loss, native_loss, rtol=5e-2, atol=5e-2)
    fused_loss.backward()
    native_loss.backward()

    torch.testing.assert_close(
        fused_query.grad.float(), native_query.grad.float(), rtol=1e-1, atol=5e-2
    )
    torch.testing.assert_close(fused_kv.grad.float(), native_kv.grad.float(), rtol=1e-1, atol=5e-2)
    torch.testing.assert_close(fused_sink.grad, native_sink.grad, rtol=1e-1, atol=5e-2)
