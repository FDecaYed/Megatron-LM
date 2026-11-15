# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU checks of the producer/layout contract, independent of CUDA collectives."""

from types import SimpleNamespace

import pytest
import torch

import megatron.core.context_parallel.layout as layout_module
from megatron.core.context_parallel import (
    ContextParallelBatch,
    ContextParallelLayoutManager,
    convert_cp_partition_mode,
    get_batches_on_this_cp_rank,
)
from megatron.core.context_parallel.layout import _build_thd_zigzag_metadata
from megatron.core.packed_seq_params import PackedSeqParams


@pytest.mark.parametrize("boundary_mode,conversion_count", [("auto", 2), ("fixed", 4)])
def test_layout_policy_keeps_metadata_and_residual_order(
    monkeypatch, boundary_mode, conversion_count
):
    """An invertible permutation models communication, not a second route implementation."""
    calls = []

    def convert(value, source, target, *args):
        calls.append((source, target))
        return value.flip(0)

    monkeypatch.setattr(layout_module, "convert_cp_layout", convert)
    manager = ContextParallelLayoutManager(
        layer_layouts=("zigzag", "zigzag", "contiguous"),
        boundary_layout="contiguous",
        sequence_parallel=False,
        cp_group=SimpleNamespace(size=lambda: 2),
        tp_group=None,
        tp_cp_group=None,
        boundary_mode=boundary_mode,
    )
    metadata = {
        mode: PackedSeqParams(qkv_format="thd", cp_partition_mode=mode)
        for mode in ("zigzag", "contiguous")
    }
    state = manager.build_forward_state(metadata["contiguous"], metadata, thd_plan=object())
    original = torch.arange(4.0, requires_grad=True)
    value = original
    for i, mode in enumerate(manager.layer_layouts):
        value, params = state.prepare_layer(i, value)
        assert params is metadata[mode]
        torch.testing.assert_close(value, original.flip(0) if mode == "zigzag" else original)
        value = state.finalize_layer(i, value)
        if boundary_mode == "fixed":
            torch.testing.assert_close(value, original)
    torch.testing.assert_close(value, original)
    value.square().sum().backward()
    torch.testing.assert_close(original.grad, 2 * original.detach())
    assert len(calls) == conversion_count


def test_batch_wrapper_does_not_mutate_shared_metadata():
    params = PackedSeqParams(qkv_format="thd")
    batch = ContextParallelBatch.from_single_layout("contiguous", {}, params)
    assert params.cp_partition_mode is None
    assert batch.get_packed_seq_params().cp_partition_mode == "contiguous"
    with pytest.raises(ValueError, match="does not match"):
        ContextParallelBatch.from_single_layout("zigzag", {}, batch.get_packed_seq_params())


def test_packed_producer_builds_distinct_physical_metadata(monkeypatch):
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: group.size())
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: group.rank())
    # Includes an empty document, plus two documents requiring distinct attention padding.
    logical = torch.tensor([[0, 0, 3, 8]], dtype=torch.int32)
    batch = get_batches_on_this_cp_rank(
        {
            "tokens": torch.arange(8).view(1, 8),
            "cu_seqlens": logical,
            "max_seqlen": torch.tensor([5], dtype=torch.int32),
        },
        boundary_layout="contiguous",
        dynamic_cp=False,
        cp_group=group,
        additional_layouts=("zigzag",),
    )
    contiguous = batch.get_packed_seq_params("contiguous")
    zigzag = batch.get_packed_seq_params("zigzag")
    assert contiguous.cp_partition_mode == "contiguous"
    assert zigzag.cp_partition_mode == "zigzag"
    assert contiguous is not zigzag
    assert batch.thd_plan.contiguous_local_token_count == 4
    assert batch.thd_plan.zigzag_local_token_count == 6
    torch.testing.assert_close(zigzag.cu_seqlens_q, logical.squeeze(0))
    assert zigzag.cu_seqlens_q_padded.tolist() == [0, 0, 4, 12]
    assert batch.get_batch("contiguous")["tokens"].tolist() == [[0, 1, 2, 3]]
    assert batch.get_batch("zigzag")["tokens"].tolist() == [[0, 0, 3, 4, 0, 0]]


@pytest.mark.parametrize(
    "logical,physical", [([1, 4], None), ([0, 5, 4], None), ([0, 3, 8], [0, 2, 8])]
)
def test_invalid_packed_boundaries_fail_before_route_allocation(logical, physical):
    with pytest.raises(ValueError, match="Packed boundaries"):
        _build_thd_zigzag_metadata(
            torch.tensor(logical, dtype=torch.int32),
            None if physical is None else torch.tensor(physical, dtype=torch.int32),
            2,
            1,
        )


def test_partition_policy_is_not_a_physical_layout():
    with pytest.raises(ValueError, match="concrete"):
        convert_cp_partition_mode(
            torch.zeros(2), source_partition_mode="auto", target_partition_mode="auto"
        )


@pytest.mark.parametrize("packed", [False, True])
def test_partition_spelling_reaches_real_batch_dispatch(monkeypatch, packed):
    from megatron.core.utils import get_batch_on_this_cp_rank

    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: group.size())
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: group.rank())
    batch = {"tokens": torch.arange(8).view(1, 8)}
    if packed:
        batch["cu_seqlens"] = torch.tensor([[0, 4, 8]], dtype=torch.int32)
    result = get_batch_on_this_cp_rank(
        batch, dynamic_cp=False, cp_group=group, cp_partition_mode="contiguous"
    )
    assert result["tokens"].tolist() == [[0, 1, 2, 3]]
    with pytest.raises(ValueError, match="conflicts"):
        get_batch_on_this_cp_rank(
            {}, dynamic_cp=False, cp_group=group, use_contiguous_cp=True, cp_partition_mode="zigzag"
        )
