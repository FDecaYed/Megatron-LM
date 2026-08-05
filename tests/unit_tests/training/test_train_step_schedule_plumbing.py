# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for sequence-packing plumbing in the training loop."""

from types import SimpleNamespace
from unittest import mock

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.rerun_state_machine import RerunMode
from megatron.training import training as training_mod


class _Rerun:
    """Run the forward/backward body once, then ask train_step to exit before optimizer.step."""

    _ran = False

    def should_run_forward_backward(self, data_iterator):
        run, self._ran = not self._ran, True
        return run

    def should_checkpoint_and_exit(self):
        return False, True, 0  # (checkpoint, exit, code)


class _RerunTwice:
    """Run two forward/backward passes while recording the iterator identity."""

    def __init__(self):
        self.calls = 0
        self.data_iterators = []

    def should_run_forward_backward(self, data_iterator):
        self.data_iterators.append(data_iterator)
        self.calls += 1
        return self.calls <= 2

    def should_checkpoint_and_exit(self):
        return False, True, 0  # (checkpoint, exit, code)


def _run(**kwargs):
    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
    )
    captured = {}
    model = [SimpleNamespace(force_all_reduce=False, zero_grad_buffer=lambda: None)]
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=_Rerun()),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
    ):
        training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=SimpleNamespace(zero_grad=lambda: None),
            opt_param_scheduler=None,
            config=SimpleNamespace(),
            forward_backward_func=lambda **kw: captured.update(kw) or [],
            iteration=0,
            **kwargs,
        )
    return captured


def test_train_step_forwards_schedule_plumbing():
    p2p, pg = object(), object()
    captured = _run(p2p_communicator=p2p, pg_collection=pg)
    assert captured["p2p_communicator"] is p2p and captured["pg_collection"] is pg


def test_train_step_defaults_to_none():
    captured = _run()
    assert captured["p2p_communicator"] is None and captured["pg_collection"] is None


def test_training_log_uses_scheduled_microbatch_count_for_mtp_moe_and_dsa():
    args = SimpleNamespace(
        timing_log_level=0,
        perform_rl_step=False,
        micro_batch_size=1,
        data_parallel_size=1,
        world_size=1,
        seq_length=8,
        freeze_all_layers=False,
        num_experts=8,
        moe_router_load_balancing_type=["aux_loss"],
        moe_z_loss_coeff=None,
        num_layers=2,
        moe_per_layer_logging=False,
        moe_layer_freq=None,
        mtp_num_layers=1,
        dsa_indexer_loss_coeff=0.1,
        log_interval=2,
    )
    moe_tracker = mock.MagicMock()
    moe_tracker.report.return_value = ""

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_tensorboard_writer", return_value=None),
        mock.patch.object(training_mod, "get_wandb_writer", return_value=None),
        mock.patch.object(training_mod, "get_one_logger", return_value=None),
        mock.patch.object(training_mod, "get_energy_monitor", return_value=None),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=11),
        mock.patch.object(training_mod.one_logger_utils, "track_app_tag"),
        mock.patch.object(
            training_mod,
            "reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value, group=None: value,
        ),
        mock.patch.object(training_mod, "is_hybrid_model", return_value=False),
        mock.patch.object(training_mod, "get_moe_metrics_tracker", return_value=moe_tracker),
        mock.patch.object(
            training_mod.MTPLossLoggingHelper, "track_mtp_metrics"
        ) as track_mtp_metrics,
        mock.patch.object(
            training_mod.DSAIndexerLossLoggingHelper, "track_indexer_metrics"
        ) as track_indexer_metrics,
    ):
        training_mod.training_log(
            loss_dict={},
            total_loss_dict={},
            learning_rate=None,
            iteration=1,
            loss_scale=1.0,
            report_memory_flag=False,
            skipped_iter=0,
            grad_norm=None,
            params_norm=None,
            num_zeros_in_grad=None,
            max_attention_logit=None,
            num_microbatches=3,
        )

    assert moe_tracker.report.call_args.kwargs["loss_scale"] == 1 / 3
    assert track_mtp_metrics.call_args.args[0] == 1 / 3
    assert track_indexer_metrics.call_args.kwargs["loss_scale"] == 1 / 3


def test_train_step_rebuilds_schedule_from_source_iterator():
    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
    )
    source_data_iterator = object()
    scheduled_data_iterators = [object(), object()]
    scheduler_pg_collection = ProcessGroupCollection()
    model = [
        SimpleNamespace(
            force_all_reduce=False,
            pg_collection=scheduler_pg_collection,
            zero_grad_buffer=lambda: None,
        )
    ]
    rerun = _RerunTwice()
    forward_backward_func = mock.MagicMock(return_value=[])

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=rerun),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=4),
        mock.patch.object(training_mod, "get_moe_router_tracer", return_value=None),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(
            training_mod,
            "wrap_data_iterator",
            side_effect=[
                (scheduled_data_iterators[0], 2, 10, 20),
                (scheduled_data_iterators[1], 3, 30, 40),
            ],
        ) as wrap_data_iterator,
        mock.patch.object(training_mod, "set_seqlen_stats_in_iteration") as set_seqlen_stats,
    ):
        result = training_mod.train_step(
            forward_step_func=lambda *args, **kwargs: None,
            data_iterator=source_data_iterator,
            model=model,
            optimizer=SimpleNamespace(zero_grad=lambda: None),
            opt_param_scheduler=None,
            config=SimpleNamespace(sequence_packing_scheduler="dp_balanced"),
            forward_backward_func=forward_backward_func,
            iteration=0,
        )

    assert rerun.data_iterators == [source_data_iterator] * 3
    assert [call.args[0] for call in wrap_data_iterator.call_args_list] == [
        source_data_iterator,
        source_data_iterator,
    ]
    assert all(
        call.kwargs["pg_collection"] is scheduler_pg_collection
        for call in wrap_data_iterator.call_args_list
    )
    assert [call.kwargs["data_iterator"] for call in forward_backward_func.call_args_list] == (
        scheduled_data_iterators
    )
    assert [call.kwargs["num_microbatches"] for call in forward_backward_func.call_args_list] == [
        2,
        3,
    ]
    assert set_seqlen_stats.call_args_list == [mock.call(10, 20), mock.call(30, 40)]
    assert result[-1] == 3


def test_evaluate_restores_hook_and_timers_when_schedule_is_exhausted():
    args = SimpleNamespace(
        eval_global_batch_size=8,
        eval_micro_batch_size=2,
        data_parallel_size=2,
        cuda_graph_impl=None,
        moe_expert_rank_capacity_factor=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        eval_iters=1,
        empty_unused_memory_level=0,
        seq_length=8,
        decoder_seq_length=None,
        consumed_valid_samples=0,
        exit_duration_in_mins=None,
    )
    timers = mock.MagicMock()
    rerun = mock.MagicMock()
    original_rerun_mode = object()
    rerun.get_mode.return_value = original_rerun_mode
    scheduler_pg_collection = ProcessGroupCollection()
    model_module = SimpleNamespace(
        eval=mock.MagicMock(), train=mock.MagicMock(), pg_collection=scheduler_pg_collection
    )
    config = SimpleNamespace(sequence_packing_scheduler="dp_balanced", timers=timers)
    process_non_loss_data = mock.MagicMock()

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=timers),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=rerun),
        mock.patch.object(training_mod, "get_forward_backward_func") as get_forward_backward_func,
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "wrap_data_iterator", side_effect=StopIteration),
        mock.patch.object(training_mod.ft_integration, "on_eval_step_start") as on_eval_step_start,
        mock.patch.object(training_mod.ft_integration, "on_eval_step_end") as on_eval_step_end,
    ):
        result = training_mod.evaluate(
            forward_step_func=lambda *args, **kwargs: None,
            data_iterator=object(),
            model=[model_module],
            process_non_loss_data_func=process_non_loss_data,
            config=config,
        )

    assert result == ({}, None, False)
    on_eval_step_start.assert_called_once_with()
    on_eval_step_end.assert_called_once_with()
    assert config.timers is timers
    model_module.eval.assert_called_once_with()
    model_module.train.assert_called_once_with()
    get_forward_backward_func.return_value.assert_not_called()
    process_non_loss_data.assert_not_called()
    assert rerun.set_mode.call_args_list == [
        mock.call(RerunMode.DISABLED),
        mock.call(original_rerun_mode),
    ]
