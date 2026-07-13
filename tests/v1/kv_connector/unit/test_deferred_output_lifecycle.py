# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler lifecycle of KV_DEFER_OUTPUT_KEY: the final EngineCoreOutput is
held until the worker reports finished_sending, then emitted with the inline
payload from finished_sending_payloads attached."""

import copy

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KV_DEFER_OUTPUT_KEY
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import FinishReason

from .utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test


def _make_deferring_scheduler():
    """Scheduler whose connector defers the final output on finish."""
    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    def request_finished(request, block_ids):
        return True, {"hidden_states_inline": True, KV_DEFER_OUTPUT_KEY: True}

    scheduler.connector.request_finished = request_finished
    scheduler.connector.request_finished_all_groups = request_finished
    return scheduler


def _finish_request(scheduler, request):
    """Run the request to completion (max_tokens=1) and pass the finished
    req id to the worker. Returns the outputs of the finishing step."""
    scheduler.add_request(request)

    # STEP 1: the request finishes generating.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output(reqs=[request])
    engine_core_outputs = scheduler.update_from_output(
        scheduler_output, model_runner_output
    )

    # STEP 2: finished req id is passed to the worker.
    scheduler_output = scheduler.schedule()
    assert request.request_id in scheduler_output.finished_req_ids
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    return engine_core_outputs


def _step_with_kv_output(scheduler, kv_connector_output):
    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = kv_connector_output
    return scheduler.update_from_output(scheduler_output, model_runner_output)


def test_output_held_until_finished_sending():
    scheduler = _make_deferring_scheduler()
    request = create_request(max_tokens=1)
    request_id = request.request_id
    payload = {"hidden_states": torch.randn(3, 2), "token_ids": torch.arange(3)}

    engine_core_outputs = _finish_request(scheduler, request)

    # The final output is held, not emitted, and blocks are not freed.
    assert not any(eco.outputs for eco in engine_core_outputs.values())
    assert request_id in scheduler.deferred_kv_outputs
    assert request_id in scheduler.requests

    # STEP 3: worker reports finished_sending with the inline payload.
    engine_core_outputs = _step_with_kv_output(
        scheduler,
        KVConnectorOutput(
            finished_sending={request_id},
            finished_sending_payloads={request_id: payload},
        ),
    )

    (output,) = engine_core_outputs[0].outputs
    assert output.request_id == request_id
    assert output.finish_reason == FinishReason.LENGTH
    # The defer key is stripped; the payload rides in kv_transfer_tensors.
    assert output.kv_transfer_params == {"hidden_states_inline": True}
    assert torch.equal(
        output.kv_transfer_tensors["hidden_states"], payload["hidden_states"]
    )
    assert torch.equal(output.kv_transfer_tensors["token_ids"], payload["token_ids"])

    assert_scheduler_empty(scheduler)


def test_payload_buffered_when_sent_before_finished_sending():
    """TP ranks can report payloads a step before the aggregated
    finished_sending fires; the scheduler must buffer them."""
    scheduler = _make_deferring_scheduler()
    request = create_request(max_tokens=1)
    request_id = request.request_id
    payload = {"hidden_states": torch.randn(3, 2)}

    _finish_request(scheduler, request)

    # STEP 3: payload arrives without finished_sending; output stays held.
    engine_core_outputs = _step_with_kv_output(
        scheduler,
        KVConnectorOutput(finished_sending_payloads={request_id: payload}),
    )
    assert not any(eco.outputs for eco in engine_core_outputs.values())
    assert request_id in scheduler.pending_kv_payloads
    assert request_id in scheduler.deferred_kv_outputs

    # STEP 4: finished_sending releases the output with the buffered payload.
    engine_core_outputs = _step_with_kv_output(
        scheduler, KVConnectorOutput(finished_sending={request_id})
    )
    (output,) = engine_core_outputs[0].outputs
    assert torch.equal(
        output.kv_transfer_tensors["hidden_states"], payload["hidden_states"]
    )

    assert_scheduler_empty(scheduler)


def test_payload_fragments_accumulate_across_steps():
    """Independently-completing sub-connectors can report different named
    tensors for the same request in separate steps. Each fragment must be
    merged into the request's payload, not overwrite the previous one."""
    scheduler = _make_deferring_scheduler()
    request = create_request(max_tokens=1)
    request_id = request.request_id
    frag_a = torch.randn(3, 2)
    frag_b = torch.arange(3)

    _finish_request(scheduler, request)

    # STEP 3 & 4: two fragments arrive in separate steps, held meanwhile.
    _step_with_kv_output(
        scheduler,
        KVConnectorOutput(finished_sending_payloads={request_id: {"a": frag_a}}),
    )
    _step_with_kv_output(
        scheduler,
        KVConnectorOutput(finished_sending_payloads={request_id: {"b": frag_b}}),
    )
    assert set(scheduler.pending_kv_payloads[request_id]) == {"a", "b"}

    # STEP 5: finished_sending releases the output with both fragments.
    engine_core_outputs = _step_with_kv_output(
        scheduler, KVConnectorOutput(finished_sending={request_id})
    )
    (output,) = engine_core_outputs[0].outputs
    assert torch.equal(output.kv_transfer_tensors["a"], frag_a)
    assert torch.equal(output.kv_transfer_tensors["b"], frag_b)

    assert_scheduler_empty(scheduler)
