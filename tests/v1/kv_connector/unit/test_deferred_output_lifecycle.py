# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler lifecycle of KV_DEFER_OUTPUT_KEY: the final EngineCoreOutput is
held until the worker reports the request's inline payload (or its
finished_sending), then emitted with the payload in kv_transfer_tensors."""

import copy
from collections import defaultdict
from typing import Any

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KV_DEFER_OUTPUT_KEY,
    KVConnectorBase_V1,
    KVConnectorMetadata,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import FinishReason, RequestStatus

from .utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

pytestmark = pytest.mark.cpu_test

INLINE_PARAMS = {"hidden_states_inline": True, KV_DEFER_OUTPUT_KEY: True}


class DeferringConnector(KVConnectorBase_V1):
    """Scheduler-side connector whose request_finished() asks the scheduler
    to hold the final output for an inline payload."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.delay_free_blocks: bool = extra.get("delay_free_blocks", True)
        self.finished_params: dict[str, Any] = extra.get(
            "finished_params", INLINE_PARAMS
        )

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def build_connector_meta(self, scheduler_output):
        return KVConnectorMetadata()

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (0, False)

    def update_state_after_alloc(self, request, blocks, num_tokens):
        pass

    def request_finished(self, request, block_ids):
        return self.delay_free_blocks, dict(self.finished_params)


KVConnectorFactory.register_connector(
    "DeferringConnector", __name__, DeferringConnector.__name__
)


def _make_deferring_scheduler(**extra_config):
    vllm_config = create_vllm_config(
        kv_connector="DeferringConnector",
        kv_connector_module_path=__name__,
        kv_role="kv_producer",
        kv_connector_extra_config=extra_config,
    )
    return create_scheduler(vllm_config)


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


def test_output_released_with_payload_and_finished_sending():
    scheduler = _make_deferring_scheduler()
    request = create_request(max_tokens=1)
    request_id = request.request_id
    payload = {"hidden_states": torch.randn(3, 2), "token_ids": torch.arange(3)}

    engine_core_outputs = _finish_request(scheduler, request)

    # The final output is held, not emitted, and blocks are not freed.
    assert not any(eco.outputs for eco in engine_core_outputs.values())
    assert request_id in scheduler.deferred_kv_outputs
    assert request_id in scheduler.requests

    # STEP 3: worker reports the payload together with finished_sending.
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


def test_payload_releases_output_before_finished_sending():
    """A connector (or a MultiConnector sibling still sending KV) may report
    the payload a step before finished_sending. The output must not wait for
    finished_sending, which may depend on a remote consumer reading this very
    response; only the blocks do."""
    scheduler = _make_deferring_scheduler()
    request = create_request(max_tokens=1)
    request_id = request.request_id
    payload = {"hidden_states": torch.randn(3, 2)}

    _finish_request(scheduler, request)

    # STEP 3: payload arrives without finished_sending; output is released.
    engine_core_outputs = _step_with_kv_output(
        scheduler,
        KVConnectorOutput(finished_sending_payloads={request_id: payload}),
    )
    (output,) = engine_core_outputs[0].outputs
    assert torch.equal(
        output.kv_transfer_tensors["hidden_states"], payload["hidden_states"]
    )
    assert request_id not in scheduler.deferred_kv_outputs
    # Blocks stay allocated until the connector is done with them.
    assert request_id in scheduler.requests

    # STEP 4: finished_sending frees the blocks and emits nothing more.
    engine_core_outputs = _step_with_kv_output(
        scheduler, KVConnectorOutput(finished_sending={request_id})
    )
    assert not any(eco.outputs for eco in engine_core_outputs.values())

    assert_scheduler_empty(scheduler)


def test_finished_sending_without_payload_releases_output():
    """A connector that reports finished_sending without ever producing a
    payload must still release the held output, not strand the client."""
    scheduler = _make_deferring_scheduler()
    request = create_request(max_tokens=1)
    request_id = request.request_id

    _finish_request(scheduler, request)

    engine_core_outputs = _step_with_kv_output(
        scheduler, KVConnectorOutput(finished_sending={request_id})
    )
    (output,) = engine_core_outputs[0].outputs
    assert output.request_id == request_id
    assert output.kv_transfer_tensors is None
    assert output.kv_transfer_params == {"hidden_states_inline": True}

    assert_scheduler_empty(scheduler)


def test_defer_key_ignored_when_blocks_are_freed():
    """Without delayed block freeing there is no finished_sending to release
    the output, so the defer key must not hold it."""
    scheduler = _make_deferring_scheduler(delay_free_blocks=False)
    request = create_request(max_tokens=1)

    engine_core_outputs = _finish_request(scheduler, request)

    (output,) = engine_core_outputs[0].outputs
    assert output.request_id == request.request_id
    assert output.kv_transfer_params == {"hidden_states_inline": True}
    assert not scheduler.deferred_kv_outputs

    assert_scheduler_empty(scheduler)


def test_defer_key_only_params_become_none():
    """A connector returning only the defer key must not leave an empty
    dict behind where other connectors return None."""
    scheduler = _make_deferring_scheduler(finished_params={KV_DEFER_OUTPUT_KEY: True})
    request = create_request(max_tokens=1)

    _finish_request(scheduler, request)
    engine_core_outputs = _step_with_kv_output(
        scheduler, KVConnectorOutput(finished_sending={request.request_id})
    )
    (output,) = engine_core_outputs[request.client_index].outputs
    assert output.kv_transfer_params is None
    assert_scheduler_empty(scheduler)


def test_abort_all_releases_held_output_without_payload():
    """finish_requests(None, ...) (engine fault, shutdown, pause-abort) must
    hand the held final output back for delivery, marked aborted, rather
    than turning a finished request into an empty abort."""
    scheduler = _make_deferring_scheduler()
    request = create_request(max_tokens=1)
    request_id = request.request_id

    _finish_request(scheduler, request)
    assert request_id in scheduler.deferred_kv_outputs

    aborted = scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)

    # Not reported as aborted: it finished generating.
    assert aborted == []
    assert request_id not in scheduler.deferred_kv_outputs
    ((client_index, held_output),) = scheduler.take_flushed_kv_outputs()
    assert client_index == request.client_index
    assert held_output.request_id == request_id
    # Tokens are kept, the status says the payload was dropped, and the
    # params that announced it go with it.
    assert held_output.new_token_ids
    assert held_output.finish_reason == FinishReason.ABORT
    assert held_output.kv_transfer_tensors is None
    assert held_output.kv_transfer_params is None
    assert scheduler.take_flushed_kv_outputs() == []

    # A late payload for the flushed request is dropped, not re-emitted.
    engine_core_outputs = _step_with_kv_output(
        scheduler,
        KVConnectorOutput(
            finished_sending={request_id},
            finished_sending_payloads={request_id: {"a": torch.zeros(1)}},
        ),
    )
    assert not any(eco.outputs for eco in engine_core_outputs.values())
    assert_scheduler_empty(scheduler)


def test_flushed_output_emitted_by_next_step_when_not_taken():
    """A client abort (finish_requests with ids) also releases the held
    output; if nobody drains it, the next step emits it."""
    scheduler = _make_deferring_scheduler()
    request = create_request(max_tokens=1)
    request_id = request.request_id

    _finish_request(scheduler, request)
    assert scheduler.finish_requests(request_id, RequestStatus.FINISHED_ABORTED) == []

    engine_core_outputs = _step_with_kv_output(
        scheduler, KVConnectorOutput(finished_sending={request_id})
    )
    (output,) = engine_core_outputs[request.client_index].outputs
    assert output.request_id == request_id
    assert output.finish_reason == FinishReason.ABORT
    assert output.kv_transfer_tensors is None
    assert output.kv_transfer_params is None
    assert_scheduler_empty(scheduler)


def test_finished_request_id_reported_with_released_output():
    """With a finished-request set (internal DP load balancing), the id of a
    held request must not be reported before its final output, or clients
    tear the stream down early."""
    scheduler = _make_deferring_scheduler()
    scheduler.finished_req_ids_dict = defaultdict(set)
    request = create_request(max_tokens=1)
    request_id = request.request_id
    payload = {"hidden_states": torch.randn(3, 2)}

    engine_core_outputs = _finish_request(scheduler, request)
    assert not any(eco.finished_requests for eco in engine_core_outputs.values()), (
        "finished id reported before the held output"
    )

    engine_core_outputs = _step_with_kv_output(
        scheduler,
        KVConnectorOutput(
            finished_sending={request_id},
            finished_sending_payloads={request_id: payload},
        ),
    )
    eco = engine_core_outputs[request.client_index]
    assert [o.request_id for o in eco.outputs] == [request_id]
    assert request_id in eco.finished_requests
    assert_scheduler_empty(scheduler)


def test_engine_core_delivers_flushed_output_with_weight_version():
    """The engine core's abort path delivers a flushed held output as a
    normal token-bearing envelope: with the sampling weight version, so the
    output processor attributes its tokens, and with its finished id."""
    from unittest.mock import MagicMock

    from vllm.v1.engine.core import EngineCoreProc

    scheduler = _make_deferring_scheduler()
    request = create_request(max_tokens=1)
    _finish_request(scheduler, request)

    # Bare EngineCoreProc with only the state _send_abort_outputs touches,
    # like tests/v1/engine/test_engine_core.py's pause tests.
    core = object.__new__(EngineCoreProc)
    core.scheduler = scheduler
    core.output_queue = MagicMock()
    core._weight_version = "v7"

    core._send_abort_outputs(
        scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    )

    core.output_queue.put_nowait.assert_called_once()
    client_index, eco = core.output_queue.put_nowait.call_args.args[0]
    assert client_index == request.client_index
    assert [o.request_id for o in eco.outputs] == [request.request_id]
    assert eco.finished_requests == [request.request_id]
    assert eco.weight_version == "v7"
