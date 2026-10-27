# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared serving helpers for connector inline return (``return_inline``):
JSON-safe tensor encoding and rejection of unsupported request shapes."""

import asyncio

import pybase64 as base64
import pytest
import torch

from vllm.entrypoints.generate.base.serving import (
    encode_kv_transfer_tensors,
    encode_kv_transfer_tensors_async,
    validate_inline_kv_transfer,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
)
from vllm.exceptions import VLLMValidationError

pytestmark = pytest.mark.cpu_test


def test_encode_passes_through_none_and_empty():
    assert encode_kv_transfer_tensors(None) is None
    assert encode_kv_transfer_tensors({}) == {}


def test_encode_replaces_tensors_and_keeps_scalars():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    encoded = encode_kv_transfer_tensors(
        {"hidden_states": hidden, "hidden_states_inline": True}
    )
    assert encoded is not None

    # Non-tensor values are untouched.
    assert encoded["hidden_states_inline"] is True

    # Tensors become a JSON-safe {dtype, shape, data} dict that round-trips.
    entry = encoded["hidden_states"]
    assert entry["dtype"] == "bfloat16"
    assert entry["shape"] == [2, 2]
    raw = base64.b64decode(entry["data"])
    restored = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(
        entry["shape"]
    )
    assert torch.equal(restored, hidden)


def test_encode_async_matches_sync_and_skips_thread_without_tensors():
    params = {"hidden_states": torch.ones(2, 3, dtype=torch.bfloat16), "flag": 1}
    assert asyncio.run(encode_kv_transfer_tensors_async(params)) == (
        encode_kv_transfer_tensors(params)
    )
    # Tensor-free params come back untouched, as the same object.
    plain = {"hidden_states_path": "/tmp/x.safetensors"}
    assert asyncio.run(encode_kv_transfer_tensors_async(plain)) is plain
    assert asyncio.run(encode_kv_transfer_tensors_async(None)) is None


def test_validate_noop_without_return_inline():
    # No kv_transfer_params, or return_inline unset/false -> never raises.
    validate_inline_kv_transfer(None, stream=True, num_prompts=4)
    validate_inline_kv_transfer({}, stream=True, num_prompts=4)
    validate_inline_kv_transfer({"return_inline": False}, stream=True, num_prompts=4)


def test_validate_rejects_inline_with_streaming():
    with pytest.raises(ValueError, match="stream=true"):
        validate_inline_kv_transfer({"return_inline": True}, stream=True)


def test_validate_rejects_inline_with_batched_prompts():
    with pytest.raises(ValueError, match="single prompt"):
        validate_inline_kv_transfer({"return_inline": True}, num_prompts=2)


def test_validate_rejects_inline_with_beam_search():
    with pytest.raises(ValueError, match="use_beam_search"):
        validate_inline_kv_transfer({"return_inline": True}, beam_search=True)


def test_validate_rejects_inline_with_builtin_tools():
    with pytest.raises(ValueError, match="builtin tools"):
        validate_inline_kv_transfer({"return_inline": True}, builtin_tools=True)


def test_validate_allows_inline_single_prompt_non_streaming():
    validate_inline_kv_transfer({"return_inline": True}, stream=False, num_prompts=1)


def test_batch_chat_request_rejects_return_inline():
    """The batched chat endpoint forwards kv_transfer_params to every
    conversation but its response never carries them back, so return_inline
    is refused when the request is parsed."""
    with pytest.raises(VLLMValidationError, match="return_inline"):
        BatchChatCompletionRequest(
            messages=[[{"role": "user", "content": "hi"}]],
            kv_transfer_params={"return_inline": True},
        )
