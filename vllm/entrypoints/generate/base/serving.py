# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import time
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, ClassVar, Generic, TypeVar

import pybase64 as base64
import torch
from fastapi import Request
from pydantic import ConfigDict
from starlette.datastructures import Headers

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.beam_search.online import BeamSearchOnlineMixin
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    GenerationError,
    PerRequestTimingMetrics,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.engine.typing import AnyRequest
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.logprobs import Logprob, PromptLogprobs
from vllm.lora.request import LoRARequest
from vllm.tokenizers import TokenizerLike
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)
from vllm.v1.metrics.stats import RequestStateStats
from vllm.v1.utils import tensor_data

logger = init_logger(__name__)

RequestT = TypeVar("RequestT", bound=AnyRequest)
_T = TypeVar("_T")


def build_per_request_timing_metrics(
    metrics: RequestStateStats | None,
    num_generation_tokens: int,
) -> PerRequestTimingMetrics:
    """Build per-request timing metrics from ``RequestStateStats``.

    ``generation_time_ms`` is the decode interval only (first output token to
    last output token); it excludes both queue wait and prefill/TTFT.
    ``tokens_per_second`` is overall output throughput: all generated tokens
    over the inference interval (scheduling to last output token), so it counts
    the prefill/TTFT phase and is not simply the reciprocal of ``mean_itl_ms``.
    Each field is left ``None`` when the timestamps it depends on are
    unavailable.
    """
    if metrics is None:
        return PerRequestTimingMetrics()

    queued_ts = metrics.queued_ts
    scheduled_ts = metrics.scheduled_ts
    first_token_ts = metrics.first_token_ts
    last_token_ts = metrics.last_token_ts

    time_to_first_token_ms: float | None = None
    generation_time_ms: float | None = None
    queue_time_ms: float | None = None
    mean_itl_ms: float | None = None
    tokens_per_second: float | None = None

    if scheduled_ts > 0 and first_token_ts > 0:
        time_to_first_token_ms = (first_token_ts - scheduled_ts) * 1000

    if first_token_ts > 0 and last_token_ts > 0:
        generation_time_ms = (last_token_ts - first_token_ts) * 1000

    if queued_ts > 0 and scheduled_ts > 0:
        queue_time_ms = (scheduled_ts - queued_ts) * 1000

    if first_token_ts > 0 and last_token_ts > 0 and num_generation_tokens > 1:
        decode_time = last_token_ts - first_token_ts
        mean_itl_ms = decode_time / (num_generation_tokens - 1) * 1000

    if scheduled_ts > 0 and last_token_ts > 0:
        inference_time_ms = (last_token_ts - scheduled_ts) * 1000
        if inference_time_ms > 0:
            tokens_per_second = num_generation_tokens / inference_time_ms * 1000

    return PerRequestTimingMetrics(
        time_to_first_token_ms=time_to_first_token_ms,
        generation_time_ms=generation_time_ms,
        queue_time_ms=queue_time_ms,
        mean_itl_ms=mean_itl_ms,
        tokens_per_second=tokens_per_second,
    )


@dataclass(kw_only=True)
class ServeContext(Generic[RequestT]):
    request: RequestT
    raw_request: Request | None = None
    model_name: str
    request_id: str
    created_time: int = field(default_factory=lambda: int(time.time()))
    lora_request: LoRARequest | None = None
    engine_inputs: list[EngineInput] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class GenerateBaseServing(BaseServing, BeamSearchOnlineMixin):
    request_id_prefix: ClassVar[str] = """
    A short string prepended to every request’s ID.
    """

    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__(
            models=models,
            model_config=engine_client.model_config,
            request_logger=request_logger,
        )

        self.engine_client = engine_client
        self.return_tokens_as_token_ids = return_tokens_as_token_ids
        self.renderer = engine_client.renderer
        self.input_processor = engine_client.input_processor
        vllm_config = getattr(engine_client, "vllm_config", None)
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        self.has_kv_connector = kv_transfer_config is not None

        # Computed once at startup (cached by ``vllm_config`` identity) and
        # stamped on non-streaming responses. Streaming chunks deliberately
        # omit it to avoid per-chunk overhead.
        from vllm.entrypoints.serve.utils.fingerprint import get_system_fingerprint

        try:
            self.system_fingerprint: str | None = get_system_fingerprint(
                engine_client.vllm_config
            )
        except Exception:
            # Never fail server startup over the fingerprint.
            self.system_fingerprint = None

    def create_streaming_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> str:
        json_str = json.dumps(
            self.create_error_response(
                message=message,
                err_type=err_type,
                status_code=status_code,
                param=param,
            ).model_dump()
        )
        return json_str

    def _raise_if_error(self, finish_reason: str | None, request_id: str) -> None:
        """Raise GenerationError if finish_reason indicates an error."""
        if finish_reason == "error":
            logger.error(
                "Request %s failed with an internal error during generation",
                request_id,
            )
            raise GenerationError("Internal server error")

    def _convert_generation_error_to_streaming_response(
        self, e: GenerationError
    ) -> str:
        """Convert GenerationError to streaming error response."""
        return self.create_streaming_error_response(
            str(e),
            err_type="InternalServerError",
            status_code=e.status_code,
        )

    async def _get_trace_headers(
        self,
        headers: Headers,
    ) -> Mapping[str, str] | None:
        is_tracing_enabled = await self.engine_client.is_tracing_enabled()

        if is_tracing_enabled:
            return extract_trace_headers(headers)

        if contains_trace_headers(headers):
            log_tracing_disabled_warning()

        return None

    @staticmethod
    def _get_data_parallel_rank(raw_request: Request | None) -> int | None:
        """Pulls the data parallel rank from a header, if provided"""
        if raw_request is None:
            return None

        rank_str = raw_request.headers.get("X-data-parallel-rank")
        if rank_str is None:
            return None

        try:
            return int(rank_str)
        except ValueError:
            return None

    async def _with_kv_transfer_rejection_cleanup(
        self,
        awaitable: Awaitable[_T],
        request: ChatCompletionRequest | CompletionRequest | ResponsesRequest,
        raw_request: Request | None,
    ) -> _T:
        """Wrap a `create_*` coroutine so that, if it raises or returns an
        ErrorResponse (i.e. the request never reached the engine), the KV
        connector is notified to free any pinned remote-prefill blocks."""
        kv_transfer_params = self.has_kv_connector and request.kv_transfer_params
        if not kv_transfer_params or not kv_transfer_params.get("do_remote_prefill"):
            return await awaitable

        notify = True
        try:
            result = await awaitable
            if not isinstance(result, ErrorResponse):
                notify = False
            return result
        finally:
            if notify:
                try:
                    await self.engine_client.notify_kv_transfer_request_rejected(
                        request.request_id,
                        kv_transfer_params,
                        data_parallel_rank=self._get_data_parallel_rank(raw_request),
                    )
                except Exception:
                    logger.warning(
                        "Failed to notify KV connector about rejected request %s",
                        request.request_id,
                        exc_info=True,
                    )

    @staticmethod
    def _get_decoded_token(
        logprob: Logprob,
        token_id: int,
        tokenizer: TokenizerLike | None,
        return_as_token_id: bool = False,
    ) -> str:
        if return_as_token_id:
            return format_token_id_placeholder(token_id)

        if logprob.decoded_token is not None:
            return logprob.decoded_token

        if tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because `skip_tokenizer_init=True`"
            )

        return tokenizer.decode([token_id])


def format_token_id_placeholder(token_id: int) -> str:
    return f"token_id:{token_id}"


def resolve_token_id_placeholder(
    token: str, tokenizer: TokenizerLike
) -> tuple[str, list[int] | None]:
    """Decode a 'token_id:N' placeholder back to a token string and UTF-8 bytes.

    Returns (token, None) unchanged if token is not a placeholder.
    This is the inverse of format_token_id_placeholder / _get_decoded_token
    when return_as_token_id=True.
    """
    suffix = token.removeprefix("token_id:")
    if suffix == token:
        return token, None
    try:
        token_id = int(suffix)
    except ValueError:
        return token, None
    token_repr = tokenizer.convert_ids_to_tokens([token_id])[0]
    if token_repr is None:
        logger.warning_once(
            "resolve_token_id_placeholder: token_id %d has no vocab entry; "
            "substituting empty string",
            token_id,
        )
        return "", None
    token_str = tokenizer.convert_tokens_to_string([token_repr])
    return token_str, list(token_str.encode("utf-8", errors="replace"))


def clamp_prompt_logprobs(
    prompt_logprobs: PromptLogprobs | None,
) -> PromptLogprobs | None:
    if prompt_logprobs is None:
        return prompt_logprobs

    for logprob_dict in prompt_logprobs:
        if logprob_dict is None:
            continue
        for logprob_values in logprob_dict.values():
            if logprob_values.logprob == float("-inf"):
                logprob_values.logprob = -9999.0
    return prompt_logprobs


def encode_kv_transfer_tensors(
    kv_transfer_params: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Make kv_transfer_params JSON-safe by base64-encoding tensor values.

    Connectors can return tensors inline (e.g. extracted hidden states),
    which surface in RequestOutput.kv_transfer_params as raw torch.Tensors.
    Each is replaced with a ``{"dtype", "shape", "data"}`` dict holding its raw
    bytes base64-encoded (``np.save`` can't represent dtypes like bfloat16).
    Clients decode with ``example_hidden_states_connector.decode_base64_tensor``.

    Shared across the completions, chat, and responses serving boundaries so
    every endpoint that accepts kv_transfer_params encodes tensors uniformly
    instead of leaking raw torch.Tensors into JSON serialization.
    """
    if not kv_transfer_params:
        return kv_transfer_params
    return {
        key: {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "data": base64.b64encode(tensor_data(value)).decode("ascii"),
        }
        if isinstance(value, torch.Tensor)
        else value
        for key, value in kv_transfer_params.items()
    }


def validate_inline_kv_transfer(
    kv_transfer_params: dict[str, Any] | None,
    *,
    stream: bool | None = False,
    num_prompts: int = 1,
    n: int | None = 1,
    beam_search: bool = False,
    builtin_tools: bool = False,
) -> None:
    """Reject unsupported combinations of connector inline return.

    Inline return (``"return_inline": true`` in kv_transfer_params) holds the
    request's final output until the connector reports the payload, then
    surfaces the tensors on the single response object. That contract requires
    exactly one response to carry the payload, so it is incompatible with:
    streaming (no place to put the deferred payload without silently dropping
    it), batched prompts (a single response has one kv_transfer_params slot),
    parallel sampling (``n > 1`` fans out to one connector request per child,
    and every child but the last has its payload discarded by the output
    processor), beam search (which spawns internal sampling requests that
    do not carry kv_transfer_params and discards their connector output), and
    builtin tool execution (which drives a multi-turn loop where each engine
    turn overwrites the previous turn's kv_transfer_params, keeping only the
    last). Reject them up front rather than lose data.

    Raises:
        ValueError: if return_inline is combined with streaming, >1 prompt,
            n > 1, beam search, or builtin tools.
    """
    if not kv_transfer_params or not kv_transfer_params.get("return_inline"):
        return
    if stream:
        raise ValueError("return_inline is not supported with stream=true")
    if num_prompts > 1:
        raise ValueError("return_inline requires a single prompt")
    if n is not None and n > 1:
        raise ValueError("return_inline requires n=1")
    if beam_search:
        raise ValueError("return_inline is not supported with use_beam_search")
    if builtin_tools:
        raise ValueError("return_inline is not supported with builtin tools")
