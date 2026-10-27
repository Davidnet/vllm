# Hidden State Extraction

The Hidden State Extraction feature allows vLLM to save intermediate layer activations from a target model during inference. This is useful for training [EAGLE](eagle.md)-style draft models, knowledge distillation, or offline analysis of model internals.

!!! note
    It is possible to save the last-layer's output hidden states by passing `num_hidden_layers` as a layer id. Note that these are _not_ normalized using the output norm.

## Offline Example

```python
import tempfile

from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector,
)

with tempfile.TemporaryDirectory() as tmpdir:
    llm = LLM(
        model="Qwen/Qwen3-8B",
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4],
                },
            },
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": tmpdir,
            },
        ),
    )

    outputs = llm.generate(
        ["The future of AI is"],
        SamplingParams(max_tokens=1),
    )

    for output in outputs:
        path = output.kv_transfer_params["hidden_states_path"]
        obj = example_hidden_states_connector.load_hidden_states(path)
        print(f"token_ids: {obj['token_ids'].shape}")
        print(f"hidden_states: {obj['hidden_states'].shape}")
```

A complete example is available at [`examples/features/speculative_decoding/extract_hidden_states_offline.py`](../../../examples/features/speculative_decoding/extract_hidden_states_offline.py).

## Online Example

For improved performance, it is recommended to use a RAM-mounted file system such as `/dev/shm/` for online usage in which the client cleans up the files soon after they are generated.

```bash
vllm serve Qwen/Qwen3-8B \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4]}}}' \
    --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/dev/shm/hidden_states"}}'
```

## Per-Request Options

Both offline and online modes support per-request options via `kv_transfer_params`:

| Parameter | Default | Description |
| --- | --- | --- |
| `hidden_states_path` | Auto-generated | Custom file path for saving hidden states. If not set, files are saved to `<shared_storage_path>/<request_id>.safetensors`. Requires `allow_custom_save_path` to be enabled in the server config. |
| `include_output_tokens` | `False` | When `True`, save hidden states for both prompt and generated output tokens. When `False`, only prompt token hidden states are saved. |
| `return_inline` | `False` | When `True`, skip the disk write and return the extracted tensors with the request output instead. See [Inline Return](#inline-return). |

### Offline usage

Pass per-request options via `extra_args` on `SamplingParams`:

```python
SamplingParams(
    max_tokens=32,
    extra_args={
        "kv_transfer_params": {
            "hidden_states_path": "/tmp/my_output.safetensors",
            "include_output_tokens": True,
        }
    },
)
```

### Online usage

Pass `kv_transfer_params` as a top-level field in the API request:

```json
{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32,
    "kv_transfer_params": {
        "hidden_states_path": "/tmp/my_output.safetensors",
        "include_output_tokens": true
    }
}
```

## Inline Return

With `"return_inline": True` in a request's `kv_transfer_params`, no file is written; the request's final output is held until the extracted tensors have been copied to the host, then released with them attached. This avoids shared storage entirely, at the cost of carrying the full payload (`num_tokens × num_extracted_layers × hidden_size` elements) through the response path. Over HTTP the API server base64-encodes the payload in a worker thread, but the JSON serialization of the resulting string runs on the event loop and scales with payload size: a 16k-token prompt with five 4096-wide layers (655 MB of bf16, 874 MB of JSON) stalls other requests on that API server for a few seconds. Keep inline extractions well below that, or use the file path for them.

Offline, `output.kv_transfer_params` contains real `torch.Tensor`s:

```python
outputs = llm.generate(
    ["The future of AI is"],
    SamplingParams(
        max_tokens=1,
        extra_args={"kv_transfer_params": {"return_inline": True}},
    ),
)
hidden_states = outputs[0].kv_transfer_params["hidden_states"]
token_ids = outputs[0].kv_transfer_params["token_ids"]
```

Online, the tensors are base64-encoded in the JSON response as `{"dtype", "shape", "data"}` dicts: `data` holds the raw little-endian bytes of the tensor, so a client without vLLM installed can decode one with `torch.frombuffer(bytearray(base64.b64decode(obj["data"])), dtype=getattr(torch, obj["dtype"])).reshape(obj["shape"])`. `decode_base64_tensor()` from the connector module does the same:

```python
import requests

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    decode_base64_tensor,
)

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "Qwen/Qwen3-8B",
        "prompt": "The future of AI is",
        "max_tokens": 1,
        "kv_transfer_params": {"return_inline": True},
    },
).json()
params = response["kv_transfer_params"]
hidden_states = decode_base64_tensor(params["hidden_states"])
token_ids = decode_base64_tensor(params["token_ids"])
```

Because the payload must ride on a single response object, inline return is only
supported for **non-streaming**, **single-prompt**, **`n=1`**, non-beam-search
requests without **builtin tool execution**. Combining `return_inline` with
`stream=true`, a batched prompt, `n > 1`, `use_beam_search=true`, or builtin
tools (which run a multi-turn loop where each turn would overwrite the previous
turn's payload) is rejected with a 400 error rather than silently dropping
tensors. Stop _strings_ are rejected too, because they are matched in the
frontend, which finishes the request before the engine releases the held
payload; use `stop_token_ids` instead. The `n > 1` and stop-string guards are
also enforced at the engine level, so offline `LLM.generate` raises instead of
losing payloads. The batched chat endpoint (`/v1/chat/completions/batch`) and
streaming-input sessions (`AsyncLLM.generate` with an input iterator) reject
`return_inline` for the same reason: neither returns a connector payload with
its final output. Inline return is also only
available on the **Python frontend**; the Rust frontend forwards
`kv_transfer_params` but does not decode inline tensor payloads, so it rejects
`return_inline` requests.

## Configuration

The `kv_connector_extra_config` dict accepts these server-level options:

| Parameter | Default | Description |
| --- | --- | --- |
| `shared_storage_path` | `/tmp` | Directory where hidden state files are saved (used when `hidden_states_path` is not set per-request) |
| `allow_custom_save_path` | `False` | Allow API clients to specify custom file paths via `hidden_states_path`. When disabled, client-provided paths are ignored with a warning. Enable only with trusted clients — custom paths can write to arbitrary locations on the server. |
| `num_writer_threads` | `8` | Thread pool size for async disk writes |
| `use_synchronization_lock` | `True` | Use file locks so concurrent readers block until writes complete. Can be disabled for batch generation where synchronization is not needed. |

## Output Format

Each request produces a `.safetensors` file containing:

- **`hidden_states`** — shape `[num_tokens, num_extracted_layers, hidden_size]`
- **`token_ids`** — shape `[num_tokens]`

The file path is returned in `output.kv_transfer_params["hidden_states_path"]`. Use `load_hidden_states()` from the connector module to read the file with proper synchronization.

With [inline return](#inline-return), the same two tensors are returned under the `hidden_states` and `token_ids` keys of `kv_transfer_params` instead, alongside `"hidden_states_inline": True`. If the engine aborts the request (shutdown, a pause in abort mode, or a fault) after generation finished but before the tensors were copied out, the response keeps its tokens, has `kv_transfer_params` set to `null` (no tensors and no `hidden_states_inline` flag), and reports `finish_reason` `"abort"`.

!!! note
    Chunked prefill is not compatible with this feature and must be disabled.
