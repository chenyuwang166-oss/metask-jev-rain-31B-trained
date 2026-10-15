---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-31B-it
pipeline_tag: text-generation
tags:
- gemma4
- compressed-tensors
- lora
- peft
- vllm
- jevbench
---

# metask-jev-rain-31B-trained

Decision model behind the `metask-jev-rain-31B-trained` JevBench service. Base model:
[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it), used through Google's
official quantization-aware-training (QAT) releases of it. The model has two parts that are loaded
together and **not merged**:

| Part | Location in the HF model repository | What it is |
|---|---|---|
| Base | repository root | The files of [`google/gemma-4-31B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) at revision `52f3f65bc7a02d555763bc923bd1d9094898219d`, byte for byte: Google's official compressed-tensors W4A16 checkpoint (4-bit weights, group size 32, symmetric). Its own README and `.gitattributes` are replaced by this card and the repository default. |
| Adapter | `adapter/` | Our LoRA (PEFT format): rank 16, alpha 32, on the language model's q/k/v/o/gate/up/down projections, trained by us in BF16 on [`google/gemma-4-31B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized) at revision `1e4d8beecacb8b7590c1d8bedd7335f687bf311f`. |

`adapter/adapter_model.safetensors` contains the trained checkpoint tensors unchanged plus, for the 10 full-attention layers (5, 11, …, 59) that compute values with the key projection, a v_proj LoRA that is a byte copy of the trained k_proj LoRA (20 tensors), so that vLLM applies the update to keys and values as in training.

`adapter/adapter_config.json` names the training base above as `base_model_name_or_path`.

This GitHub repository contains service code; base weights and `adapter/` are in the
[Hugging Face model repository](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained/tree/2feff076f07711fe2fefe7a6fdcedfe4132b773e).

## Serving

Service code is at the root of this GitHub mirror and under `service/` in the HF model
repository. It starts vLLM 0.31.0 on the downloaded HF model repository's root
with `enable_lora`, `max_lora_rank=16`, and the `adapter/` folder as the single LoRA, applied to
every request. Equivalent vLLM usage:

```python
from vllm import LLM
from vllm.lora.request import LoRARequest
llm = LLM(model=MODEL_DIR, dtype="bfloat16", enable_lora=True, max_lora_rank=16)
adapter = LoRARequest("adapter", 1, MODEL_DIR + "/adapter")
# llm.generate(prompts, sampling_params, lora_request=adapter)
```

Loading only the root files gives Google's base checkpoint without our adapter.

Facts on data and the measurement conditions of cost and latency are in
[DISCLOSURE.md](DISCLOSURE.md) here (`service/DISCLOSURE.md` in the HF model repository).

