# metask-jev-rain-31B-trained: disclosure

## Model

- Base model: `google/gemma-4-31B-it`. The weights used are Google's official
  quantization-aware-training (QAT) releases of this model:
  - training base: `google/gemma-4-31B-it-qat-q4_0-unquantized` at revision
    `1e4d8beecacb8b7590c1d8bedd7335f687bf311f` (BF16);
  - served base: `google/gemma-4-31B-it-qat-w4a16-ct` at revision
    `52f3f65bc7a02d555763bc923bd1d9094898219d`, Google's compressed-tensors W4A16 checkpoint
    (4-bit weights, group size 32, symmetric; activations unquantized). It is used unchanged: the
    files at the root of our model repository are byte-identical to that revision. We did not
    quantize anything.
- Own trained weights: the LoRA adapter in `adapter/` (PEFT format), rank 16, alpha 32, on the
  language model's q/k/v/o/gate/up/down projections, trained by us in BF16 on the training base.
- The adapter is not merged: vLLM 0.31.0 loads it with `enable_lora` and `max_lora_rank=16` and
  applies it to every request on top of the W4A16 base.
- In the 10 full-attention layers (5, 11, …, 59) Gemma 4 computes the attention values with the key projection (`attention_k_eq_v`); these layers have no v_proj, and vLLM serves them with separate key and value slots. `adapter/adapter_model.safetensors` therefore contains the trained checkpoint tensors unchanged plus, for these 10 layers, a v_proj LoRA that is a byte copy of the trained k_proj LoRA (20 tensors), so that the update applies to keys and values as in training.

## Data

- Training and checkpoint selection used only our own generated data; no public or held-out JevBench item was used for training or checkpoint selection. The 231 public JevBench items were used to fit the per-type temperatures of the service (unchanged from the previous version of this service) and the threshold of the drafting pass, and, together with our own development data, to choose the drafting share, the draft length and the drafting prompt limit; they were also used in our cost, latency and repeatability measurements. Fits and choices made on these public items are in-sample; their local results are not independent evaluation results.
- The yes/no confidence constant 0.898 (answers carry 0.898 or 0.102) is the yes/no accuracy of our own held-out development data (1,419 yes/no items), rounded to 3 decimals; its value does not depend on any leaderboard result.

## Serving

- Each decision is first read in one prompt pass: the answer distribution comes from the next-token log-probabilities of the answer forms. If the gap between the two highest answer scores of this pass is below a threshold (fitted on the 231 public items so that 13 of them both fall below it and can take the drafting pass), the decision may take a second, drafting pass: the model writes at most 256 tokens of reasoning (the instruction asks for about 160 words), then the answer distribution is read again. If the draft ends without an answer line, an answer line is appended and the answer is read after the draft. A drafting pass that does not finish within 60 s is abandoned and the first-pass answer is returned. Answers after a drafting pass use the same calibration parameters (per-type temperatures and yes/no confidence) as first-pass answers.
- Decisions whose drafting prompt (the first-pass prompt plus the drafting instruction) is longer than 840 tokens are answered in one pass. Inputs longer than 6,653 tokens never take the drafting pass.
- Drafting share (slow-path quota 5.6%): a counter kept below the service's state directory admits a drafting pass only while drafting decisions stay at most 5.6% of all decisions so far.
- Cost fuse: from the 101st decision on, when the running engine cost per decision reaches 90% of USD 0.0646 per 1,000 decisions, drafting passes are limited to the share that brings the running cost back to 85% of that limit (none while first-pass decisions alone cost more than 85% of it), until the running cost falls to 85% of the limit. The fuse never changes the first pass.
- Inputs longer than 6,653 tokens are answered from a context shortened to at most 6,653 tokens by
  the service's existing extraction rule; an input is not refused because of its length.

## Measurement conditions

- Cost and latency were measured by us on one NVIDIA H200 (RunPod) with vLLM 0.31.0,
  `VLLM_BATCH_INVARIANT=1` (also set by the launcher for serving) and
  `VLLM_USE_FLASHINFER_SAMPLER=0`, by direct local calls, one request at a time; model loading and
  warm-up are excluded.
- Cost: measured engine token counts (input tokens minus cached prefix tokens; generated tokens)
  at USD 0.09 / 0.34 per million input / output tokens.
- Latency: adjusted p50 = 2 × raw p50 + 0.15 s. This conversion was not recalibrated on the H200.

| Measurement (the 231 public JevBench items) | Result |
|---|---|
| Engine cost per 1,000 decisions | USD 0.0603 |
| Raw p50 / adjusted p50 latency per decision | 0.237 s / 0.624 s |
| Repeatability, 50 public items × 4 fresh processes: largest probability difference | 0.000000 (limit 0.005) |
| Decisions with a drafting pass | 4 of 231 (1.7%) |
