# metask-jev-rain-31B-trained

MeTask's 31B model makes structured decisions: given a state and a question, this service returns a judgment and probabilities for the available options. The [trained weights](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained) can be downloaded for private deployment. As of **2026-10-17**, its organizer-measured JevBench Capability Score is **79.6**, ranked **#3** in the qualifying open-weight capability ranking (service v3).

The entries ranked #3 and #5 use the same 31B model and identical weights. The #3 evaluation (79.6) updated the service layer, including the probability setting for yes/no questions; the earlier service scored 78.9. We proactively disclosed the changes to the benchmark operator, which records them on the leaderboard. The weights are public and available for download, use and inspection.

The leaderboard records this disclosure on the #3 row. The yes/no probability constant actually deployed is the yes/no accuracy on our own 1,419-item development set; an earlier draft value that had drawn on the previous row's published aggregate was dropped before submission. The data behind every service parameter is listed in the [service disclosure](DISCLOSURE.md).

This is the public service-code mirror; the score belongs to the evaluated HF service v3 release, not a new evaluation of this mirror. Service code is at this repository's root and under `service/` on HF. See [technical notes](TECHNICAL_NOTES.md) for the evaluated revision and mirror verification.

## Current trained models

Results checked on **2026-10-17** against [JevBench board v1.7.46](https://benchmarkheaven.com/jev-models) (page updated 2026-10-16). Scores are organizer-measured **Capability Scores**. Capability ranks are among **129 qualifying open-weight systems** under the official cost and latency limits; they are not composite-score ranks. The leaderboard's 'official #' label refers to its separate composite ranking.

For these full-set results, the operator measures each system on 1,500 decisions: 1,200 sealed items that submitters cannot access and 300 public items.

| Model | Capability Score | Capability rank | Weights | Service |
|---|---:|---|---|---|
| MeTask 31B (service v3) | 79.6 | 3 | [Model card](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained) | [31B service](https://github.com/chenyuwang166-oss/metask-jev-rain-31B-trained) |
| MeTask 12B | 77.9 | 6; highest among entries labeled 12B in the system name | [Model card](https://huggingface.co/MeTask-Rain/metask-jev-rain-12B-trained) | [12B service](https://github.com/chenyuwang166-oss/metask-jev-rain-12B-trained) |


Published by MeTask under the public GitHub account `chenyuwang166-oss`. Project and contact entry point: [MeTask-Rain](https://huggingface.co/MeTask-Rain).

## License and attribution

We trained this model for decision tasks on Google's open-weight Gemma 4 31B base. Use the base checkpoint together with our trained adapter. Model provenance, adapter structure and quantization are documented in the [model card](MODEL_CARD.md) and [disclosure](DISCLOSURE.md).

The applicable license is provided in [LICENSE](LICENSE). Retain the applicable license and attribution notices when redistributing.

Contact: chenyuwang166@gmail.com

## Requirements

- One NVIDIA GPU used only by this service: compute capability 8.0 or newer, NVIDIA driver 580 or
  newer. GPU memory: measurements used one NVIDIA H200 with 141 GB. The evaluated service also
  completed the public 231-item README workflow on one NVIDIA A100 80GB (231/231, zero failed).
  Use an 80 GB or larger GPU; smaller GPUs have not been validated. The service starts vLLM with `gpu_memory_utilization=0.94`
  (it reserves 94% of the GPU memory), `max_model_len=6656`, `enable_lora` and `max_lora_rank=16`.
  The model files take 23.3 GB (base) plus 495 MB (adapter). The commands use the host's GPU 0
  (`--gpus '"device=0"'`).
- Docker with the NVIDIA Container Toolkit.
- On the host: Git and Python with pip; step 2 installs the Hugging Face command line tool `hf`.
- Runtime image `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`
  (Python 3.12.3, vLLM 0.31.0, torch 2.13.0+cu130), the linux/amd64 image of vLLM release
  `v0.31.0`. Nothing is installed and the service has no network access.
- Model folder: the model files of https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained/tree/2feff076f07711fe2fefe7a6fdcedfe4132b773e as regular
  files (step 2). The repository root holds the official base files; `adapter/` holds the adapter
  (`adapter_config.json`, `adapter_model.safetensors`). A Hugging Face cache `snapshots/<revision>`
  directory holds links and cannot be mounted on its own. At startup the service checks the SHA-256
  of the small model files, the size of the weight file and the full SHA-256 of both adapter files
  (`MODEL_MANIFEST.json`), and refuses other weight, tokenizer, processor or template files in the
  folder; it never writes to the folder.
- Code: this repository (download command in step 1). Its source is the evaluated HF `service/` directory.
- Harness: https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc

## Run

Copy the commands as they are, in the same Bash terminal.

1. Obtain the code and harness, and create a fresh work directory. Run from a writable parent
   directory; the three new directories must not already contain another run:

   ```bash
   git clone https://github.com/chenyuwang166-oss/metask-jev-rain-31B-trained.git
   PKG="$(pwd)/metask-jev-rain-31B-trained"
   git clone https://github.com/fstandhartinger/jevbench.git jevbench
   git -C jevbench checkout bb05a335bc809e61b20c0f745d25499a82b326fc
   JEVBENCH="$(pwd)/jevbench"
   MODEL="$(pwd)/model-31b"
   WORK="$(pwd)/work-31b-run1"
   mkdir "$MODEL" "$WORK"
   TASKS=/jevbench/datasets/public/easy.jsonl,/jevbench/datasets/public/original.jsonl,/jevbench/datasets/public/hard.jsonl
   ```

2. Install the download tool, fetch pinned weights, and pull the pinned runtime image:

   ```bash
   python3 -m pip install huggingface_hub
   hf download MeTask-Rain/metask-jev-rain-31B-trained --revision 2feff076f07711fe2fefe7a6fdcedfe4132b773e --local-dir "$MODEL" --exclude 'service/*'
   docker pull vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
   ```

3. Start the service:

   ```bash
   docker run -d --name jev31b --init --gpus '"device=0"' --network none --read-only --shm-size 8g \
     -v "$PKG":/pkg:ro -v "$MODEL":/model:ro -v "$WORK":/work -v "$JEVBENCH":/jevbench:ro \
     --entrypoint bash vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b \
     /pkg/serve.sh --model /model --state-dir /work/state --run-id run1
   ```

4. Follow the log until the service is ready:

   ```bash
   docker logs -f jev31b
   ```

   When a line starting with `{"status": "ok"` appears, press Ctrl-C (this ends only the log
   display) and check health before step 5:

   ```bash
   docker exec jev31b python3 -c 'import urllib.request; print(urllib.request.urlopen("http://localhost:8000/health").read().decode())'
   ```

    If the log ends before that line, the container has exited: the
   reason is in `docker logs jev31b` and in `$WORK/state/run1-startup/attempt-*.log`; go to step 6.
   After correcting the cause, use a new `WORK` and restart from step 3.

5. Run the official harness in the same container, then summarize:

   ```bash
   docker exec -e PYTHONPATH=/jevbench jev31b python3 -B -m jevbench.cli run --adapter typesafe \
     --endpoint http://localhost:8000 --model metask-jev-rain-31B-trained --key-env '' --tasks "$TASKS" \
     --results /work/run1/results.jsonl --ledger /work/run1/ledger.jsonl --raw-dir /work/run1/raw \
     --manifest /work/run1/manifest.json --cap-usd 15 --price-in-per-m 0.09 --price-out-per-m 0.34
   docker exec -e PYTHONPATH=/jevbench jev31b python3 -B -m jevbench.cli summarize --tasks "$TASKS" \
     --results /work/run1/results.jsonl --ledger /work/run1/ledger.jsonl
   ```

6. Stop and remove the container:

   ```bash
   docker stop -t 40 jev31b && docker rm jev31b
   ```

Each evaluation uses a new `WORK` and a new container. A run name that already exists below the
state directory is refused at startup ("use a new --state-dir"); with these commands that means a
new `WORK`.

## What to expect

- Health line: `"status": "ok"`, `"warnings": []`, `"model_key": "metask-gemma4-31b-lora-w4a16"`,
  `"mode": "routed"`, `"quantization_kernel": "compressed-tensors"`, `"prefix_caching": true`, `"supports_slow_session": true`, and
  `threshold_identity.adapter` equal to `/model/adapter`, the verified adapter folder.
- Startup: `launch.py` verifies the model folder and the adapter, writes the run configuration
  (with the adapter folder filled in) to `$WORK/state/run1-startup/config.runtime.json`, and starts
  vLLM with `enable_lora` and `max_lora_rank=16`; it sets `VLLM_BATCH_INVARIANT=1`,
  `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_WORKER_MULTIPROC_METHOD=spawn`. Each startup attempt waits at most 1800 s for readiness: the
  first lets vLLM select the attention backend; only if the engine fails to start does a second
  attempt request `TRITON_ATTN`. The log of attempt N is `$WORK/state/run1-startup/attempt-N.log`.
- Each decision is first read in one prompt pass. At most 5.6% of decisions take a second, drafting pass (at most 256 generated tokens) before the answer is read again; DISCLOSURE.md states the rule and the cost fuse. Inputs longer than 6,653 tokens are answered in one pass from a context shortened to at most 6,653 tokens by the service's extraction rule.
- Writes: the service writes files only below `$WORK/state` (plus shared memory in `/dev/shm`) and
  the harness only below `$WORK/run1`. The package and model mounts are read-only. Cache and
  configuration path variables (`HF_HOME`, `XDG_*`, `TRITON_*`, `TORCH_HOME` and similar) must not
  be passed to the container: a value outside `/work/state` is refused at startup. No script of
  this package deletes directories recursively.
- Responses: an item the service cannot process returns HTTP 422 without answer and without usage;
  after an engine failure, and from the third consecutive item failure on, the service answers
  every `POST /v1/systemone` with HTTP 503 and `/health` reports `"status": "error"`. No substitute
  answer is ever returned.


The evaluated health response retains `threshold_identity.quant: "awq"` as a historical
configuration alias; the actual backend is **compressed-tensors W4A16**, reported by
`quantization_kernel`. There is no top-level `quant` field. This mirror preserves that behavior.
Mirror identity and historical validation: [technical notes](TECHNICAL_NOTES.md).

## Package checks (offline, no GPU or network)

```bash
cd "$PKG"
python3 -B pkg_check.py             # package integrity, configuration and adapter contract
python3 -B selftest.py --offline    # synthetic software tests, including long-input compression
```

`MODEL_MANIFEST.json` pins the repository, revision and every model and adapter file;
`ADAPTER_MANIFEST.json` records the adapter's origin by SHA-256; `FREEZE.json` records the freeze
of the fitted configuration; `MANIFEST.sha256` binds all package files. `MODEL_CARD.md` describes the weights hosted on Hugging Face. `MIRROR.json` binds this public
copy to the original service hashes; `FREEZE.json` retains the historical freeze evidence,
with obsolete authorization metadata removed. The mirror checker verifies both histories
and the current files rather than presenting this copy as the original freeze.
