# Service provenance and verification

Evaluated version: [`2feff076f07711fe2fefe7a6fdcedfe4132b773e`](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained/tree/2feff076f07711fe2fefe7a6fdcedfe4132b773e).
This repository is its public service-code mirror, with documentation and packaging-hygiene
corrections only; all runtime files are byte-for-byte identical to the evaluated service.
See [CHANGES.md](CHANGES.md).
Service code is at the root of this GitHub mirror; the same code is under `service/`
in the HF model repository. Paths beginning with `service/` in the model card refer to HF.
Weights, calibration, routing, decoding settings and decision computation are unchanged.
The model manifest's earlier weight revision is retained because it pins the same model files.
The official score above belongs to the evaluated HF version; this mirror has not been re-evaluated.

A self-hosted decision service for JevBench. It serves `POST /v1/systemone` for the unchanged
`typesafe` adapter of the official harness, and `GET /health`. The model has two parts that are
loaded together and never merged: Google's official W4A16 checkpoint
`google/gemma-4-31B-it-qat-w4a16-ct` (unchanged), and our LoRA adapter, which vLLM applies to every
request. Facts and measurement conditions are in [DISCLOSURE.md](DISCLOSURE.md).

For a quick check: (1) read the dated official result and its sources in the [README](README.md);
(2) check the evaluated revision and mirror scope here; (3) follow [Run](README.md#run).
Evidence files (`MEASURED_EVIDENCE.json`, `FREEZE.json`, `fit_report.json`, and related
manifests) support detailed audits; most readers can use these three checks.


All runtime files are byte-for-byte identical to the evaluated version, which passed the
A100 80GB cold run (231/231 public items, zero failed). No additional mirror GPU run is required
for these documentation and packaging changes. To verify runtime identity, run
`python3 -B pkg_check.py`; compare the runtime SHA-256 values with `MIRROR.json.source_files`
and the pinned HF service files for independent verification.
