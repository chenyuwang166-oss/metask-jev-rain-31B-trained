# Public mirror changes

Source: [evaluated HF service v3](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained/tree/2feff076f07711fe2fefe7a6fdcedfe4132b773e/service).
Official results refer to that source, not to a fresh evaluation of this copy.

- README: official result table, explicit source pin, code download, A100 80GB validation,
  health check and removal of an obsolete release-approval command.
- LICENSE: correct the weight attribution to the 31B base and unmerged adapter.
- DISCLOSURE: clarify the yes/no sample denominator, independence of the constant from
  leaderboard results, and the in-sample nature of public-item fitting.
- MODEL_CARD: clarify the model/code repository split and HF `service/` paths.
- README final wording: publish the approved contact address, explain the earlier 78.9
  entry and current service v3 result, and add a short guide to the audit evidence.
- PIPELINE_PROVENANCE: clear `documents_pending_final_wording` after documentation
  finalization. This is packaging metadata, not a runtime file. Source hashes are retained.
- This documentation revision leaves every runtime file and config.json unchanged.
- Package checker: remove the internal editorial word lists and the always-failing release
  approval mode. Keep integrity, model identity, policy, calibration and offline checks;
  bind this copy to its historical evaluated source with MIRROR.json.
- Metadata: use neutral adapter/selection labels, remove obsolete authorization flags,
  correct completed fit/freeze status, and update dependent hashes. Historical source hashes
  and freeze records remain distinguishable from this mirror's checksums.
- Runtime files, including server.py: byte-for-byte identical to the evaluated service.
  The historical threshold_identity.quant alias is `awq`; actual quantization is
  compressed-tensors W4A16 (`quantization_kernel` in health). This is documented only.

These are documentation and packaging-hygiene corrections. No new weights, fit, threshold, quota, prompt, generation setting or score is introduced.
The model loading profile still uses its historical `awq` alias; vLLM auto-detects the pinned
compressed-tensors checkpoint. Changing that configuration would change its identity checks.

See MIRROR.json for original and public file hashes; MANIFEST.sha256 covers every public file
except itself. Original source files remain available at the immutable HF revision above.
