#!/usr/bin/env bash
# NETWORKED maintainer preparation, never called by serve.sh.
# Downloads the pinned revision: the official base files at the folder root and the unmerged LoRA in adapter/.
set -euo pipefail
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec "${PYTHON:-python3}" -B - "$PKG" "${1:?new absolute external model directory}" <<'PY'
import json,re,sys
from pathlib import Path
pkg=Path(sys.argv[1]).resolve(); dest=Path(sys.argv[2])
m=json.loads((pkg/'MODEL_MANIFEST.json').read_text())
if not dest.is_absolute():raise SystemExit('Use an absolute destination')
dest=dest.resolve()
if dest.exists() or dest.is_relative_to(pkg) or pkg.is_relative_to(dest):raise SystemExit('Use a new separate model directory')
if m['repo_id'].startswith('OWNER/') or not re.fullmatch('[0-9a-f]{40}',str(m.get('revision') or '')):
    raise SystemExit('Set the own Hugging Face repository and a complete 40-character revision in MODEL_MANIFEST.json')
if not m['files'] or not m['adapter']['files']:raise SystemExit('Model hash manifest is pending')
patterns=[*m['files'],*(m['adapter']['subdir']+'/'+name for name in m['adapter']['files'])]
from huggingface_hub import snapshot_download
snapshot_download(repo_id=m['repo_id'],revision=m['revision'],local_dir=dest,allow_patterns=patterns)
sys.path.insert(0,str(pkg))
from launch import check_model
check_model(dest,full=True)
print('Downloaded base and adapter files match the pinned manifest; no inference or submission performed.')
PY
