"""Report the attention backend named in the engine log; the requested setting is reported separately."""
import os
from pathlib import Path
import re

# vLLM 0.31 logs "Using AttentionBackendEnum.<NAME> backend."; older formats name the backend in prose.
_PATTERN = re.compile(r'\bUsing\s+(?:AttentionBackendEnum\.([A-Za-z0-9_]+)|([A-Za-z][A-Za-z0-9_ -]*?))\s+backend\b', re.I)
_ALIASES = {'FLASH_ATTENTION': 'FLASH_ATTN', 'TRITON_ATTENTION': 'TRITON_ATTN',
            'FLASH_INFER': 'FLASHINFER', 'FLASHINFER_ATTENTION': 'FLASHINFER'}


def attention_health():
    """Backend names found in the engine log of this service process (JEV_ATTENTION_LOG).

    The requested value comes from JEV_ATTENTION_BACKEND; 'auto' means vLLM selected the backend.
    A request is never reported as the active backend.
    """
    requested = os.environ.get('JEV_ATTENTION_BACKEND') or 'auto'
    text = ''
    path = os.environ.get('JEV_ATTENTION_LOG')
    if path:
        try:
            text = Path(path).read_text(encoding='utf-8', errors='replace')
        except OSError:
            pass
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    names = []
    for enum_name, prose in _PATTERN.findall(text):
        raw = (enum_name or prose).strip().upper().replace('-', '_').replace(' ', '_')
        if not enum_name and 'ATTN' not in raw and 'ATTENTION' not in raw:
            continue  # "Using X backend" lines about other kernels
        names.append(_ALIASES.get(raw, raw.removesuffix('_ATTENTION')))
    names = list(dict.fromkeys(names))
    return {'attention_backend': names[-1] if names else None,
            'attention_backends': names,
            'attention_backend_requested': requested,
            'attention_backend_source': 'engine_log' if names else 'unreported'}
