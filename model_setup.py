"""Report a missing processor config in the model folder; never downloads or copies."""
from pathlib import Path

# Gemma 4 keeps its processor settings in processor_config.json; the checkpoint has no
# preprocessor_config.json or video_preprocessor_config.json and needs neither.
NAMES=('processor_config.json',)

def report_processor_configs(model_dir):
    """Print nothing for a complete model folder; return the missing names."""
    p=Path(model_dir)
    missing=[n for n in NAMES if not (p/n).is_file()]
    if missing:
        print('[model_setup] processor config not in the model folder: '+', '.join(missing),flush=True)
    return missing
