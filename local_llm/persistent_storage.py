from __future__ import annotations

import os
from pathlib import Path

ENV_VOLUME_ROOT = "LOCAL_LLM_VOLUME_ROOT"
ENV_HF_HOME = "HF_HOME"
ENV_HF_HUB_CACHE = "HF_HUB_CACHE"
ENV_HUGGINGFACE_HUB_CACHE = "HUGGINGFACE_HUB_CACHE"
ENV_TRANSFORMERS_CACHE = "TRANSFORMERS_CACHE"
ENV_SENTENCE_TRANSFORMERS_HOME = "SENTENCE_TRANSFORMERS_HOME"
ENV_PIP_CACHE_DIR = "PIP_CACHE_DIR"


def detect_persistent_root() -> Path | None:
    env_root = os.environ.get(ENV_VOLUME_ROOT)
    if env_root:
        candidate = Path(os.path.expandvars(os.path.expanduser(env_root))).resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate
        return None

    default_workspace = Path("/workspace")
    if default_workspace.exists() and default_workspace.is_dir():
        return default_workspace
    return None


def apply_persistent_cache_env_defaults() -> dict[str, str]:
    root = detect_persistent_root()
    if root is None:
        return {}

    cache_root = (root / ".cache").resolve()
    hf_home = (cache_root / "huggingface").resolve()
    hf_hub_cache = (hf_home / "hub").resolve()
    transformers_cache = (hf_home / "transformers").resolve()
    sentence_cache = (cache_root / "sentence-transformers").resolve()
    pip_cache = (cache_root / "pip").resolve()

    hf_home.mkdir(parents=True, exist_ok=True)
    hf_hub_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    sentence_cache.mkdir(parents=True, exist_ok=True)
    pip_cache.mkdir(parents=True, exist_ok=True)

    defaults = {
        ENV_HF_HOME: str(hf_home),
        ENV_HF_HUB_CACHE: str(hf_hub_cache),
        ENV_HUGGINGFACE_HUB_CACHE: str(hf_hub_cache),
        ENV_TRANSFORMERS_CACHE: str(transformers_cache),
        ENV_SENTENCE_TRANSFORMERS_HOME: str(sentence_cache),
        ENV_PIP_CACHE_DIR: str(pip_cache),
    }

    applied: dict[str, str] = {}
    for key, value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value
    return applied

