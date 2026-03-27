#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

config_path="${CONFIG_PATH:-settings.json}"
server_dry_run="${SERVER_DRY_RUN:-0}"
model_alias="${MODEL_ALIAS:-}"
download_model="${DOWNLOAD_MODEL:-0}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -f "${config_path}" ]]; then
  echo "Config file not found: ${config_path}" >&2
  exit 1
fi

if [[ -z "${model_alias}" ]]; then
  model_alias="$(
    python3 - "${config_path}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
try:
    payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
except Exception:
    print("qwen35_35b")
    raise SystemExit(0)
model = str(payload.get("default_model") or "").strip()
print(model or "qwen35_35b")
PY
  )"
fi

export LOCAL_LLM_VOLUME_ROOT="${LOCAL_LLM_VOLUME_ROOT:-/workspace}"
export LOCAL_LLM_MODELS_ROOT="${LOCAL_LLM_MODELS_ROOT:-${LOCAL_LLM_VOLUME_ROOT}/models}"
export LOCAL_LLM_LLAMA_SERVER_PATH="${LOCAL_LLM_LLAMA_SERVER_PATH:-${LOCAL_LLM_VOLUME_ROOT}/bin/llama-server}"
export HF_HOME="${HF_HOME:-${LOCAL_LLM_VOLUME_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${LOCAL_LLM_VOLUME_ROOT}/.cache/sentence-transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${LOCAL_LLM_VOLUME_ROOT}/.cache/pip}"

mkdir -p "${LOCAL_LLM_MODELS_ROOT}"
mkdir -p "$(dirname "${LOCAL_LLM_LLAMA_SERVER_PATH}")"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${SENTENCE_TRANSFORMERS_HOME}" "${PIP_CACHE_DIR}"

if [[ ! -x "${LOCAL_LLM_LLAMA_SERVER_PATH}" ]]; then
  echo "llama-server not found at ${LOCAL_LLM_LLAMA_SERVER_PATH}"
  echo "Run: bash scripts/setup-runpod-git.sh"
  exit 1
fi

echo "[start-runpod] config=${config_path} model=${model_alias} download_model=${download_model} dry_run=${server_dry_run}"

if [[ "${download_model}" == "1" ]]; then
  python3 models.py --config "${config_path}" --model "${model_alias}"
fi

server_args=(python3 server.py --config "${config_path}" --model "${model_alias}")
if [[ "${server_dry_run}" == "1" ]]; then
  server_args+=(--dry-run)
fi
exec "${server_args[@]}"
