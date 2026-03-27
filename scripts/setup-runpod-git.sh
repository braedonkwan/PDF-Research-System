#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

log() {
  echo "[setup-runpod] $*"
}

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

run_apt_install=0
if command -v apt-get >/dev/null 2>&1; then
  if [[ -n "${SUDO}" || "$(id -u)" -eq 0 ]]; then
    run_apt_install=1
  else
    log "apt-get is available but this user has no sudo access; skipping apt package install."
  fi
fi

if [[ "${run_apt_install}" -eq 1 ]]; then
  log "Installing base system dependencies with apt-get..."
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    cmake \
    ninja-build \
    build-essential \
    ca-certificates
fi

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}" >&2
    exit 1
  fi
}

pip_install() {
  if python3 -m pip "$@"; then
    return 0
  fi
  python3 -m pip "$@" --break-system-packages
}

require_cmd python3
require_cmd git
require_cmd cmake

log "Installing Python dependencies..."
pip_install install --upgrade pip
pip_install install -r requirements.txt

# sentence-transformers/transformers model loading path now requires torch>=2.6
# for safe torch.load behavior (CVE-2025-32434). Install a matched PyTorch trio
# to avoid torchvision operator mismatch errors at import time.
torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
torch_version="${TORCH_VERSION:-2.6.0}"
torchvision_version="${TORCHVISION_VERSION:-0.21.0}"
torchaudio_version="${TORCHAUDIO_VERSION:-2.6.0}"

log "Installing PyTorch stack..."
if ! pip_install install --upgrade --force-reinstall --index-url "${torch_index_url}" \
  "torch==${torch_version}" \
  "torchvision==${torchvision_version}" \
  "torchaudio==${torchaudio_version}"; then
  log "CUDA PyTorch install failed from ${torch_index_url}; retrying with default index."
  pip_install install --upgrade --force-reinstall \
    "torch==${torch_version}" \
    "torchvision==${torchvision_version}" \
    "torchaudio==${torchaudio_version}"
fi

python3 - <<'PY'
import sentence_transformers  # noqa: F401
import torch
import torchvision
print(f"PyTorch stack OK: torch={torch.__version__}, torchvision={torchvision.__version__}")
PY

volume_root="${LOCAL_LLM_VOLUME_ROOT:-/workspace}"
export LOCAL_LLM_VOLUME_ROOT="${volume_root}"
export LOCAL_LLM_MODELS_ROOT="${LOCAL_LLM_MODELS_ROOT:-${volume_root}/models}"
export LOCAL_LLM_LLAMA_SERVER_PATH="${LOCAL_LLM_LLAMA_SERVER_PATH:-${volume_root}/bin/llama-server}"
export HF_HOME="${HF_HOME:-${LOCAL_LLM_VOLUME_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${LOCAL_LLM_VOLUME_ROOT}/.cache/sentence-transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${LOCAL_LLM_VOLUME_ROOT}/.cache/pip}"

mkdir -p "${LOCAL_LLM_MODELS_ROOT}"
mkdir -p "$(dirname "${LOCAL_LLM_LLAMA_SERVER_PATH}")"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${SENTENCE_TRANSFORMERS_HOME}" "${PIP_CACHE_DIR}"

runpod_env_file="${LOCAL_LLM_VOLUME_ROOT}/.local_llm_runpod_env.sh"
cat > "${runpod_env_file}" <<EOF
#!/usr/bin/env bash
export LOCAL_LLM_VOLUME_ROOT="${LOCAL_LLM_VOLUME_ROOT}"
export LOCAL_LLM_MODELS_ROOT="${LOCAL_LLM_MODELS_ROOT}"
export LOCAL_LLM_LLAMA_SERVER_PATH="${LOCAL_LLM_LLAMA_SERVER_PATH}"
export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"
EOF
chmod +x "${runpod_env_file}"

llama_cpp_dir="${LLAMA_CPP_DIR:-${volume_root}/llama.cpp}"
llama_cpp_ref="${LLAMA_CPP_REF:-b8070}"
llama_cpp_build_dir="${LLAMA_CPP_BUILD_DIR:-${llama_cpp_dir}/build}"
llama_cpp_extra_args="${LLAMA_CPP_CMAKE_EXTRA_ARGS:-}"

if [[ ! -d "${llama_cpp_dir}/.git" ]]; then
  log "Cloning llama.cpp into ${llama_cpp_dir}..."
  git clone https://github.com/ggml-org/llama.cpp.git "${llama_cpp_dir}"
fi

log "Preparing llama.cpp checkout (${llama_cpp_ref})..."
git -C "${llama_cpp_dir}" fetch --all --tags
git -C "${llama_cpp_dir}" checkout "${llama_cpp_ref}"

cmake_generator_args=()
if command -v ninja >/dev/null 2>&1; then
  cmake_generator_args=(-G Ninja)
fi

common_cmake_args=(
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DCMAKE_BUILD_TYPE=Release
)

if [[ -n "${llama_cpp_extra_args}" ]]; then
  # shellcheck disable=SC2206
  extra_cmake_args=( ${llama_cpp_extra_args} )
else
  extra_cmake_args=()
fi

build_parallelism="$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)"

log "Configuring llama.cpp (CUDA build attempt)..."
if cmake -S "${llama_cpp_dir}" -B "${llama_cpp_build_dir}" \
  "${cmake_generator_args[@]}" \
  "${common_cmake_args[@]}" \
  -DGGML_CUDA=ON \
  "${extra_cmake_args[@]}" \
  && cmake --build "${llama_cpp_build_dir}" --target llama-server -j"${build_parallelism}"; then
  log "Built llama-server with CUDA support."
else
  log "CUDA build failed; retrying CPU-only build."
  rm -rf "${llama_cpp_build_dir}"
  cmake -S "${llama_cpp_dir}" -B "${llama_cpp_build_dir}" \
    "${cmake_generator_args[@]}" \
    "${common_cmake_args[@]}" \
    -DGGML_CUDA=OFF \
    "${extra_cmake_args[@]}"
  cmake --build "${llama_cpp_build_dir}" --target llama-server -j"${build_parallelism}"
fi

llama_server_bin="${llama_cpp_build_dir}/bin/llama-server"
if [[ ! -x "${llama_server_bin}" ]]; then
  llama_server_bin="$(find "${llama_cpp_build_dir}" -maxdepth 5 -type f -name llama-server | head -n 1 || true)"
fi
if [[ -z "${llama_server_bin}" || ! -f "${llama_server_bin}" ]]; then
  echo "Could not locate built llama-server binary under ${llama_cpp_build_dir}" >&2
  exit 1
fi

install -m 0755 "${llama_server_bin}" "${LOCAL_LLM_LLAMA_SERVER_PATH}"
chmod +x "${LOCAL_LLM_LLAMA_SERVER_PATH}"

cat <<EOF
Setup complete.

Run these in this shell:
  source ${runpod_env_file}
  python3 models.py --model qwen35_35b
  python3 server.py --model qwen35_35b
EOF
