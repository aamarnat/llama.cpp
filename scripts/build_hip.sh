#!/usr/bin/env bash
# Build script for HIP and HIP Graph with automatic GPU gfx target detection.
# Usage:
#   scripts/build_hip.sh [hip|graph] [--clean]
# Environment overrides:
#   GPU_TARGETS - set explicit gfx target (e.g. gfx1102)
#   JOBS        - parallel build jobs (default: 16)
#
# Examples:
#   scripts/build_hip.sh hip
#   scripts/build_hip.sh graph
#   GPU_TARGETS=gfx1030 scripts/build_hip.sh hip
#   JOBS=32 scripts/build_hip.sh graph --clean

set -Eeuo pipefail

print_usage() {
  echo "Usage: $0 [hip|graph] [--clean]"
  echo "  hip     - build with HIP"
  echo "  graph   - build with HIP Graphs (adds -DGGML_HIP_GRAPHS=ON)"
  echo "Options:"
  echo "  --clean - remove existing build directories before configuring"
  echo "Env:"
  echo "  GPU_TARGETS - override auto-detected gfx target (e.g. gfx1102)"
  echo "  JOBS        - parallel build jobs for Ninja (default: 16)"
}

MODE="${1:-hip}"
case "${MODE}" in
  -h|--help)
    print_usage
    exit 0
    ;;
  hip|graph)
    ;;
  *)
    echo "Error: invalid mode '${MODE}'. Must be 'hip' or 'graph'."
    print_usage
    exit 1
    ;;
esac

CLEAN="${2:-}"
JOBS="${JOBS:-16}"

if ! command -v hipconfig >/dev/null 2>&1; then
  echo "Error: 'hipconfig' not found. Ensure ROCm HIP is installed and in PATH."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${CLEAN}" == "--clean" ]]; then
  echo "Cleaning build directories..."
  rm -rf "${REPO_ROOT}/build" "${REPO_ROOT}/build_graph"
fi

HIPCXX="$(hipconfig -l)/clang"
HIP_PATH="$(hipconfig -R)"

detect_gpu_targets() {
  local targets=""
  # Try hipconfig --amdgpu-targets (ROCm may list multiple; pick the first)
  if hipconfig --amdgpu-targets >/dev/null 2>&1; then
    targets="$(hipconfig --amdgpu-targets 2>/dev/null \
      | tr ';' '\n' \
      | tr ' ' '\n' \
      | grep -E '^gfx[0-9]+' \
      | head -n1 || true)"
  fi
  # Fallback: parse rocminfo output (Name: gfxXXXX)
  if [[ -z "${targets}" ]] && command -v rocminfo >/dev/null 2>&1; then
    targets="$(rocminfo \
      | sed -n 's/^[[:space:]]*Name:[[:space:]]*\(gfx[0-9]\+\).*/\1/p' \
      | head -n1 || true)"
  fi
  echo "${targets}"
}

GPU_TARGETS="${GPU_TARGETS:-$(detect_gpu_targets)}"
if [[ -z "${GPU_TARGETS}" ]]; then
  GPU_TARGETS="gfx1100"
  echo "Warning: Could not auto-detect GPU target. Falling back to ${GPU_TARGETS}."
fi
echo "Using GPU_TARGETS=${GPU_TARGETS}"

CMAKE_COMMON_ARGS=(
  -S "${REPO_ROOT}"
  -G "Ninja"
  -DGGML_HIP=ON
  -DGPU_TARGETS="${GPU_TARGETS}"
  -DCMAKE_BUILD_TYPE=Release
)

if [[ "${MODE}" == "hip" ]]; then
  BUILD_DIR="${REPO_ROOT}/build"
  echo "Configuring HIP build in ${BUILD_DIR}..."
  HIPCXX="${HIPCXX}" HIP_PATH="${HIP_PATH}" \
    cmake "${CMAKE_COMMON_ARGS[@]}" -B "${BUILD_DIR}"
  echo "Building HIP..."
  cmake --build "${BUILD_DIR}" --config Release -- -j "${JOBS}"
else
  BUILD_DIR="${REPO_ROOT}/build_graph"
  echo "Configuring HIP Graph build in ${BUILD_DIR}..."
  HIPCXX="${HIPCXX}" HIP_PATH="${HIP_PATH}" \
    cmake "${CMAKE_COMMON_ARGS[@]}" -DGGML_HIP_GRAPHS=ON -B "${BUILD_DIR}"
  echo "Building HIP Graph..."
  cmake --build "${BUILD_DIR}" --config Release -- -j "${JOBS}"
fi

echo "Done."
