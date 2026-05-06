#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

normalize_path() {
  realpath -m "$1"
}

export TORCH_CACHE_PATH="${TORCH_CACHE_PATH:-$(normalize_path "${REPO_ROOT}/torch_cache")}"
export SOURCE_FOLDER_PATH="${SOURCE_FOLDER_PATH:-$(normalize_path "${REPO_ROOT}/../datasets_gs")}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source=*)
      export SOURCE_FOLDER_PATH="$(normalize_path "${1#*=}")"
      shift
      ;;
    --source)
      export SOURCE_FOLDER_PATH="$(normalize_path "$2")"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--source=/path/to/source_folder]"
      exit 1
      ;;
  esac
done

mkdir -p "${TORCH_CACHE_PATH}"

echo "Using repo root: ${REPO_ROOT}"
echo "Using source folder mount: ${SOURCE_FOLDER_PATH}"
echo "Using cache path: ${TORCH_CACHE_PATH}"

cd "${SCRIPT_DIR}"
docker compose run --rm gs_sdf
