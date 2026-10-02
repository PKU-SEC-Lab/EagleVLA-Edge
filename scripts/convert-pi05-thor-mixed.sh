#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 INPUT_F16_GGUF OUTPUT_GGUF [BUILD_DIR]" >&2
    exit 2
fi

input_model=$1
output_model=$2
build_dir=${3:-build-thor}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
type_file="$repo_dir/configs/pi05-thor-bf16-prefix.tensor-types"
quantize="$repo_dir/$build_dir/bin/llama-quantize"

if [[ ! -f "$input_model" ]]; then
    echo "input model not found: $input_model" >&2
    exit 1
fi
if [[ ! -x "$quantize" ]]; then
    echo "llama-quantize not found: $quantize" >&2
    echo "build it with: cmake --build $build_dir --target llama-quantize -j" >&2
    exit 1
fi

"$quantize" \
    --tensor-type-file "$type_file" \
    "$input_model" \
    "$output_model" \
    F16
