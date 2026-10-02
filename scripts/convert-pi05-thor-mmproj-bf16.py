#!/usr/bin/env python3
"""Convert only the GEMM weights in a PI0/PI0.5 mmproj GGUF to BF16."""

import argparse
from pathlib import Path
import sys

import numpy as np


def f32_to_bf16_raw(values: np.ndarray) -> np.ndarray:
    """Round float values to BF16 (round-to-nearest-even), returned as raw uint16."""
    f32 = np.asarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) >> np.uint32(16)).astype(np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input F16 mmproj GGUF")
    parser.add_argument("output", type=Path, help="output mixed F16/BF16 mmproj GGUF")
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        parser.error(f"input file not found: {input_path}")
    if input_path == output_path:
        parser.error("input and output paths must be different")
    if output_path.exists() and not args.force:
        parser.error(f"output already exists: {output_path} (use --force to replace it)")

    repo_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_dir / "gguf-py"))
    import gguf

    reader = gguf.GGUFReader(input_path, "r")
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    if arch != "clip":
        parser.error(f"expected a CLIP/mmproj GGUF, got architecture {arch!r}")

    writer = gguf.GGUFWriter(output_path, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        subtype = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=subtype)

    tensors = []
    converted = []
    for tensor in reader.tensors:
        # Keep the 4-D patch convolution in F16: the CUDA im2col path currently
        # accepts F16/F32 only. The 2-D weights are consumed by GEMMs and support BF16.
        if tensor.tensor_type == gguf.GGMLQuantizationType.F16 and tensor.data.ndim == 2:
            data = f32_to_bf16_raw(tensor.data)
            raw_type = gguf.GGMLQuantizationType.BF16
            converted.append(tensor.name)
        else:
            data = tensor.data
            raw_type = tensor.tensor_type
        tensors.append(data)
        writer.add_tensor_info(tensor.name, data.shape, data.dtype, data.nbytes, raw_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for data in tensors:
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()
    print(f"converted {len(converted)} two-dimensional F16 tensors to BF16")
    print(output_path)


if __name__ == "__main__":
    main()
