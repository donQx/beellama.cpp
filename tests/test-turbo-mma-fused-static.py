#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    f16 = read("ggml/src/ggml-cuda/fattn-mma-f16.cuh")
    fattn = read("ggml/src/ggml-cuda/fattn.cu")
    turbo = read("ggml/src/ggml-cuda/fattn-mma-turbo.cuh")
    generator = read("ggml/src/ggml-cuda/template-instances/generate_cu_files.py")
    instance = read("ggml/src/ggml-cuda/template-instances/fattn-mma-turbo-instance-dkq128-ncols1_8-ncols2_1.cu")
    kv_cache = read("src/llama-kv-cache.cpp")

    straight_types = [
        "GGML_TYPE_TURBO2_0",
        "GGML_TYPE_TURBO3_0",
        "GGML_TYPE_TURBO4_0",
    ]

    for ggml_type in straight_types:
        suffix = ggml_type.replace("GGML_TYPE_", "").lower()
        require(
            f"flash_attn_ext_{suffix}_load_tile" in f16,
            f"missing fused MMA tile loader for {ggml_type}",
        )
        require(
            f"type_K == {ggml_type}" in f16,
            f"missing fused MMA K load branch for {ggml_type}",
        )
        require(
            f"type_V == {ggml_type}" in f16,
            f"missing fused MMA V load branch for {ggml_type}",
        )
        require(
            f"TURBO_FUSED_DISPATCH({ggml_type},   {ggml_type})" in fattn,
            f"missing fused dispatch for {ggml_type}",
        )
        require(
            ggml_type in turbo,
            f"missing extern declaration for {ggml_type}",
        )
        require(
            f'("{ggml_type}", "{ggml_type}")' in generator,
            f"missing generated instance type for {ggml_type}",
        )
        require(
            f"{ggml_type}, {ggml_type}" in instance,
            f"missing generated instance declaration for {ggml_type}",
        )

    require(
        "TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO3_TCQ" not in fattn,
        "TCQ fused MMA dispatch must stay disabled until TCQ tile loaders are implemented",
    )
    require(
        "Unsupported turbo K type in fused MMA" in f16,
        "missing compile-time guard for unsupported fused MMA K types",
    )
    require(
        "Unsupported turbo V type in fused MMA" in f16,
        "missing compile-time guard for unsupported fused MMA V types",
    )
    require(
        "require all KV cache layers on GPU" in kv_cache,
        "turbo partial offload must fail before scheduler graph construction",
    )
    require(
        "turbo KV cache falling back to q8_0 for CPU-bound layers" not in kv_cache,
        "turbo partial offload must not silently create mixed q8/turbo KV layers",
    )


if __name__ == "__main__":
    main()
