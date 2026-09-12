# SVDQuant W4A4

## Overview

[SVDQuant](https://arxiv.org/abs/2411.05007) combines four-bit weights
and activations with a small low-rank branch that corrects part of the
quantization error. vLLM-Omni consumes an offline-quantized checkpoint;
it does not calibrate the model while loading.

## NVFP4 checkpoint contract

Place the following entry in the diffusion transformer's `config.json`:

```json
{
  "quantization_config": {
    "quant_method": "svdquant",
    "rank": 32,
    "precision": "nvfp4",
    "act_unsigned": false,
    "modules_to_not_convert": []
  }
}
```

For a quantized linear with input size `K`, output size `N`, and correction
rank `R`, the checkpoint stores:

| Suffix | Shape | dtype |
| --- | --- | --- |
| `qweight` | `(N, K / 2)` | `int8` (two packed FP4 values per byte) |
| `wscales` | `(K / 16, N)` | `float8_e4m3fn` |
| `proj_down` | `(K, R)` | `bfloat16` |
| `proj_up` | `(N, R)` | `bfloat16` |
| `smooth_factor` | `(K,)` | `bfloat16` |
| `wcscales` | `(N,)` | `bfloat16` |
| `wtscale` | `(1,)` | `bfloat16` |

`K` must be divisible by 16 on every tensor-parallel rank. Set `wcscales` to
ones when no per-output correction is needed. Modules listed in
`modules_to_not_convert` keep their checkpoint precision.

## MXFP4 checkpoint contract

AutoRound's `svdquant_omni` format exports Wan2.2 with `precision="mxfp4"`
and `fuse_qkv=false`. It uses the same tensor orientations as above, with
these differences:

- `qweight` stores standard E2M1 values, low nibble first.
- `wscales` has shape `(K / 32, N)` and dtype `uint8`, holding raw UE8M0
  exponent bytes. Do not numerically cast the bytes to floating point.
- `wcscales` and `wtscale` are absent; block scales fully describe the residual.
- Every tensor-parallel input partition must be divisible by 32.

The residual GEMM receives `x / smooth_factor`. The BF16 correction receives
the original `x`: `(x @ proj_down) @ proj_up.T`. Bias is added once after
summing both branches. Independently decomposed Wan Q/K/V retain separate
projections, preserving their different smoothing and low-rank factors.

Each A14B expert supplies its own quantization metadata in `config.json`.
The pipeline auto-detects it, including `transformer_2`. See the
[Wan export and inference commands](../../../examples/offline_inference/text_to_video/README_SVDQUANT_MXFP4.md).

## Runtime support

The compatibility path accepts BF16 inputs and executes an NVFP4 GEMM followed
by the BF16 rank correction. It supports vLLM's FlashInfer, CUTLASS, and FBGEMM
NVFP4 tensor layouts; incompatible forced backends fail during model loading.
For NVFP4, SM103 is the currently validated and enabled hardware target. Native fusion of
the NVFP4 GEMM and rank correction is separate from this checkpoint-loading
contract.

MXFP4 uses vLLM's FlashInfer CuTe DSL path on B200/B300 (SM100/SM103), and
the native b12x backend on RTX Blackwell (SM120/SM121). Runtime packing is
device-specific; the disk checkpoint is shared. b12x input padding is applied
only to the residual GEMM, leaving the original low-rank factors intact.
Backends that silently ignore activation quantization are rejected.

The native numerical smoke test is
`tests/diffusion/quantization/test_svdquant_mxfp4_native.py`. Run it on every
target architecture. Local RTX execution and mocked B200/B300 selection tests
do not establish B200/B300 execution or full-model video quality.
