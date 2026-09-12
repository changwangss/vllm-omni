# AutoRound Wan2.2 SVDQuant MXFP4

Load a Wan2.2 Diffusers pipeline exported with AutoRound's `svdquant_omni`
format. Both A14B experts (`transformer` and `transformer_2`) are supported.
No Nunchaku runtime is required. Each expert's `config.json` supplies the
quantization configuration automatically; do not pass `--quantization mxfp4`
(that selects a different quantization format).

## Hardware and installation

The same exported checkpoint is portable across B200 (SM100), B300 (SM103),
and RTX 5090/5090D (SM120). Runtime packing is performed after loading:

| Device | Native residual GEMM |
| --- | --- |
| B200 / B300 | vLLM `FlashInferMxFp4LinearKernel`, FlashInfer CuTe DSL MXFP4 |
| RTX Blackwell | vLLM `B12xMxFp4LinearKernel` |

Use the CUDA installation instructions for this Omni branch. The integration
was developed with vLLM 0.29.0, PyTorch 2.13.0+cu130, FlashInfer 0.6.18,
CuTe DSL 4.6.2, b12x 1.2.6, and Diffusers 0.40.0. On RTX Blackwell, install
`vllm[b12x]==0.29.0` and use a CUDA 13 toolkit. If multiple toolkits are
installed, set `CUDA_HOME` to the CUDA 13 installation before launching.
Keep quantization and inference in separate environments if their package
requirements differ.

Automatic backend selection uses b12x for SM120/SM121 and the vLLM registry
for SM100/SM103. An explicitly selected vLLM backend is preserved. Weight-only
backends that ignore activation quantization are rejected.

## Export on the quantization machine

Use AutoRound branch `wangchang/wan-svdquant-omni`. This data-free command
decomposes both experts with no smoothing and RTN residual quantization:

```bash
python scripts/quantize_wan_svdquant_nunchaku.py \
  --model /models/Wan2.2-T2V-A14B-Diffusers \
  --output /models/Wan2.2-A14B-svdquant-omni \
  --format svdquant_omni \
  --rank 32 --residual-iters 1 --devices cuda:0
```

For calibrated SVDQuant/SignRound, use
`scripts/quantize_wan_a14b_svdquant.py --format svdquant_omni` in that branch.
To convert an existing complete Wan Nunchaku pipeline without re-quantizing:

```bash
python scripts/convert_wan_svdquant_omni.py \
  --source /models/Wan2.2-A14B-nunchaku \
  --output /models/Wan2.2-A14B-svdquant-omni
```

Transfer the whole output directory, including the model index, both experts,
tokenizer, text encoder, VAE, and scheduler. A transformer-only conversion
must be placed into a complete Diffusers pipeline before using the command below.

## Inference

From Omni branch `wangchang/wan-svdquant-mxfp4`:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/offline_inference/text_to_video/text_to_video.py \
  --model /models/Wan2.2-A14B-svdquant-omni \
  --prompt "A cat walks through a sunny garden, cinematic lighting." \
  --height 384 --width 640 --num-frames 33 \
  --num-inference-steps 30 --seed 42 \
  --enable-cpu-offload --vae-use-tiling --enforce-eager \
  --output wan-a14b-svdquant.mp4
```

This starts with CPU offloading for the 32 GB RTX target. B200/B300 can omit
`--enable-cpu-offload` when sufficient memory is available. Keep the model's
boundary ratio so the denoising schedule uses both experts.

## Validation

Run this numerical test on each target machine, without downloading a model:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest \
  tests/diffusion/quantization/test_svdquant_mxfp4_native.py -q -s
```

It checks native W4A4 execution, integer packing and exponent scales,
runtime K padding, smoothing, low-rank correction, and bias against decoded
weights. CPU backend-selection tests alone do not validate a GPU kernel.
Local RTX testing does not establish B200/B300 execution or full A14B video
quality; run the numerical test and video command on the target hardware.

The implementation retains separate Q/K/V projections when `fuse_qkv=false`,
preserving independent smoothing and low-rank factors. The BF16 correction is
computed separately from the FP4 GEMM; this is not a fused Nunchaku kernel.
