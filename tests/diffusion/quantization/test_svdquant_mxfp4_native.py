# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native W4A4 numerical smoke for B200/B300 and RTX Blackwell.

Run on each target GPU; CPU tests of backend selection do not replace this test.
No model download or AutoRound/Nunchaku installation is required.
"""

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_omni.quantization.svdquant_config import DiffusionSVDQuantConfig, DiffusionSVDQuantLinearMethod

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA Blackwell and a native MXFP4 backend")
@pytest.mark.parametrize("k", [256, 320])
@torch.inference_mode()
def test_native_mxfp4_with_low_rank_matches_decoded_weights(k):
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3), (12, 0), (12, 1)}:
        pytest.skip("Requires B200/B300 or RTX Blackwell")
    torch.manual_seed(23)
    n, rank = 256, 16
    config = VllmConfig()
    with set_current_vllm_config(config):
        method = DiffusionSVDQuantLinearMethod(DiffusionSVDQuantConfig(precision="mxfp4", rank=rank))
        layer = torch.nn.Module()
        with torch.device("cuda"):
            method.create_weights(layer, k, [n], k, n, torch.bfloat16)
        values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
        codes = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
        layer.qweight.copy_((codes[:, ::2] | (codes[:, 1::2] << 4)).view(torch.int8))
        layer.wscales.random_(120, 126)
        weight = (values[codes.long()] * torch.exp2(layer.wscales.T.float() - 127).repeat_interleave(32, -1)).bfloat16()
        activation_codes = torch.randint(0, 16, (17, k), device="cuda")
        activation_codes[:, ::32] = 7  # Each group's amax is 6: scale is exactly 1.
        activation = values[activation_codes].bfloat16()
        layer.smooth_factor.fill_(2)
        x = activation * layer.smooth_factor
        layer.proj_down.normal_(std=0.01)
        layer.proj_up.normal_(std=0.01)
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.01
        expected = torch.addmm(activation @ weight.T, x @ layer.proj_down, layer.proj_up.T) + bias
        method.process_weights_after_loading(layer)
        actual = method.apply(layer, x, bias)
        torch.accelerator.synchronize()
        relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert actual.isfinite().all()
        assert relative_l2 < 0.005
        print(
            f"{torch.cuda.get_device_name()}: {type(layer.svdquant_kernel).__name__}, K={k}, relative_L2={relative_l2.item()}"
        )
