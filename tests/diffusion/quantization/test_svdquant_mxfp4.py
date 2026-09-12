# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic

from vllm_omni.quantization import svdquant_config as svdquant
from vllm_omni.quantization.factory import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_mxfp4_checkpoint_config():
    config = build_quant_config({"quant_method": "svdquant", "precision": "mxfp4", "fuse_qkv": False})
    assert config.precision == "mxfp4"
    assert config.fuse_qkv is False
    assert svdquant.DiffusionSVDQuantConfig().fuse_qkv is True
    with pytest.raises(ValueError, match="fuse_qkv"):
        svdquant.DiffusionSVDQuantConfig(fuse_qkv="false")


def test_mxfp4_selects_dynamic_activation_kernel(monkeypatch):
    from vllm.model_executor.kernels.linear.mxfp4.flashinfer import FlashInferMxFp4LinearKernel

    factory = Mock(return_value=Mock(spec=FlashInferMxFp4LinearKernel))
    monkeypatch.setattr(svdquant, "init_mxfp4_linear_kernel", factory)
    assert svdquant._mxfp4_kernel() is factory.return_value
    factory.assert_called_once_with(activation_quant_key=kMxfp4Dynamic)


def test_mxfp4_rejects_weight_only_backend(monkeypatch):
    monkeypatch.setattr(svdquant, "init_mxfp4_linear_kernel", Mock(return_value=object()))
    with pytest.raises(RuntimeError, match="native W4A4"):
        svdquant._mxfp4_kernel()


@pytest.mark.parametrize(
    ("sm", "requested", "expected"),
    [
        (100, "auto", "auto"),
        (103, "auto", "auto"),
        (120, "auto", "b12x"),
        (121, "auto", "b12x"),
        (100, "flashinfer_cutlass", "flashinfer_cutlass"),
        (120, "flashinfer_cutlass", "flashinfer_cutlass"),
    ],
)
def test_blackwell_backend_selection_preserves_caller_config(monkeypatch, sm, requested, expected):
    from vllm.config import get_current_vllm_config, set_current_vllm_config
    from vllm.model_executor.kernels.linear.mxfp4.flashinfer import FlashInferMxFp4LinearKernel

    config = SimpleNamespace(kernel_config=SimpleNamespace(linear_backend=requested))
    selected = []

    def factory(**kwargs):
        selected.append(get_current_vllm_config().kernel_config.linear_backend)
        return Mock(spec=FlashInferMxFp4LinearKernel)

    monkeypatch.setattr(
        svdquant.current_platform, "is_device_capability_family", lambda family: sm // 10 == family // 10
    )
    monkeypatch.setattr(svdquant, "init_mxfp4_linear_kernel", factory)
    with set_current_vllm_config(config):
        svdquant._mxfp4_kernel()
        assert get_current_vllm_config() is config
    assert selected == [expected]
    assert config.kernel_config.linear_backend == requested


def test_mxfp4_layout_and_correction(monkeypatch):
    monkeypatch.setattr(svdquant, "_assert_mxfp4_supported", lambda: None)
    method = svdquant.DiffusionSVDQuantLinearMethod(svdquant.DiffusionSVDQuantConfig(precision="mxfp4", rank=4))
    layer = torch.nn.Module()
    method.create_weights(layer, 128, [64], 128, 64, torch.bfloat16)
    assert layer.qweight.shape == (64, 64)
    assert layer.wscales.shape == (4, 64)
    assert layer.wscales.dtype == torch.uint8
    assert layer.wscales.input_dim == 0 and layer.wscales.output_dim == 1
    assert not hasattr(layer, "wtscale")
    torch.manual_seed(7)
    with torch.no_grad():
        layer.qweight.random_(-128, 127)
        layer.wscales.random_(120, 135)
        layer.smooth_factor.copy_(torch.linspace(0.5, 2, 128))
        layer.proj_down.normal_(std=0.01)
        layer.proj_up.normal_(std=0.01)
    raw_weight = layer.qweight.detach().clone()
    raw_scale = layer.wscales.detach().clone()
    kernel = Mock()
    base_weight = torch.randn(128, 64, dtype=torch.bfloat16) * 0.01
    kernel.apply_weights.side_effect = lambda *, layer, x, bias: x @ base_weight
    factory = Mock(return_value=kernel)
    monkeypatch.setattr(svdquant, "_mxfp4_kernel", factory)
    method.process_weights_after_loading(layer)
    assert torch.equal(layer.weight.view(torch.int8), raw_weight)
    assert torch.equal(layer.weight_scale, raw_scale.T)
    assert layer.weight_scale.dtype == torch.uint8
    kernel.process_weights_after_loading.assert_called_once_with(layer)
    x = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    bias = torch.randn(64, dtype=torch.bfloat16)
    actual = method.apply(layer, x, bias)
    x2d = x.reshape(-1, 128)
    expected = torch.addmm((x2d / layer.smooth_factor) @ base_weight, x2d @ layer.proj_down, layer.proj_up.T)
    expected += bias
    torch.testing.assert_close(actual, expected.reshape(2, 3, 64), rtol=0, atol=0)
    assert kernel.apply_weights.call_args.kwargs["bias"] is None
    factory.assert_called_once()  # Forward must use the backend that repacked this layer.


def test_mxfp4_requires_block32(monkeypatch):
    monkeypatch.setattr(svdquant, "_assert_mxfp4_supported", lambda: None)
    method = svdquant.DiffusionSVDQuantLinearMethod(svdquant.DiffusionSVDQuantConfig(precision="mxfp4"))
    with pytest.raises(ValueError, match="block size 32"):
        method.create_weights(torch.nn.Module(), 48, [64], 48, 64, torch.bfloat16)


@pytest.mark.parametrize("axis", ["row", "column"])
def test_mxfp4_tensor_parallel_loads_actual_shards(monkeypatch, axis):
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

    monkeypatch.setattr(svdquant, "_assert_mxfp4_supported", lambda: None)
    linear_type = RowParallelLinear if axis == "row" else ColumnParallelLinear
    layer = object.__new__(linear_type)
    torch.nn.Module.__init__(layer)
    layer.tp_rank = 1
    method = svdquant.DiffusionSVDQuantLinearMethod(svdquant.DiffusionSVDQuantConfig(precision="mxfp4", rank=8))
    method.create_weights(
        layer,
        64 if axis == "row" else 128,
        [128 if axis == "row" else 64],
        128,
        128,
        torch.bfloat16,
        weight_loader=layer.weight_loader,
    )
    shapes = {
        "qweight": (128, 64),
        "wscales": (4, 128),
        "proj_down": (128, 8),
        "proj_up": (128, 8),
        "smooth_factor": (128,),
    }
    for name, shape in shapes.items():
        param = getattr(layer, name)
        source = torch.arange(torch.Size(shape).numel()).remainder(123).reshape(shape).to(param.dtype)
        param.weight_loader(param, source)
        dim = getattr(param, "input_dim" if axis == "row" else "output_dim", None)
        expected = source if dim is None else source.narrow(dim, param.shape[dim], param.shape[dim])
        torch.testing.assert_close(param, expected)
