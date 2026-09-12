# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.wan2_2 import wan2_2_transformer as wan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("fused", [True, False])
def test_wan_loads_qkv_payloads_without_prefix_collision(monkeypatch, fused):
    monkeypatch.setattr(wan, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(wan, "get_tensor_model_parallel_world_size", lambda: 1)
    model = object.__new__(wan.WanTransformer3DModel)
    torch.nn.Module.__init__(model)
    model.fuse_qkv = fused
    block = torch.nn.Module()
    block.attn1 = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([block])
    state = {}
    for offset, projection in enumerate(("to_qkv",) if fused else ("to_q", "to_k", "to_v"), 1):
        layer = torch.nn.Module()
        setattr(block.attn1, projection, layer)
        for name, shape, dtype in [
            ("qweight", (128, 64), torch.int8),
            ("wscales", (4, 128), torch.uint8),
            ("proj_down", (128, 16), torch.bfloat16),
            ("proj_up", (128, 16), torch.bfloat16),
            ("smooth_factor", (128,), torch.bfloat16),
            ("bias", (128,), torch.bfloat16),
        ]:
            layer.register_parameter(name, torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False))
            state[f"blocks.0.attn1.{projection}.{name}"] = torch.full(shape, offset, dtype=dtype)
    loaded = model.load_weights(state.items())
    assert set(state) <= loaded
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, state[name])


def test_wan_still_fuses_separate_dense_qkv(monkeypatch):
    from vllm.model_executor.layers.linear import QKVParallelLinear

    monkeypatch.setattr(wan, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(wan, "get_tensor_model_parallel_world_size", lambda: 1)
    model = object.__new__(wan.WanTransformer3DModel)
    torch.nn.Module.__init__(model)
    model.fuse_qkv = True
    block = torch.nn.Module()
    block.attn1 = torch.nn.Module()
    layer = object.__new__(QKVParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.tp_rank = 0
    layer.total_num_heads = layer.total_num_kv_heads = layer.num_heads = layer.num_kv_heads = 2
    layer.num_kv_head_replicas = 1
    layer.head_size = layer.v_head_size = 8
    weight = torch.nn.Parameter(torch.zeros(48, 16), requires_grad=False)
    weight.output_dim = 0
    weight.weight_loader = layer.weight_loader
    layer.register_parameter("weight", weight)
    block.attn1.to_qkv = layer
    model.blocks = torch.nn.ModuleList([block])
    tensors = [
        (f"blocks.0.attn1.to_{name}.weight", torch.full((16, 16), float(i)))
        for i, name in enumerate(("q", "k", "v"), 1)
    ]
    model.load_weights(tensors)
    torch.testing.assert_close(layer.weight, torch.cat([value for _, value in tensors]))


@pytest.mark.parametrize("preserve", [True, False])
def test_wan_preserves_protected_fp32_svdquant_weights(monkeypatch, preserve):
    monkeypatch.setattr(wan, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(wan, "get_tensor_model_parallel_world_size", lambda: 1)
    model = object.__new__(wan.WanTransformer3DModel)
    torch.nn.Module.__init__(model)
    model.fuse_qkv = False
    model.preserve_svdquant_fp32 = preserve
    model.condition_embedder = torch.nn.Module()
    model.condition_embedder.time_embedder = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    name = "condition_embedder.time_embedder.weight"
    source = torch.full((4, 4), 1.000123, dtype=torch.float32)
    model.load_weights([(name, source)])
    actual = dict(model.named_parameters())[name]
    assert actual.dtype == (torch.float32 if preserve else torch.bfloat16)
    torch.testing.assert_close(actual, source.to(actual.dtype), atol=0, rtol=0)
