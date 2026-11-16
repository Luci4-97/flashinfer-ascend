"""Public contract tests for attention.general_indexer."""

import os

import pytest
import torch

from flashinfer.ascendc.attention.general_indexer.implementation import IMPLEMENTATION
from flashinfer.operators.attention.general_indexer.operator import OPERATOR


class Tensor:
    def __init__(self, shape, dtype="float16", device="npu:0"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device


def args(*, layout_q="SND", layout_k="SND", return_value=False, return_score=False,
         query=None, key=None, weights=None, block_table=None, cu_q=None, cu_k=None,
         seqused_q=None, seqused_k=None, cmp_residual=None, offset=None,
         mask_mode=0, cmp_ratio=1, max_seq_len_k=0, k=4):
    if query is None:
        query = Tensor((8, 2, 16))
    if key is None:
        key = Tensor((16, 16))
    if weights is None:
        weights = Tensor((8, 2))
    return dict(
        query=query, key=key, weights=weights, k=k,
        layout_q=layout_q, layout_k=layout_k, mask_mode=mask_mode,
        cmp_ratio=cmp_ratio, max_seq_len_k=max_seq_len_k, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        seqused_q=seqused_q, seqused_k=seqused_k,
        cmp_residual_k=cmp_residual, block_table=block_table,
        output_idx_offset=offset, return_value=return_value,
        return_score=return_score,
    )


def infer(**kwargs):
    return OPERATOR.infer(args(**kwargs), None)


def test_flat_contract_keeps_fixed_three_outputs():
    invocation = infer()
    assert [o.shape for o in invocation.outputs] == [(8, 4), (0,), (0,)]
    assert [o.dtype for o in invocation.outputs] == ["int32", "float32", "float32"]


def test_flat_optional_outputs_have_logical_shapes():
    invocation = infer(return_value=True, return_score=True)
    assert [o.shape for o in invocation.outputs] == [(8, 4), (8, 4), (8, 16)]


def test_bsnd_outputs_are_packed_to_li_shapes():
    invocation = infer(
        layout_q="BSND", layout_k="BSND", return_value=True, return_score=True,
        query=Tensor((2, 8, 2, 16)), key=Tensor((2, 16, 1, 16)),
        weights=Tensor((2, 8, 2)),
    )
    assert [o.shape for o in invocation.outputs] == [(2, 8, 1, 4), (2, 8, 1, 4), (2, 8, 16)]


def test_paged_key_uses_seqused_and_block_table():
    invocation = infer(
        layout_k="PA_BBND", return_score=True,
        key=Tensor((4, 16, 16)), block_table=Tensor((1, 4), "int32"),
        seqused_k=Tensor((1,), "int32"), max_seq_len_k=48,
    )
    assert invocation.outputs[0].shape == (8, 4)
    assert invocation.outputs[2].shape == (8, 48)


@pytest.mark.parametrize("change,match", [
    (dict(layout_q="HND"), "layout_q"),
    (dict(layout_q="BSND", layout_k="SND"), "both"),
    (dict(mask_mode=2), "mask_mode"),
    (dict(cmp_ratio=0), "cmp_ratio"),
    (dict(k=0), "k must"),
    (dict(layout_k="PA_BBND", return_score=True, key=Tensor((4, 16, 16)),
          block_table=Tensor((1, 4), "int32"), seqused_k=Tensor((1,), "int32")),
     "max_seq_len_k"),
    (dict(query=Tensor((8, 2, 12))), "head dim"),
    (dict(query=Tensor((8, 257, 16)), weights=Tensor((8, 257))), "N1"),
])
def test_invalid_shape_or_attribute_is_rejected(change, match):
    with pytest.raises(ValueError, match=match):
        infer(**change)


def test_native_recipe_declares_full_ordered_abi():
    from flashinfer.ascendc.attention.general_indexer.implementation import NATIVE
    from flashinfer.jit.recipe import CATLASS_REVISION

    assert NATIVE.kernel_id == (
        "ascendc/attention/general_indexer/catlass/op_package/Ascend950"
    )
    assert [item.name for item in NATIVE.tensors] == [
        "query", "key", "weights", "block_table", "cu_seqlens_q",
        "cu_seqlens_k", "seqused_q", "seqused_k", "cmp_residual_k",
        "output_idx_offset", "score_zero_page", "indices", "values", "scores",
    ]
    assert len(NATIVE.dtype_signatures) == 6
    assert NATIVE.build.dependencies[0].revision == CATLASS_REVISION
    assert NATIVE.allow_jit_fallback is False


def test_top_level_export_is_installed():
    import flashinfer

    assert callable(flashinfer.general_indexer)


def test_cpu_golden_is_stable_and_pads_values():
    import torch

    from flashinfer.operators.attention.general_indexer.cases import golden

    q = torch.tensor([[[1.0], [1.0]], [[2.0], [1.0]]])
    k = torch.tensor([[1.0], [2.0], [3.0]])
    w = torch.ones((2, 2))
    indices, values, scores = golden(
        torch, q, k, w, 2, return_value=True, return_score=True
    )
    assert indices.tolist() == [[2, 1], [2, 1]]
    assert scores.shape == (2, 3)
    assert torch.equal(values, torch.gather(scores, 1, indices.long()))


def test_paged_requires_explicit_max_seq_len():
    with pytest.raises(ValueError, match="max_seq_len_k"):
        infer(
            layout_k="PA_BBND", key=Tensor((4, 16, 16)),
            block_table=Tensor((1, 4), "int32"),
            seqused_k=Tensor((1,), "int32"),
        )


def test_flat_seqused_q_without_cu_seqlens_is_accepted():
    arguments = args(seqused_q=torch.tensor([5], dtype=torch.int32))
    OPERATOR.validate_values(arguments)


@pytest.mark.parametrize("change", [
    dict(k=8, key=Tensor((4, 16))),
    dict(
        layout_q="TND", layout_k="TND", k=4,
        key=Tensor((32, 16)),
        cu_q=torch.tensor([0, 4, 8], dtype=torch.int32),
        cu_k=torch.tensor([0, 2, 32], dtype=torch.int32),
    ),
    dict(
        layout_q="BSND", layout_k="BSND", k=8,
        query=Tensor((2, 4, 2, 16)), key=Tensor((2, 4, 16)),
        weights=Tensor((2, 4, 2)),
    ),
])
def test_k_must_not_exceed_any_request_key_length(change):
    with pytest.raises(ValueError, match="k must not exceed"):
        OPERATOR.validate_values(args(**change))


def _torch_arguments(*, layout_q="SND", layout_k="SND", dtype=None, return_value=False,
                     return_score=False, paged=False):
    import torch
    dtype = dtype or torch.float16
    if layout_q == "BSND":
        query = torch.zeros(2, 8, 2, 16, dtype=dtype)
        weights = torch.zeros(2, 8, 2, dtype=dtype)
        key = torch.zeros(4, 16, 16, dtype=dtype) if paged else torch.zeros(2, 32, 16, dtype=dtype)
        kwargs = dict(layout_q=layout_q, layout_k=layout_k)
    else:
        query = torch.zeros(8, 2, 16, dtype=dtype)
        weights = torch.zeros(8, 2, dtype=dtype)
        key = torch.zeros(2, 16, 16, dtype=dtype) if paged else torch.zeros(32, 16, dtype=dtype)
        kwargs = dict(layout_q=layout_q, layout_k=layout_k)
    if paged:
        kwargs.update(block_table=torch.zeros(2 if layout_q == "BSND" else 1, 2, dtype=torch.int32),
                      seqused_k=torch.full((2 if layout_q == "BSND" else 1,), 32, dtype=torch.int32),
                      max_seq_len_k=32)
    elif layout_q == "TND":
        kwargs.update(cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32),
                      cu_seqlens_k=torch.tensor([0, 32], dtype=torch.int32))
    return OPERATOR.bind(query, key, weights, 4, return_value=return_value,
                        return_score=return_score, **kwargs)


@pytest.mark.parametrize("layout_q,layout_k,paged", [
    ("SND", "SND", False), ("TND", "TND", False), ("BSND", "BSND", False),
    ("SND", "PA_BBND", True), ("BSND", "PA_BBND", True),
])
def test_fake_metadata_and_dispatch_cover_all_layouts(layout_q, layout_k, paged):
    from flashinfer.operators.runtime import fake, signature
    from flashinfer.runtime import HardwareSignature, dispatch

    args = _torch_arguments(layout_q=layout_q, layout_k=layout_k, paged=paged,
                            return_value=True, return_score=True)
    outputs = fake(OPERATOR.op, args)
    assert len(outputs) == 3
    request, _, _ = signature(OPERATOR.op, args)
    decision = dispatch(request, HardwareSignature("Ascend950", 28))
    assert decision.kernel_id == (
        "ascendc/attention/general_indexer/catlass/op_package/Ascend950"
    )


def test_paged_4d_key_dispatches_exactly():
    from flashinfer.operators.runtime import fake, signature
    from flashinfer.runtime import HardwareSignature, dispatch

    args = _torch_arguments(layout_q="SND", layout_k="PA_BBND", paged=True,
                            return_value=True, return_score=True)
    args["key"] = args["key"].unsqueeze(2)
    outputs = fake(OPERATOR.op, args)
    assert outputs[0].shape == (8, 4)
    request, _, _ = signature(OPERATOR.op, args)
    decision = dispatch(request, HardwareSignature("Ascend950", 28))
    assert decision.kernel_id == (
        "ascendc/attention/general_indexer/catlass/op_package/Ascend950"
    )


def test_fake_rejects_invalid_contract_and_out_alias():
    import torch

    from flashinfer.operators.runtime import fake, signature

    args = _torch_arguments()
    with pytest.raises(ValueError, match="mask_mode"):
        fake(OPERATOR.op, dict(args, mask_mode=1))
    outputs = fake(OPERATOR.op, args)
    overlap = args["query"].view(torch.int32).reshape(8, 16)[:, :4]
    with pytest.raises(ValueError, match="overlap"):
        signature(OPERATOR.op, dict(args, out=(overlap, *outputs[1:])))
    with pytest.raises(ValueError, match="out shape/dtype"):
        fake(OPERATOR.op, dict(args, out=(outputs[0], torch.zeros(8, 5), outputs[2])))


def test_runner_adapts_disabled_outputs_to_hidden_native_buffers():
    from flashinfer.ascendc.attention.general_indexer.runner import GeneralIndexerRunner

    args = _torch_arguments()
    invocation = OPERATOR.infer(args, None)
    outputs = tuple(torch.empty(s.shape, dtype=getattr(torch, s.dtype))
                    for s in invocation.outputs)
    observed = []
    runner = GeneralIndexerRunner(lambda *tensors, **attrs: observed.append((tensors, attrs)))
    runner(*invocation.inputs, *outputs, **invocation.attributes)
    tensors, attrs = observed[0]
    assert len(tensors) == 14
    assert tensors[11].shape == (8, 4) and tensors[11].data_ptr() != outputs[1].data_ptr()
    assert tensors[12].shape == (8, 4) and tensors[12].data_ptr() != outputs[1].data_ptr()
    assert tensors[13].shape == (8, 32) and tensors[13].data_ptr() != outputs[2].data_ptr()
    assert tensors[0] is args["query"]
    assert attrs["is_paged"] == 0 and attrs["k"] == 4


def test_fake_torch_registration_returns_fixed_three_outputs():
    from torch._subclasses.fake_tensor import FakeTensorMode

    import flashinfer

    with FakeTensorMode():
        q = torch.empty(8, 2, 16, dtype=torch.float16)
        k = torch.empty(32, 16, dtype=torch.float16)
        w = torch.empty(8, 2, dtype=torch.float16)
        outputs = flashinfer.general_indexer(q, k, w, 4)
        assert outputs[0].shape == (8, 4) and outputs[0].dtype == torch.int32
        assert outputs[1].shape == outputs[2].shape == (0,)
        supplied = tuple(torch.empty_like(t) for t in outputs)
        returned = flashinfer.general_indexer(q, k, w, 4, out=supplied)
        assert all(actual is expected for actual, expected in zip(returned, supplied))


def test_prepared_recipe_is_aot_only_pinned_and_not_capturable():
    from pathlib import Path

    from flashinfer.jit.recipe import CATLASS_REVISION
    from flashinfer.runtime import GraphCapability

    root = Path(__file__).resolve().parents[1]
    prepared = IMPLEMENTATION.prepare_outputs(root)
    descriptor, recipe = prepared["descriptor"], prepared["recipe"]
    assert descriptor.graph_capability == GraphCapability.NOT_CAPTURABLE
    assert descriptor.capability_binding.domain == "operator"
    assert recipe.allow_jit_fallback is False
    assert recipe.target_soc == "Ascend950"
    assert recipe.package_op_type == "FiGeneralIndexer"
    assert recipe.dependencies[0].revision == CATLASS_REVISION
    assert len(recipe.tensor_parameters) == 14 and len(recipe.attributes) == 5


def test_production_inventory_declares_general_indexer_exactly():
    from tests.production_inventory import (
        DEFAULT_DESCRIPTOR_COUNT,
        PRODUCTION,
        PRODUCTION_KERNEL_IDS,
        PRODUCTION_OPS,
        STACK_COUNTS,
    )

    kernel_id = "ascendc/attention/general_indexer/catlass/op_package/Ascend950"
    assert kernel_id in PRODUCTION_KERNEL_IDS
    assert "attention.general_indexer" in PRODUCTION_OPS
    assert any(row[0] == kernel_id and row[1:] == (
        "attention.general_indexer", "general_indexer", "catlass", "Ascend950"
    ) for row in PRODUCTION)
    assert len(PRODUCTION) == STACK_COUNTS["ascendc"] == 10
    assert DEFAULT_DESCRIPTOR_COUNT == 91


def test_real_release_environment_is_process_local(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path

    import flashinfer.operators.pytest_support as support

    test_file = Path(__file__).with_name("test_general_indexer_real.py")
    spec = importlib.util.spec_from_file_location("_general_indexer_real_contract", test_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        support,
        "prepare_release",
        lambda *_args: {
            "FLASHINFER_AOT_PATHS": str(tmp_path / "release"),
            "FLASHINFER_ACLNN_ADAPTER_PATH": str(tmp_path / "adapter"),
        },
    )
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "0")

    class Factory:
        def mktemp(self, name):
            return tmp_path / name

    release = module.general_indexer_environment.__wrapped__(Factory())
    assert release["FLASHINFER_AOT_PATHS"] == str(tmp_path / "release")
    assert release["FLASHINFER_ACLNN_ADAPTER_PATH"] == str(tmp_path / "adapter")
    assert release["FLASHINFER_NPU_JIT_DISABLE"] == "1"
    assert os.environ["FLASHINFER_NPU_JIT_DISABLE"] == "0"
    assert "FLASHINFER_AOT_PATHS" not in os.environ
