"""Public contract tests for attention.general_indexer."""

from types import SimpleNamespace

import pytest

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
