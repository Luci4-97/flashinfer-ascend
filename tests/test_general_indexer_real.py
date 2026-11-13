"""Real-NPU edge acceptance for attention.general_indexer."""

import pytest
import torch

from flashinfer.operators.attention.general_indexer.cases import golden
from flashinfer.operators.runtime import public_function

pytestmark = pytest.mark.real_compile


def _device(tensor):
    import torch_npu
    return torch_npu.npu_format_cast(tensor.npu(), 2)


def _call(query, key, weights, topk, **options):
    from flashinfer.aclnn import enable_aclnn
    enable_aclnn()
    options = {
        name: (_device(value) if isinstance(value, torch.Tensor) else value)
        for name, value in options.items()
    }
    return public_function("general_indexer")(
        _device(query), _device(key), _device(weights), topk,
        backend="ascendc", **options,
    )


def _assert_outputs(actual, expected):
    assert len(actual) == len(expected) == 3
    tolerances = ((0, 0), (0.02, 0.02), (0.02, 0.02))
    for received, reference, (atol, rtol) in zip(actual, expected, tolerances):
        torch.testing.assert_close(
            received.cpu().double(), reference.double(), atol=atol, rtol=rtol
        )


def test_tnd_uneven_non_aligned_mask_seqused():
    torch.manual_seed(901)
    ql, kl, n1, d, topk = [5, 8], [16, 24], 4, 16, 4
    tq, tk = sum(ql), sum(kl)
    query = torch.randn(tq, n1, d, dtype=torch.float16)
    key = torch.randn(tk, d, dtype=torch.float16)
    weights = torch.randn(tq, n1, dtype=torch.float16)
    cu_q = torch.tensor([0, ql[0], tq], dtype=torch.int32)
    cu_k = torch.tensor([0, kl[0], tk], dtype=torch.int32)
    seq_q = torch.tensor([4, 7], dtype=torch.int32)
    seq_k = torch.tensor([15, 20], dtype=torch.int32)
    actual = _call(
        query, key, weights, topk, layout_q="TND", layout_k="TND",
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        seqused_q=seq_q, seqused_k=seq_k, mask_mode=3,
        return_value=True, return_score=True,
    )
    expected = golden(
        torch, query, key, weights, topk, layout_q="TND", layout_k="TND",
        cu_q=cu_q, cu_k=cu_k, seqused_q=seq_q, seqused_k=seq_k,
        mask_mode=3, return_value=True, return_score=True,
    )
    _assert_outputs(actual, expected)


def test_paged_scrambled_tail_and_illegal_page():
    torch.manual_seed(902)
    s1, n1, s2, d, topk, block = 8, 2, 40, 32, 8, 16
    query = torch.randn(s1, n1, d, dtype=torch.float16)
    key = torch.randn(s2, d, dtype=torch.float16)
    weights = torch.randn(s1, n1, dtype=torch.float16)
    pages_count = (s2 + block - 1) // block
    pages = torch.zeros(pages_count, block, d, dtype=torch.float16)
    table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    for logical in range(pages_count):
        lo = logical * block
        hi = min(s2, lo + block)
        pages[table[0, logical], :hi - lo] = key[lo:hi]
    # Drop the middle logical page; kernel must treat it as zero keys.
    table[0, 1] = -1
    seqused = torch.tensor([s2], dtype=torch.int32)
    actual = _call(
        query, pages, weights, topk, layout_k="PA_BBND",
        block_table=table, seqused_k=seqused, max_seq_len_k=s2,
        return_value=True, return_score=True,
    )
    expected = golden(
        torch, query, pages, weights, topk, layout_k="PA_BBND",
        block_table=table, seqused_k=seqused, max_seq_len_k=s2,
        return_value=True, return_score=True,
    )
    _assert_outputs(actual, expected)


def test_large_s2_and_stable_tie_break():
    torch.manual_seed(903)
    s1, n1, s2, d, topk = 8, 2, 8448, 16, 8
    query = torch.ones(s1, n1, d, dtype=torch.float16)
    key = torch.randn(s2, d, dtype=torch.float16)
    key[128:512] = key[128:129].expand(384, -1)
    weights = torch.ones(s1, n1, dtype=torch.float16)
    actual = _call(query, key, weights, topk, return_value=True, return_score=True)
    expected = golden(
        torch, query, key, weights, topk,
        return_value=True, return_score=True,
    )
    _assert_outputs(actual, expected)
    rows = actual[0].cpu()
    scores = actual[2].cpu()
    for row in rows:
        values = scores[0, row.long()]
        for left in range(topk - 1):
            if values[left] == values[left + 1]:
                assert row[left] < row[left + 1]


def test_topk_upper_bound_and_d64_n64():
    torch.manual_seed(904)
    s1, n1, s2, d, topk = 4, 1, 8192, 16, 8192
    query = torch.randn(s1, n1, d, dtype=torch.float16)
    key = torch.randn(s2, d, dtype=torch.float16)
    weights = torch.randn(s1, n1, dtype=torch.float16)
    actual = _call(query, key, weights, topk, return_value=True, return_score=False)
    expected = golden(torch, query, key, weights, topk, return_value=True, return_score=False)
    assert actual[0].shape == expected[0].shape
    assert torch.equal(actual[2].cpu(), expected[2])
    actual_sorted = actual[1].cpu().double().sort(dim=1).values
    expected_sorted = expected[1].double().sort(dim=1).values
    torch.testing.assert_close(actual_sorted, expected_sorted, atol=0.02, rtol=0.02)
    for row in actual[0].cpu():
        assert torch.equal(row.sort().values, torch.arange(8192, dtype=torch.int32))

    torch.manual_seed(905)
    s1, n1, s2, d, topk = 8, 64, 32, 64, 16
    query = torch.randn(s1, n1, d, dtype=torch.bfloat16) * 0.1
    key = torch.randn(s2, d, dtype=torch.bfloat16) * 0.1
    weights = torch.randn(s1, n1, dtype=torch.float32)
    actual = _call(query, key, weights, topk, return_value=True, return_score=True)
    expected = golden(
        torch, query, key, weights, topk,
        return_value=True, return_score=True,
    )
    _assert_outputs(actual, expected)
