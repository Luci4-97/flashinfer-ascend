"""Subprocess acceptance cases for attention.general_indexer."""

import argparse

import torch

from flashinfer.operators.attention.general_indexer.cases import golden
from flashinfer.operators.runtime import public_function


def _device(tensor):
    import torch_npu

    return torch_npu.npu_format_cast(tensor.npu(), 2)


def _call(query, key, weights, topk, *, backend="ascendc", out=None, **options):
    from flashinfer.aclnn import enable_aclnn

    enable_aclnn()
    options = {
        name: (_device(value) if isinstance(value, torch.Tensor) else value)
        for name, value in options.items()
    }
    if backend is not None:
        options["backend"] = backend
    if out is not None:
        options["out"] = out
    return public_function("general_indexer")(
        _device(query), _device(key), _device(weights), topk, **options,
    )


def _assert_outputs(actual, expected):
    assert len(actual) == len(expected) == 3
    tolerances = ((0, 0), (0.02, 0.02), (0.02, 0.02))
    for received, reference, (atol, rtol) in zip(actual, expected, tolerances):
        torch.testing.assert_close(
            received.cpu().double(), reference.double(), atol=atol, rtol=rtol
        )


def tnd_uneven_non_aligned_mask_seqused():
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


def paged_scrambled_tail_and_illegal_page():
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


def large_s2_and_stable_tie_break():
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


def topk_upper_bound_and_d64_n64():
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


def empty_key_request_initializes_invalid_outputs():
    torch.manual_seed(906)
    query = torch.randn(8, 2, 16, dtype=torch.float16)
    key = torch.randn(32, 16, dtype=torch.float16)
    weights = torch.randn(8, 2, dtype=torch.float16)
    cu_q = torch.tensor([0, 4, 8], dtype=torch.int32)
    cu_k = torch.tensor([0, 0, 32], dtype=torch.int32)
    actual = _call(
        query, key, weights, 4, layout_q="TND", layout_k="TND",
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        return_value=True, return_score=True,
    )
    expected = golden(
        torch, query, key, weights, 4, layout_q="TND", layout_k="TND",
        cu_q=cu_q, cu_k=cu_k, return_value=True, return_score=True,
    )
    _assert_outputs(actual, expected)


def flat_seqused_q_without_cu_seqlens():
    torch.manual_seed(907)
    query = torch.randn(8, 2, 16, dtype=torch.float16)
    key = torch.randn(32, 16, dtype=torch.float16)
    weights = torch.randn(8, 2, dtype=torch.float16)
    seqused_q = torch.tensor([5], dtype=torch.int32)
    actual = _call(
        query, key, weights, 4, seqused_q=seqused_q,
        return_value=True, return_score=True,
    )
    expected = golden(
        torch, query, key, weights, 4, seqused_q=seqused_q,
        return_value=True, return_score=True,
    )
    _assert_outputs(actual, expected)


def paged_4d_key_and_multi_batch_offset():
    torch.manual_seed(908)
    tq, n1, s2, d, topk, block, batch = 8, 2, 32, 16, 4, 16, 2
    query = torch.randn(tq, n1, d, dtype=torch.float16)
    weights = torch.randn(tq, n1, dtype=torch.float16)
    max_blocks = (s2 + block - 1) // block
    pages = torch.randn(batch * max_blocks, block, 1, d, dtype=torch.float16)
    table = torch.arange(batch * max_blocks, dtype=torch.int32).reshape(batch, max_blocks)
    seqused_k = torch.full((batch,), s2, dtype=torch.int32)
    cu_q = torch.tensor([0, 4, 8], dtype=torch.int32)
    offset = torch.arange(100, 100 + tq, dtype=torch.int32)
    actual = _call(
        query, pages, weights, topk, layout_q="TND", layout_k="PA_BBND",
        cu_seqlens_q=cu_q, block_table=table, seqused_k=seqused_k,
        max_seq_len_k=s2, output_idx_offset=offset,
        return_value=True, return_score=True,
    )
    expected = golden(
        torch, query, pages, weights, topk, layout_q="TND", layout_k="PA_BBND",
        cu_q=cu_q, block_table=table, seqused_k=seqused_k,
        max_seq_len_k=s2, output_idx_offset=offset,
        return_value=True, return_score=True,
    )
    _assert_outputs(actual, expected)


def default_backend_hidden_outputs_and_out_identity():
    torch.manual_seed(909)
    query = torch.randn(8, 2, 16, dtype=torch.float16)
    key = torch.randn(32, 16, dtype=torch.float16)
    weights = torch.randn(8, 2, dtype=torch.float16)
    expected = golden(
        torch, query, key, weights, 4,
        return_value=False, return_score=False,
    )
    actual = _call(
        query, key, weights, 4, backend=None,
        return_value=False, return_score=False,
    )
    _assert_outputs(actual, expected)
    supplied = tuple(_device(torch.empty_like(value)) for value in expected)
    returned = _call(
        query, key, weights, 4, backend=None, out=supplied,
        return_value=False, return_score=False,
    )
    assert all(received is output for received, output in zip(returned, supplied))
    _assert_outputs(returned, expected)


CASES = {
    "tnd_uneven_non_aligned_mask_seqused": tnd_uneven_non_aligned_mask_seqused,
    "paged_scrambled_tail_and_illegal_page": paged_scrambled_tail_and_illegal_page,
    "large_s2_and_stable_tie_break": large_s2_and_stable_tie_break,
    "topk_upper_bound_and_d64_n64": topk_upper_bound_and_d64_n64,
    "empty_key_request_initializes_invalid_outputs": empty_key_request_initializes_invalid_outputs,
    "flat_seqused_q_without_cu_seqlens": flat_seqused_q_without_cu_seqlens,
    "paged_4d_key_and_multi_batch_offset": paged_4d_key_and_multi_batch_offset,
    "default_backend_hidden_outputs_and_out_identity": default_backend_hidden_outputs_and_out_identity,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=tuple(CASES))
    args = parser.parse_args()
    CASES[args.case]()
    print("PASS", args.case)


if __name__ == "__main__":
    main()
