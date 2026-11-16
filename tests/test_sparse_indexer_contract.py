"""Producer-consumer contract between sparse indexers and sparse attention."""

import torch

from flashinfer.operators.attention.sparse_flash_attention.operator import (
    OPERATOR as SPARSE_ATTENTION,
)


def _token_topk_to_blocks(token_indices, sparse_block_size):
    rows = []
    for row in token_indices:
        blocks = []
        seen = set()
        for token in row.tolist():
            if token < 0:
                continue
            block = token // sparse_block_size
            if block not in seen:
                seen.add(block)
                blocks.append(block)
        padded = blocks + [-1] * (len(row) - len(blocks))
        rows.append(padded)
    return torch.tensor(rows, dtype=torch.int32).unsqueeze(1)


def test_token_topk_requires_explicit_conversion_to_sparse_blocks():
    token_indices = torch.tensor([[0, 2, 3, -1], [5, 6, 7, 8]], dtype=torch.int32)
    blocks = _token_topk_to_blocks(token_indices, 2)
    assert blocks.shape == (2, 1, 4)
    assert blocks.tolist() == [[[0, 1, -1, -1]], [[2, 3, 4, -1]]]


def test_converted_blocks_satisfy_sparse_attention_metadata_contract():
    blocks = torch.tensor([[[0, 1, -1, -1]], [[2, 3, 4, -1]]], dtype=torch.int32)
    query = torch.zeros(2, 1, 4, 512, dtype=torch.float16)
    key = torch.zeros(2, 32, 1, 512, dtype=torch.float16)
    arguments = SPARSE_ATTENTION.bind(
        query, key, key, blocks, 1.0,
        sparse_block_size=2, layout_query="BSND", layout_kv="BSND",
        sparse_mode=0, return_softmax_lse=False,
    )
    SPARSE_ATTENTION.validate_values(arguments)
