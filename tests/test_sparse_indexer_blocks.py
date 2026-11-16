"""Source contracts for sparse-indexer common block configuration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCK = ROOT / "csrc/kernels/sparse_indexer/common/op_kernel/block"
KERNEL = ROOT / "csrc/kernels/sparse_indexer/general_indexer/op_kernel/general_indexer_kernel.hpp"
COMMON = ROOT / "csrc/kernels/sparse_indexer/common"


def test_runtime_blocks_define_params_and_do_not_read_concrete_tiling():
    for name in (
        "block_scheduler.hpp",
        "key_loader.hpp",
        "block_mmad.hpp",
        "block_reduce_head.hpp",
        "block_topk.hpp",
    ):
        text = (BLOCK / name).read_text()
        assert "struct Params" in text, name
        assert "GeneralIndexerTilingData" not in text, name
        assert "general_indexer_tiling_data.h" not in text, name


def test_kernel_entry_is_the_only_tiling_to_params_adapter():
    text = KERNEL.read_text()
    assert "MakeSchedulerParams" in text
    assert "MakeKeyLoaderParams" in text
    assert "MakeReduceParams" in text
    assert "MakeTopKParams" in text


def test_common_alignment_is_not_owned_by_general_indexer_tiling():
    align = COMMON / "include/sparse_indexer_align.hpp"
    assert align.is_file()
    assert "SparseIndexer::Common" in align.read_text()
    for path in BLOCK.glob("*.hpp"):
        text = path.read_text()
        assert "namespace GeneralIndexerAlign" not in text, path
        assert "GeneralIndexerAlign::" not in text, path
