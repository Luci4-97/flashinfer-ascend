"""Source contracts for sparse-indexer common block configuration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCK = ROOT / "csrc/kernels/sparse_indexer/common/op_kernel/block"
KERNEL = ROOT / "csrc/kernels/sparse_indexer/general_indexer/op_kernel/arch35/general_indexer_kernel.h"
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
    common_header = COMMON / "include/sparse_indexer_common.hpp"
    assert common_header.is_file()
    assert "SparseIndexer::Common" in common_header.read_text()
    for path in BLOCK.glob("*.hpp"):
        text = path.read_text()
        assert "namespace GeneralIndexerAlign" not in text, path
        assert "GeneralIndexerAlign::" not in text, path
        assert "GENERAL_INDEXER" not in text, path


def test_general_indexer_uses_arch35_entry_layout_without_dead_entries():
    op_kernel = ROOT / "csrc/kernels/sparse_indexer/general_indexer/op_kernel"
    arch = op_kernel / "arch35"
    assert not (op_kernel / "general_indexer.hpp").exists()
    assert not (op_kernel / "general_indexer_kernel.hpp").exists()
    assert not (arch / "general_indexer_arch35.h").exists()
    assert (arch / "general_indexer.h").is_file()
    assert (arch / "general_indexer_kernel.h").is_file()
    entry = (op_kernel / "general_indexer.cpp").read_text()
    assert '#include "arch35/general_indexer.h"' in entry
    assert "block/key_loader.hpp" not in entry
    for path in op_kernel.rglob("*"):
        if path.is_file() and path.suffix in {".cpp", ".h", ".hpp"}:
            assert "GeneralIndexerEntry" not in path.read_text(), path


def test_general_indexer_host_sources_are_split_by_responsibility():
    op_host = ROOT / "csrc/kernels/sparse_indexer/general_indexer/op_host"
    assert (op_host / "general_indexer_def.cpp").is_file()
    assert (op_host / "general_indexer_infershape.cpp").is_file()
    assert (op_host / "arch35/general_indexer_tiling.cpp").is_file()
    assert not (op_host / "general_indexer.cpp").exists()
