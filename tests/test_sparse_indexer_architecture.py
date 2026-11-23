"""Compile-time architecture checks for the shared sparse-indexer layers."""

from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "csrc/kernels/sparse_indexer/common/include"
BLOCK = COMMON / "sparse_indexer/block"
FIXTURES = ROOT / "tests/fixtures/sparse_indexer"

CANONICAL_BLOCKS = {
    "block_schedule.hpp",
    "block_key_load.hpp",
    "block_qk_mmad.hpp",
    "block_reduce_quant_head.hpp",
    "block_topk.hpp",
}


def test_l1_contains_exactly_five_flat_block_headers():
    assert {p.name for p in BLOCK.iterdir()} == CANONICAL_BLOCKS
    assert not any(p.is_dir() for p in BLOCK.iterdir())
    assert not (BLOCK / "detail").exists()
    assert not (BLOCK / "prototype").exists()
    common = ROOT / "csrc/kernels/sparse_indexer/common"
    layers = {p.name for p in (common / "include/sparse_indexer").iterdir() if p.is_dir()}
    assert layers == {"block", "foundation", "pipeline"}


def test_block_includes_stay_behind_the_five_public_headers():
    roots = (
        ROOT / "csrc/kernels/sparse_indexer/common/include/sparse_indexer/pipeline",
        ROOT / "csrc/kernels/sparse_indexer/quant_lightning_indexer/op_kernel",
    )
    include = re.compile(r'#include "sparse_indexer/block/([^"]+)"')
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".h", ".hpp", ".cpp"}:
                continue
            for name in include.findall(path.read_text()):
                assert name in CANONICAL_BLOCKS, f"{path}: non-canonical block include {name}"


def test_irs_mock_covers_required_capability_matrix_without_qli_types():
    compiler = shutil.which("c++")
    assert compiler, "host C++ compiler is required for compile-time policy contracts"
    source = FIXTURES / "irs_mock_config_probe.cpp"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsyntax-only",
            "-I" + str(COMMON),
            "-I" + str(FIXTURES),
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    text = (FIXTURES / "irs_mock_config.hpp").read_text()
    for token in (
        "ContinuousCache",
        "PagedCache",
        "NoMask",
        "Mask3",
        "RegionCausal",
        "TokenSelection",
        "RegionSelection",
        "IndicesOnly",
        "ScoreAndIndices",
        "128U",
        "256U",
    ):
        assert token in text
    assert "quant_lightning_indexer_tiling" not in text
    assert "QuantLightningIndexer" not in text
