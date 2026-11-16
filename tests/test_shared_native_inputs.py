"""Shared native source closure contracts for Ascend C authoring."""

from dataclasses import replace
from pathlib import Path

import pytest

from flashinfer.ascendc.authoring import prepare_native
from flashinfer.ascendc.definition import BuildConfig
from flashinfer.ascendc.sparse_indexer.general_indexer.implementation import (
    IMPLEMENTATION,
)
from flashinfer.runtime import SchemaError

ROOT = Path(__file__).resolve().parents[1]


def _build(**changes):
    return BuildConfig(
        device="op_kernel/general_indexer.cpp",
        host=("op_host/general_indexer.cpp",),
        op_type="FiGeneralIndexer",
        tiling="FiGeneralIndexerTiling",
        includes=("op_kernel", "op_host", "include"),
        **changes,
    )


def test_build_config_accepts_shared_native_roots_and_includes():
    build = _build(
        shared_native_roots=("csrc/kernels/sparse_indexer/common/op_kernel/block",),
        shared_includes=("csrc/kernels/sparse_indexer/common/op_kernel/block",),
    )
    assert build.shared_native_roots == (
        "csrc/kernels/sparse_indexer/common/op_kernel/block",
    )
    assert build.shared_includes == (
        "csrc/kernels/sparse_indexer/common/op_kernel/block",
    )


@pytest.mark.parametrize("root", ["../outside", "/tmp/outside", "."])
def test_prepare_native_rejects_unsafe_shared_roots(root):
    native = replace(
        IMPLEMENTATION.native,
        build=replace(IMPLEMENTATION.native.build, shared_native_roots=(root,)),
    )
    with pytest.raises(SchemaError, match="shared native"):
        prepare_native(ROOT, ROOT / "flashinfer/ascendc/sparse_indexer/general_indexer", native)
