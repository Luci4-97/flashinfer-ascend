"""TDD contracts for LI V2-style runtime metadata and scheduler extension points."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "csrc/kernels/sparse_indexer/common"
BLOCK = COMMON / "op_kernel/block"
GI = ROOT / "csrc/kernels/sparse_indexer/general_indexer"
SCHEDULER_FIXTURE = ROOT / "tests/fixtures/sparse_indexer_scheduler_contract.cpp"


def test_common_scheduler_is_generic_over_request_source_and_policy():
    text = (BLOCK / "block_scheduler.hpp").read_text()
    assert "template <class RequestRangeSource_, class SchedulePolicy_>" in text
    assert "struct ScheduleCoord" in text
    assert "class RequestBlockDiagonalPolicy" in text
    assert "cuQ" not in text
    assert "cuK" not in text
    assert "seqused" not in text
    assert "cmpResidual" not in text
    assert "GeneralIndexerTilingData" not in text


def test_common_scheduler_accepts_a_different_variant_metadata_schema(tmp_path):
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    assert compiler, "a C++17 host compiler is required for the scheduler contract fixture"
    executable = tmp_path / "sparse_indexer_scheduler_contract"
    command = [
        compiler,
        "-std=c++17",
        "-O0",
        "-D__forceinline__=inline",
        "-D__aicore__=",
        f"-I{COMMON / 'op_kernel'}",
        f"-I{COMMON / 'include'}",
        f"-I{ROOT / '3rdparty/catlass/include'}",
        str(SCHEDULER_FIXTURE),
        "-o",
        str(executable),
    ]
    compiled = subprocess.run(command, text=True, capture_output=True, check=False)
    assert compiled.returncode == 0, (
        "scheduler fixture did not compile:\n"
        + " ".join(shlex.quote(item) for item in command)
        + "\n"
        + compiled.stdout
        + compiled.stderr
    )
    executed = subprocess.run([str(executable)], text=True, capture_output=True, check=False)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert executed.stdout.strip() == "4:0:0:0:4\n4:0:0:4:2\n3:4:0:6:2\n4:0:0:0:6\n3:4:0:6:2"


def test_kernel_entry_forwards_runtime_metadata_instead_of_void_casting_it():
    entry = (GI / "op_kernel/general_indexer.cpp").read_text()
    for name in ("cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k", "cmp_residual_k"):
        assert f"(void){name};" not in entry
    adapter = (GI / "op_kernel/arch35/general_indexer.h").read_text()
    for name in ("cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k", "cmp_residual_k"):
        assert name in adapter
    assert "GM_ADDR cu_seqlens_q" in adapter
    assert "GM_ADDR cmp_residual_k" in adapter


def test_tiling_does_not_snapshot_runtime_metadata_values():
    tiling_data = (GI / "include/general_indexer_tiling_data.h").read_text()
    for field in ("cuSeqlensQ", "cuSeqlensK", "sequsedQ", "sequsedK", "cmpResidualK"):
        assert f"{field}[" not in tiling_data
    tiling = (GI / "include/general_indexer_tiling.h").read_text()
    assert "ACL_MEMCPY_DEVICE_TO_HOST" not in tiling
    assert "aclrtMemcpy" not in tiling


def test_key_loader_consumes_resolved_key_coordinates():
    text = (BLOCK / "key_loader.hpp").read_text()
    assert "struct KeyCoord" in text
    assert "globalNStart" in text
    assert "requestNStart" in text
    assert "cuK" not in text
    assert "seqused" not in text
    assert "cmpResidual" not in text
