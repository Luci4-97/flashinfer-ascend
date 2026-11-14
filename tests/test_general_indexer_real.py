"""Real-NPU edge acceptance for attention.general_indexer in fresh processes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.real_compile
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def general_indexer_environment(tmp_path_factory):
    from flashinfer.operators.pytest_support import prepare_release

    release = prepare_release(
        "general_indexer", tmp_path_factory.mktemp("general-indexer-aot")
    )
    environment = dict(os.environ, **release, FLASHINFER_NPU_JIT_DISABLE="1")
    # CANN snapshots custom OPP paths before the first NPU operation. Every case
    # must receive this release environment in a fresh interpreter.
    environment.pop("ASCEND_CUSTOM_OPP_PATH", None)
    return environment


def _run_case(environment, case):
    result = subprocess.run(
        [sys.executable, "-B", "-m", "tests.general_indexer_acceptance", case],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS " + case in result.stdout


def test_tnd_uneven_non_aligned_mask_seqused(general_indexer_environment):
    _run_case(general_indexer_environment, "tnd_uneven_non_aligned_mask_seqused")


def test_paged_scrambled_tail_and_illegal_page(general_indexer_environment):
    _run_case(general_indexer_environment, "paged_scrambled_tail_and_illegal_page")


def test_large_s2_and_stable_tie_break(general_indexer_environment):
    _run_case(general_indexer_environment, "large_s2_and_stable_tie_break")


def test_topk_upper_bound_and_d64_n64(general_indexer_environment):
    _run_case(general_indexer_environment, "topk_upper_bound_and_d64_n64")


def test_empty_key_request_initializes_invalid_outputs(general_indexer_environment):
    _run_case(general_indexer_environment, "empty_key_request_initializes_invalid_outputs")


def test_flat_seqused_q_without_cu_seqlens(general_indexer_environment):
    _run_case(general_indexer_environment, "flat_seqused_q_without_cu_seqlens")


def test_paged_4d_key_and_multi_batch_offset(general_indexer_environment):
    _run_case(general_indexer_environment, "paged_4d_key_and_multi_batch_offset")


def test_default_backend_hidden_outputs_and_out_identity(general_indexer_environment):
    _run_case(general_indexer_environment, "default_backend_hidden_outputs_and_out_identity")
