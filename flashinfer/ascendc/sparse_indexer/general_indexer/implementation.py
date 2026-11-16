"""Declarative Ascend C package for sparse_indexer.general_indexer."""

from flashinfer.ascendc.definition import (
    AscendCImplementation,
    BuildConfig,
    NativeBinding,
)
from flashinfer.jit.aclnn_contract import AttributeParameter
from flashinfer.jit.recipe import (
    CATLASS_REVISION,
    BuildDependency,
    ShapeConstraint,
    TensorParameter,
)
from flashinfer.operators.sparse_indexer.general_indexer.operator import (
    MAX_BATCH,
    MAX_HEAD_DIM,
    MAX_HEADS,
    MAX_TOPK,
    MAX_TOTAL,
    OPERATOR,
)
from flashinfer.operators.author_api import Support
from flashinfer.runtime import CapabilityCase

from .runner import admission, make_runner

KERNEL_ID = "ascendc/sparse_indexer/general_indexer/catlass/op_package/Ascend950"


def _tensor(name, role, dtype, shape, *, alternative=(), optional=False):
    return TensorParameter(
        name=name,
        role=role,
        dtype=dtype,
        shape=shape,
        shape_alternatives=alternative,
        optional=optional,
    )


_TENSORS = (
    _tensor("query", "input", "float16", ("TQ", "N1", "D")),
    _tensor("key", "input", "float16", ("TK", "D"), alternative=(("PAGES", "BLOCK", "D"),)),
    _tensor("weights", "input", "float16", ("TQ", "N1")),
    _tensor("block_table", "input", "int32", ("B", "MB"), optional=True),
    _tensor("cu_seqlens_q", "input", "int32", ("BC",), optional=True),
    _tensor("cu_seqlens_k", "input", "int32", ("BC",), optional=True),
    _tensor("seqused_q", "input", "int32", ("B",), optional=True),
    _tensor("seqused_k", "input", "int32", ("B",), optional=True),
    _tensor("cmp_residual_k", "input", "int32", ("B",), optional=True),
    _tensor("output_idx_offset", "input", "int32", ("TQ",), alternative=(("TQ", 1),), optional=True),
    _tensor("score_zero_page", "input", "float16", ("BLOCK", "D"), optional=True),
    _tensor("indices", "output", "int32", ("TQ", "K")),
    _tensor("values", "output", "float32", ("TQ", "K")),
    _tensor("scores", "output", "float32", ("TQ", "TK")),
)

_DTYPE_SIGNATURES = tuple(
    (qk, qk, w, "int32", "int32", "int32", "int32", "int32", "int32",
     "int32", qk, "int32", "float32", "float32")
    for qk in ("float16", "bfloat16")
    for w in ("float16", "bfloat16", "float32")
)

IMPLEMENTATION = AscendCImplementation(
    operator=OPERATOR,
    kernel_id=KERNEL_ID,
    support=Support(
        soc_models=("Ascend950",),
        cases=(
            CapabilityCase(
                name="ascend950_lightning_indexer",
                compute_dtypes=("float16", "bfloat16"),
                accumulator_dtypes=("float32",),
                attention_variants=("lightning_indexer",),
                kv_cache_types=("dense", "ragged", "paged"),
                causal_values=(False, True),
            ),
        ),
    ),
    build=BuildConfig(
        device="op_kernel/general_indexer.cpp",
        host=("op_host/general_indexer.cpp",),
        op_type="FiGeneralIndexer",
        tiling="FiGeneralIndexerTiling",
        includes=("op_kernel", "op_host", "include"),
        shared_native_roots=("csrc/kernels/sparse_indexer/common/include",),
        shared_includes=("csrc/kernels/sparse_indexer/common/include",),
        implementation="catlass",
        dependencies=(BuildDependency(name="catlass", revision=CATLASS_REVISION),),
        definitions=("CATLASS_ARCH=3510",),
        compile_options=("-O2",),
        link_options=("-lascendcl",),
        workspace_bytes=0,
        device_entry="fi_general_indexer",
        host_entry="aclnnFiGeneralIndexer",
        launch_abi="aclnn.general_indexer.v2",
        package_vendor="flashinfer_general_indexer",
        recipe_id="general_indexer",
    ),
    source_dir="csrc/kernels/attention/general_indexer",
    binding=NativeBinding(
        tensors=_TENSORS,
        dimensions=(
            ShapeConstraint("TQ", 1, MAX_TOTAL),
            ShapeConstraint("TK", 1, MAX_TOTAL),
            ShapeConstraint("N1", 1, MAX_HEADS),
            ShapeConstraint("D", 16, MAX_HEAD_DIM, 16),
            ShapeConstraint("PAGES", 1, 1 << 20),
            ShapeConstraint("BLOCK", 1, 4096),
            ShapeConstraint("B", 1, MAX_BATCH),
            ShapeConstraint("MB", 1, 1 << 16),
            ShapeConstraint("BC", 2, MAX_BATCH + 1),
            ShapeConstraint("K", 1, MAX_TOPK),
        ),
        dtype_signatures=_DTYPE_SIGNATURES,
        attributes=(
            AttributeParameter("k", "int64", 1, 1, MAX_TOPK),
            AttributeParameter("is_paged", "int64", 0, 0, 1),
            AttributeParameter("mask_mode", "int64", 0, 0, 3),
            AttributeParameter("cmp_ratio", "int64", 1, 1, 128),
            AttributeParameter("max_seq_len_k", "int64", 0, 0, MAX_TOTAL),
        ),
    ),
    runner_factory=make_runner,
    admission=admission,
    workspace_bound=(1 << 30) + (128 << 20),
    allow_jit_fallback=False,
)

NATIVE = IMPLEMENTATION.native
descriptor = IMPLEMENTATION.bind
