"""Public semantics for the Ascend950 general indexer.

The operator computes, per query row::

    scores = sum_h weights[row, h] * relu(query[row, h] @ key.T)
    indices = stable_topk(scores, k)

Multi-request and paged execution stay inside one native launch. Rendering
layout aliases, output packing and index localization belong to the runner.
"""

from types import SimpleNamespace

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec

OP = "sparse_indexer.general_indexer"
MAX_TOPK = 8192
MAX_TOTAL = 16384
MAX_HEAD_DIM = 64
MAX_HEADS = 64
MAX_BATCH = 16
SCORE_WORKSPACE_LIMIT = 1 << 30


def _dtype(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def _layout(value, name, allowed):
    value = str(value).upper()
    if value not in allowed:
        raise ValueError("%s must be one of %s" % (name, "/".join(allowed)))
    return value


def _head_dim(value):
    if not 16 <= value <= MAX_HEAD_DIM or value % 16:
        raise ValueError("head dim must be in [16, 64] and multiple of 16")


def _positive(name, value):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(name + " must be a positive integer")


def _meta(arguments):
    """Validate shape/metadata facts without reading Tensor values."""
    query = arguments["query"]
    key = arguments["key"]
    weights = arguments["weights"]
    k = arguments["k"]
    layout_q = _layout(arguments["layout_q"], "layout_q", ("SND", "TND", "BSND"))
    layout_k = _layout(
        arguments["layout_k"], "layout_k", ("SND", "TND", "BSND", "PA_BBND")
    )
    _positive("k", k)
    if k > MAX_TOPK:
        raise ValueError("k must be <= 8192")
    if layout_k == "PA_BBND":
        if layout_q not in ("SND", "TND", "BSND"):
            raise ValueError("layout_k=PA_BBND requires a flat or BSND layout_q")
    elif (layout_q == "BSND") != (layout_k == "BSND"):
        raise ValueError("layout_q and layout_k must both be BSND or both be flat")

    qshape = tuple(int(v) for v in query.shape)
    wshape = tuple(int(v) for v in weights.shape)
    kshape = tuple(int(v) for v in key.shape)
    if layout_q == "BSND":
        if len(qshape) != 4 or len(wshape) != 3:
            raise ValueError("layout_q=BSND expects query [B,S1,N1,D] and weights [B,S1,N1]")
        batch, s1, n1, d = qshape
        tq = batch * s1
        if wshape != (batch, s1, n1):
            raise ValueError("weights shape must match query [B,S1,N1]")
    else:
        if len(qshape) != 3 or len(wshape) != 2:
            raise ValueError("flat query expects query [Tq,N1,D] and weights [Tq,N1]")
        tq, n1, d = qshape
        if wshape != (tq, n1):
            raise ValueError("weights shape must match query [Tq,N1]")
        batch = 1
        s1 = None

    _positive("Tq", tq)
    if tq > MAX_TOTAL:
        raise ValueError("Tq must be <= 16384")
    if not 1 <= n1 <= MAX_HEADS:
        raise ValueError("N1 must be in [1, 64]")
    _head_dim(d)
    if _dtype(query) not in ("float16", "bfloat16"):
        raise ValueError("query requires float16 or bfloat16")
    if _dtype(key) != _dtype(query):
        raise ValueError("key dtype must match query")
    if _dtype(weights) not in ("float16", "bfloat16", "float32"):
        raise ValueError("weights requires float16, bfloat16 or float32")

    cu_q = arguments["cu_seqlens_q"]
    cu_k = arguments["cu_seqlens_k"]
    seqused_q = arguments["seqused_q"]
    seqused_k = arguments["seqused_k"]
    residual = arguments["cmp_residual_k"]
    table = arguments["block_table"]
    offset = arguments["output_idx_offset"]
    max_seq_len_k = arguments["max_seq_len_k"]

    if not isinstance(max_seq_len_k, int) or max_seq_len_k < 0:
        raise ValueError("max_seq_len_k must be a non-negative integer")

    if cu_q is not None or cu_k is not None:
        if layout_q == "BSND":
            raise ValueError("cu_seqlens is only valid with flat layouts")
        if cu_q is None:
            raise ValueError("cu_seqlens_q is required when cu_seqlens_k is provided")
        if layout_k != "PA_BBND" and cu_k is None:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")
        if len(tuple(cu_q.shape)) != 1 or tuple(cu_q.shape)[0] < 2:
            raise ValueError("cu_seqlens_q must have shape [B+1]")
        if _dtype(cu_q) != "int32":
            raise ValueError("cu_seqlens_q must be int32")
        if cu_k is not None and (
            len(tuple(cu_k.shape)) != 1
            or tuple(cu_k.shape)[0] != tuple(cu_q.shape)[0]
            or _dtype(cu_k) != "int32"
        ):
            raise ValueError("cu_seqlens_k must be int32 [B+1] matching cu_seqlens_q")
        batch = int(cu_q.shape[0]) - 1
    elif layout_q == "BSND":
        batch = qshape[0]

    if not 1 <= batch <= MAX_BATCH:
        raise ValueError("batch must be in [1, 16]")

    if layout_k == "PA_BBND":
        if table is None or seqused_k is None:
            raise ValueError("layout_k=PA_BBND requires block_table and seqused_k")
        tshape = tuple(int(v) for v in table.shape)
        if _dtype(table) != "int32" or len(tshape) != 2 or tshape[0] != batch or tshape[1] < 1:
            raise ValueError("block_table must be int32 [B, max_blocks]")
        if len(kshape) not in (3, 4):
            raise ValueError("paged key expects [pages, block_size, D] or [pages, block_size, 1, D]")
        if len(kshape) == 4 and kshape[2] != 1:
            raise ValueError("paged key requires N2 == 1")
        if kshape[-1] != d or kshape[0] < 1 or kshape[1] < 1:
            raise ValueError("paged key must have positive pages/block_size and matching D")
        if tuple(seqused_k.shape) != (batch,) or _dtype(seqused_k) != "int32":
            raise ValueError("seqused_k must be int32 [batch]")
        if layout_q != "BSND" and batch > 1 and cu_q is None:
            raise ValueError("flat PA with batch > 1 requires cu_seqlens_q")
        if max_seq_len_k <= 0:
            raise ValueError("max_seq_len_k is required for paged execution")
        if max_seq_len_k > MAX_TOTAL:
            raise ValueError("max_seq_len_k must be <= 16384")
        s2 = max_seq_len_k
        tk = batch * s2
    else:
        if max_seq_len_k:
            raise ValueError("max_seq_len_k is only valid for layout_k=PA_BBND")
        if table is not None:
            raise ValueError("block_table is only valid for layout_k=PA_BBND")
        if layout_k == "BSND":
            if len(kshape) not in (3, 4) or kshape[0] != batch:
                raise ValueError("layout_k=BSND expects key [B,S2,D] or [B,S2,1,D]")
            if len(kshape) == 4 and kshape[2] != 1:
                raise ValueError("BSND key requires N2 == 1")
            if kshape[-1] != d:
                raise ValueError("key head dim must match query")
            s2 = int(kshape[1])
        else:
            if len(kshape) == 2:
                s2 = int(kshape[0])
            elif len(kshape) == 3 and kshape[1] == 1:
                s2 = int(kshape[0])
            else:
                raise ValueError("flat key expects [Tk,D] or [Tk,1,D]")
            if kshape[-1] != d:
                raise ValueError("key head dim must match query")
        if s2 <= 0:
            raise ValueError("S2/Tk must be positive")
        tk = batch * s2 if layout_q == "BSND" else s2

    if tk > MAX_TOTAL:
        raise ValueError("Tk must be <= 16384")
    if tq * tk * 4 > SCORE_WORKSPACE_LIMIT:
        raise ValueError("Tq*Tk*4 must be <= 1 GiB")

    if seqused_q is not None:
        if tuple(seqused_q.shape) != (batch,) or _dtype(seqused_q) != "int32":
            raise ValueError("seqused_q must be int32 [batch]")
    if seqused_k is not None:
        if tuple(seqused_k.shape) != (batch,) or _dtype(seqused_k) != "int32":
            raise ValueError("seqused_k must be int32 [batch]")

    mask_mode = arguments["mask_mode"]
    cmp_ratio = arguments["cmp_ratio"]
    if type(mask_mode) is not int or mask_mode not in (0, 3):
        raise ValueError("mask_mode must be 0 or 3")
    if type(cmp_ratio) is not int or not 1 <= cmp_ratio <= 128:
        raise ValueError("cmp_ratio must be in [1, 128]")
    if mask_mode == 3 and cmp_ratio > 1:
        if residual is None:
            raise ValueError("cmp_residual_k is required when mask_mode=3 and cmp_ratio>1")
    elif residual is not None:
        raise ValueError("cmp_residual_k must be omitted when mask_mode=0 or cmp_ratio==1")
    if residual is not None and (
        tuple(residual.shape) != (batch,) or _dtype(residual) != "int32"
    ):
        raise ValueError("cmp_residual_k must be int32 [batch]")

    if offset is not None:
        oshape = tuple(int(v) for v in offset.shape)
        if _dtype(offset) != "int32" or oshape not in ((tq,), (tq, 1)):
            raise ValueError("output_idx_offset must be int32 [Tq] or [Tq,1]")

    if layout_q == "BSND":
        indices_shape = (batch, s1, 1, k)
        score_shape = (batch, s1, s2)
    else:
        indices_shape = (tq, k)
        score_shape = (tq, s2)

    if arguments["return_value"]:
        values_shape = indices_shape
    else:
        values_shape = (0,)
    if arguments["return_score"]:
        scores_shape = score_shape
    else:
        scores_shape = (0,)

    kv_cache_type = "paged" if layout_k == "PA_BBND" else (
        "ragged" if layout_q == "TND" or cu_q is not None else "dense"
    )
    return SimpleNamespace(
        batch=batch, tq=tq, tk=tk, s1=s1, s2=s2, n1=n1, d=d,
        layout_q=layout_q, layout_k=layout_k, k=k,
        indices_shape=indices_shape, values_shape=values_shape,
        scores_shape=scores_shape, dtype=_dtype(query),
        kv_cache_type=kv_cache_type,
    )


def infer(arguments, layouts):
    meta = _meta(arguments)
    if layouts is not None and any(value != "ND" for value in layouts.values()):
        raise ValueError("general_indexer requires contiguous ND tensors")
    device = arguments["query"].device
    inputs = tuple(
        arguments[name]
        for name in (
            "query", "key", "weights", "block_table", "cu_seqlens_q",
            "cu_seqlens_k", "seqused_q", "seqused_k", "cmp_residual_k",
            "output_idx_offset",
        )
    )
    return Invocation(
        inputs,
        (
            OutputSpec(meta.indices_shape, "int32", device),
            OutputSpec(meta.values_shape, "float32", device),
            OutputSpec(meta.scores_shape, "float32", device),
        ),
        {
            "layout_q": meta.layout_q,
            "layout_k": meta.layout_k,
            "mask_mode": arguments["mask_mode"],
            "cmp_ratio": arguments["cmp_ratio"],
            "max_seq_len_k": arguments["max_seq_len_k"],
            "return_value": arguments["return_value"],
            "return_score": arguments["return_score"],
        },
        InferenceSpec(
            compute_dtype=meta.dtype,
            accumulator_dtype="float32",
            attention_variant="lightning_indexer",
            kv_cache_type=meta.kv_cache_type,
            causal=arguments["mask_mode"] == 3,
        ),
    )


_METADATA_TENSOR_NAMES = (
    "cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k",
    "cmp_residual_k", "block_table", "output_idx_offset",
)


def _strict_metadata(arguments):
    if arguments.get("validate_metadata", False):
        return True
    for name in _METADATA_TENSOR_NAMES:
        tensor = arguments.get(name)
        if tensor is not None and getattr(tensor.device, "type", None) != "cpu":
            return False
    return True


def validate_values(arguments):
    # Shape/dtype/presence validation already happens in _meta/infer. Reading
    # NPU metadata values here would add a host D2H on every eager launch, so
    # value-domain checks are opt-in for device tensors.
    if not _strict_metadata(arguments):
        return
    meta = _meta(arguments)
    cu_q = arguments["cu_seqlens_q"]
    cu_k = arguments["cu_seqlens_k"]
    seq_q = arguments["seqused_q"]
    seq_k = arguments["seqused_k"]
    residual = arguments["cmp_residual_k"]
    table = arguments["block_table"]
    offset = arguments["output_idx_offset"]

    q_lens = ([meta.s1] * meta.batch if meta.layout_q == "BSND"
              else [meta.tq] * meta.batch)
    k_lens = [meta.s2] * meta.batch
    if cu_q is not None:
        q = cu_q.cpu().tolist()
        if q[0] != 0 or q[-1] != meta.tq or any(a > b for a, b in zip(q, q[1:])):
            raise ValueError("cu_seqlens_q must be nondecreasing, start at 0 and end at Tq")
        q_lens = [b - a for a, b in zip(q, q[1:])]
    if cu_k is not None:
        k = cu_k.cpu().tolist()
        if k[0] != 0 or k[-1] != meta.tk or any(a > b for a, b in zip(k, k[1:])):
            raise ValueError("cu_seqlens_k must be nondecreasing, start at 0 and end at Tk")
        if cu_q is not None and len(k) != len(q):
            raise ValueError("cu_seqlens_q/k batch dimensions differ")
        k_lens = [b - a for a, b in zip(k, k[1:])]
    if seq_q is not None:
        values = seq_q.cpu().tolist()
        if any(v < 0 or v > q_lens[i] for i, v in enumerate(values)):
            raise ValueError("seqused_q must be in [0, request query length]")
    if seq_k is not None:
        values = seq_k.cpu().tolist()
        if any(v < 0 or v > k_lens[i] for i, v in enumerate(values)):
            raise ValueError("seqused_k must be in [0, request key length]")
    if meta.k and any(length and meta.k > length for length in k_lens):
        raise ValueError("k must not exceed any request key length")
    if residual is not None:
        ratio = arguments["cmp_ratio"]
        if any(v < 0 or v >= ratio for v in residual.cpu().tolist()):
            raise ValueError("cmp_residual_k must satisfy 0 <= residual < cmp_ratio")
    if table is not None:
        used = seq_k.cpu().tolist()
        block = int(arguments["key"].shape[1])
        width = int(table.shape[1])
        if any((v + block - 1) // block > width for v in used):
            raise ValueError("block_table width is smaller than seqused_k requires")
    if offset is not None and tuple(offset.shape) not in ((meta.tq,), (meta.tq, 1)):
        raise ValueError("output_idx_offset shape must match Tq")


OPERATOR = OperatorDefinition(
    name="general_indexer",
    domain="sparse_indexer",
    op=OP,
    schema=(
        "(Tensor query, Tensor key, Tensor weights, int k, *, "
        "str layout_q='SND', str layout_k='SND', int mask_mode=0, int cmp_ratio=1, "
        "int max_seq_len_k=0, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
        "Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, "
        "Tensor? block_table=None, Tensor? output_idx_offset=None, "
        "bool return_value=False, bool return_score=False, "
        "bool validate_metadata=False) -> (Tensor, Tensor, Tensor)"
    ),
    output_names=("indices", "values", "scores"),
    infer=infer,
    validate_values=validate_values,
    description=(
        "Ascend950 general/lightning indexer. Computes stable TopK of "
        "sum_h W*ReLU(Q@K^T), with dense/TND/BSND/PA_BBND, causal/compressed "
        "masking and optional values/debug scores. Eager AOT-only and not graph capturable."
    ),
)
