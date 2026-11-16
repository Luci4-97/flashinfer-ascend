"""Independent CPU golden and deterministic general-indexer samples."""

import math


def _stable_topk(torch, scores, k):
    rows = []
    for row in scores:
        finite = torch.isfinite(row)
        ids = torch.arange(row.numel(), dtype=torch.int64)[finite]
        if ids.numel() == 0:
            rows.append(torch.full((k,), -1, dtype=torch.int32))
            continue
        values = row[ids]
        order = sorted(range(ids.numel()), key=lambda i: (-float(values[i]), int(ids[i])))[:k]
        result = torch.full((k,), -1, dtype=torch.int32)
        if order:
            result[:len(order)] = ids[torch.tensor(order, dtype=torch.int64)].to(torch.int32)
        rows.append(result)
    return torch.stack(rows)


def _gather_values(torch, scores, indices):
    safe = indices.long().clamp_min(0)
    result = torch.gather(scores, 1, safe)
    return result.masked_fill(indices < 0, float("-inf"))


def _reduce(torch, query, key, weights):
    return (torch.relu(torch.einsum("tnd,sd->tns", query.float(), key.float()))
            * weights.float().unsqueeze(-1)).sum(dim=1)


def _mask3_visible_len(local_q, q_len, k_act, cmp_ratio=1, residual=0):
    raw = (k_act * cmp_ratio + residual - q_len + local_q + 1) / float(cmp_ratio)
    if raw < 0:
        return 0
    return min(k_act, math.floor(raw))


def _mask_region(torch, scores, q0, q1, k0, k1, *, q_act, k_act,
                 mask_mode, cmp_ratio, residual):
    q_len, k_len = q1 - q0, k1 - k0
    if q_act < q_len:
        scores[q0 + q_act:q1, k0:k1] = float("-inf")
    if k_act < k_len:
        scores[q0:q1, k0 + k_act:k1] = float("-inf")
    if mask_mode == 3:
        for local_q in range(q_act):
            visible = _mask3_visible_len(local_q, q_act, k_act, cmp_ratio, residual)
            if visible < k_act:
                scores[q0 + local_q, k0 + visible:k0 + k_act] = float("-inf")


def _apply_offset(torch, indices, offset):
    if offset is None:
        return indices
    valid = indices >= 0
    values = offset.reshape(-1).to(torch.int32).unsqueeze(1)
    return torch.where(valid, indices + values, indices)


def _topk(torch, scores, k, offset=None):
    indices = _stable_topk(torch, scores, k)
    values = _gather_values(torch, scores, indices)
    return _apply_offset(torch, indices, offset), values


def golden(torch, query, key, weights, k, *, layout_q="SND", layout_k="SND",
           return_value=False, return_score=False, cu_q=None, cu_k=None,
           block_table=None, seqused_q=None, seqused_k=None, mask_mode=0,
           cmp_ratio=1, residual=None, max_seq_len_k=0, output_idx_offset=None):
    q = query.float().cpu()
    w = weights.float().cpu()
    if block_table is not None:
        pages = key.float().cpu()
        if pages.dim() == 4:
            pages = pages.squeeze(2)
        table = block_table.cpu()
        used = seqused_k.cpu()
        batch, s2 = table.shape[0], max_seq_len_k
        logical = []
        for b in range(batch):
            rows = torch.zeros((s2, pages.shape[-1]), dtype=torch.float32)
            count = int(used[b])
            block = pages.shape[1]
            for token in range(count):
                page = int(table[b, token // block])
                if 0 <= page < pages.shape[0]:
                    rows[token] = pages[page, token % block]
            logical.append(rows)
    else:
        key_work = key.float().cpu()
        if key_work.dim() == 4:
            key_work = key_work.squeeze(2)
        if layout_q == "BSND":
            logical = [key_work[b] for b in range(key_work.shape[0])]
        else:
            logical = None
        s2 = key_work.shape[-2]

    seq_q = seqused_q.cpu().tolist() if seqused_q is not None else None
    seq_k = seqused_k.cpu().tolist() if seqused_k is not None else None
    residuals = residual.cpu().tolist() if residual is not None else [0] * 16

    if layout_q == "BSND":
        batch, s1 = q.shape[0], q.shape[1]
        rows_i, rows_v, rows_s = [], [], []
        for b in range(batch):
            key_b = logical[b]
            score = _reduce(torch, q[b], key_b, w[b])
            q_act = seq_q[b] if seq_q is not None else s1
            k_act = seq_k[b] if seq_k is not None else s2
            _mask_region(torch, score, 0, s1, 0, s2, q_act=q_act, k_act=k_act,
                         mask_mode=mask_mode, cmp_ratio=cmp_ratio,
                         residual=residuals[b])
            offset = None if output_idx_offset is None else output_idx_offset.reshape(-1)[b * s1:(b + 1) * s1]
            idx, val = _topk(torch, score, k, offset)
            rows_i.append(idx.reshape(1, s1, 1, k))
            rows_v.append(val.reshape(1, s1, 1, k))
            rows_s.append(score.reshape(1, s1, s2))
        indices = torch.cat(rows_i)
        values = torch.cat(rows_v)
        scores = torch.cat(rows_s)
    elif logical is not None:
        ql = cu_q.cpu().tolist() if cu_q is not None else [0, q.shape[0]]
        kl = cu_k.cpu().tolist() if cu_k is not None else [b * s2 for b in range(len(ql))]
        rows_i, rows_v, rows_s = [], [], []
        for b in range(len(ql) - 1):
            q0, q1, k0, k1 = ql[b], ql[b + 1], kl[b], kl[b + 1]
            q_len, k_len = q1 - q0, k1 - k0
            score = _reduce(torch, q[q0:q1], logical[b], w[q0:q1])
            q_act = seq_q[b] if seq_q is not None else q_len
            k_act = seq_k[b] if seq_k is not None else k_len
            _mask_region(torch, score, 0, q_len, 0, k_len, q_act=q_act, k_act=k_act,
                         mask_mode=mask_mode, cmp_ratio=cmp_ratio,
                         residual=residuals[b])
            offset = None if output_idx_offset is None else output_idx_offset.reshape(-1)[q0:q1]
            idx, val = _topk(torch, score, k, offset)
            rows_i.append(idx)
            rows_v.append(val)
            rows_s.append(score)
        indices = torch.cat(rows_i)
        values = torch.cat(rows_v)
        scores = torch.cat(rows_s)
    else:
        key_work = key.float().cpu()
        if key_work.dim() == 3:
            key_work = key_work.squeeze(1)
        if cu_q is None:
            ql, kl = [0, q.shape[0]], [0, key_work.shape[0]]
            scores = _reduce(torch, q, key_work, w)
        else:
            ql = cu_q.cpu().tolist()
            kl = cu_k.cpu().tolist()
            scores = torch.full((q.shape[0], key_work.shape[0]), float("-inf"), dtype=torch.float32)
        for b in range(len(ql) - 1):
            q0, q1, k0, k1 = ql[b], ql[b + 1], kl[b], kl[b + 1]
            if cu_q is not None and q1 > q0 and k1 > k0:
                scores[q0:q1, k0:k1] = _reduce(torch, q[q0:q1], key_work[k0:k1], w[q0:q1])
            q_act = seq_q[b] if seq_q is not None else q1 - q0
            k_act = seq_k[b] if seq_k is not None else k1 - k0
            _mask_region(torch, scores, q0, q1, k0, k1, q_act=q_act, k_act=k_act,
                         mask_mode=mask_mode, cmp_ratio=cmp_ratio,
                         residual=residuals[b])
        indices, values = _topk(torch, scores, k, output_idx_offset)

    if not return_value:
        values = torch.empty((0,), dtype=torch.float32)
    if not return_score:
        scores = torch.empty((0,), dtype=torch.float32)
    return indices.to(torch.int32), values.float(), scores.float()


def sample(torch, torch_npu, *, dtype="float16", w_dtype=None, layout="SND",
           paged=False, batch=2, s1=8, s2=32, n1=2, d=16, topk=4,
           return_value=True, return_score=True, mask_mode=0, cmp_ratio=1,
           cmp_residual=0, seqused=False, offset=False, seed=7):
    rng = torch.Generator().manual_seed(seed)
    qk_dtype = getattr(torch, dtype)
    w_dtype = qk_dtype if w_dtype is None else getattr(torch, w_dtype)
    block = 16

    def device(tensor):
        return torch_npu.npu_format_cast(tensor.npu(), 2)

    def weights_for(tq):
        return (torch.randn((tq, n1), generator=rng) * 0.25).to(w_dtype)

    cu_q = cu_k = None
    seqused_q = seqused_k = None
    residual = None
    table = None
    if layout == "BSND":
        query = (torch.randn((batch, s1, n1, d), generator=rng) * 0.25).to(qk_dtype)
        weights = weights_for(batch * s1).reshape(batch, s1, n1)
        if paged:
            max_blocks = (s2 + block - 1) // block
            pages = torch.zeros((batch * max_blocks, block, d), dtype=qk_dtype)
            table = torch.full((batch, max_blocks), -1, dtype=torch.int32)
            used = torch.full((batch,), s2, dtype=torch.int32)
            for bi in range(batch):
                values = (torch.randn((s2, d), generator=rng) * 0.25).to(qk_dtype)
                for local in range(max_blocks):
                    phys = bi * max_blocks + local
                    table[bi, local] = phys
                    lo = local * block
                    hi = min(s2, lo + block)
                    pages[phys, :hi - lo] = values[lo:hi]
            key = pages
            seqused_k = used
            options = dict(layout_q="BSND", layout_k="PA_BBND",
                           block_table=device(table), seqused_k=device(used),
                           max_seq_len_k=s2)
        else:
            key = (torch.randn((batch, s2, d), generator=rng) * 0.25).to(qk_dtype)
            options = dict(layout_q="BSND", layout_k="BSND")
        tq = batch * s1
        args = (device(query), device(key), device(weights), topk)
    else:
        batch = 1 if layout == "SND" and not paged else batch
        tq = batch * s1
        query = (torch.randn((tq, n1, d), generator=rng) * 0.25).to(qk_dtype)
        weights = weights_for(tq)
        if paged:
            max_blocks = (s2 + block - 1) // block
            pages = torch.zeros((batch * max_blocks, block, d), dtype=qk_dtype)
            table = torch.full((batch, max_blocks), -1, dtype=torch.int32)
            used = torch.full((batch,), s2, dtype=torch.int32)
            for bi in range(batch):
                values = (torch.randn((s2, d), generator=rng) * 0.25).to(qk_dtype)
                for local in range(max_blocks):
                    phys = bi * max_blocks + local
                    table[bi, local] = phys
                    lo = local * block
                    hi = min(s2, lo + block)
                    pages[phys, :hi - lo] = values[lo:hi]
            key = pages
            seqused_k = used
            options = dict(layout_q="TND" if batch > 1 else "SND",
                           layout_k="PA_BBND", block_table=device(table),
                           seqused_k=device(used), max_seq_len_k=s2)
            if batch > 1 or seqused:
                cu_q = torch.tensor([i * s1 for i in range(batch + 1)], dtype=torch.int32)
                options["cu_seqlens_q"] = device(cu_q)
        else:
            key = (torch.randn((batch * s2, d), generator=rng) * 0.25).to(qk_dtype)
            options = dict(layout_q="TND" if batch > 1 else "SND",
                           layout_k="TND" if batch > 1 else "SND")
            if batch > 1 or seqused or mask_mode == 3:
                cu_q = torch.tensor([i * s1 for i in range(batch + 1)], dtype=torch.int32)
                cu_k = torch.tensor([i * s2 for i in range(batch + 1)], dtype=torch.int32)
                options["cu_seqlens_q"] = device(cu_q)
                options["cu_seqlens_k"] = device(cu_k)
        args = (device(query), device(key), device(weights), topk)

    if seqused:
        qv = torch.tensor([max(1, s1 - 1)] * batch, dtype=torch.int32)
        kv = torch.tensor([max(1, s2 - 2)] * batch, dtype=torch.int32)
        seqused_q, seqused_k = device(qv), device(kv)
        options["seqused_q"] = seqused_q
        options["seqused_k"] = seqused_k
    if mask_mode == 3 and cmp_ratio > 1:
        residual = device(torch.full((batch,), cmp_residual, dtype=torch.int32))
        options["cmp_residual_k"] = residual
    if offset:
        output_offset = device(torch.arange(100, 100 + tq, dtype=torch.int32))
        options["output_idx_offset"] = output_offset
    else:
        output_offset = None

    options.update(mask_mode=mask_mode, cmp_ratio=cmp_ratio,
                   return_value=return_value, return_score=return_score,
                   backend="ascendc")
    expected = golden(
        torch, query, key, weights, topk, layout_q=options["layout_q"],
        layout_k=options["layout_k"], return_value=return_value,
        return_score=return_score, cu_q=cu_q, cu_k=cu_k, block_table=table,
        seqused_q=seqused_q, seqused_k=seqused_k, mask_mode=mask_mode,
        cmp_ratio=cmp_ratio, residual=residual,
        max_seq_len_k=options.get("max_seq_len_k", 0),
        output_idx_offset=None if output_offset is None else output_offset.cpu(),
    )
    return args, options, expected


CASE_PARAMETERS = (
    dict(dtype="float16", layout="SND"),
    dict(dtype="bfloat16", layout="SND", return_score=False),
    dict(dtype="float16", w_dtype="bfloat16", layout="SND", return_score=False),
    dict(dtype="float16", w_dtype="float32", layout="SND"),
    dict(dtype="bfloat16", w_dtype="float16", layout="SND", return_score=False),
    dict(dtype="bfloat16", w_dtype="float32", layout="SND"),
    dict(dtype="float16", layout="TND", batch=2),
    dict(dtype="float16", layout="TND", batch=2, mask_mode=3, seqused=True, return_score=False),
    dict(dtype="float16", layout="TND", mask_mode=3, cmp_ratio=4, cmp_residual=1, seqused=True),
    dict(dtype="float16", layout="BSND", batch=2, return_score=False),
    dict(dtype="float16", layout="BSND", batch=2, mask_mode=3, seqused=True),
    dict(dtype="float16", layout="SND", paged=True, batch=1),
    dict(dtype="float16", layout="TND", paged=True, batch=2, mask_mode=3, seqused=True),
    dict(dtype="float16", layout="BSND", paged=True, batch=2, mask_mode=3, seqused=True),
    dict(dtype="float16", layout="SND", topk=1, offset=True),
)
TOLERANCES = (
    (0.0, 0.0),
    (0.02, 0.02),
    (0.02, 0.02),
)
