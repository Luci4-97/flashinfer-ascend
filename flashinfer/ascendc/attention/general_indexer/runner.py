"""Runner adaptation from the public layout contract to the flat native kernel."""

from dataclasses import dataclass

from flashinfer.jit.tensor_adapter import package_tensor_runner


def _squeeze_n2(tensor):
    if tensor.dim() == 4 and tensor.size(2) == 1:
        return tensor.squeeze(2)
    if tensor.dim() == 3 and tensor.size(1) == 1:
        return tensor.squeeze(1)
    return tensor


def _row_batch_ids(torch, cu_seqlens, tq, device):
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(lengths.numel(), dtype=torch.int32, device=device),
        lengths,
        output_size=tq,
    )


def _localize_indices(torch, indices, batch_ids, s2):
    flat = indices.reshape(-1, indices.shape[-1])
    valid = flat >= 0
    localized = flat - batch_ids.to(torch.int32).unsqueeze(1) * s2
    flat.copy_(torch.where(valid, localized, flat))


def _copy_block_diagonal(torch, native_scores, public_scores, cu_q, cu_k, batch, s2):
    q = cu_q.cpu().tolist()
    k = cu_k.cpu().tolist()
    rows = public_scores.reshape(-1, s2)
    for index in range(batch):
        q0, q1 = q[index], q[index + 1]
        k0 = k[index]
        rows[q0:q1].copy_(native_scores[q0:q1, k0:k0 + s2])


@dataclass(frozen=True)
class GeneralIndexerRunner:
    native: object

    def __call__(self, query, key, weights, block_table, cu_seqlens_q,
                 cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k,
                 output_idx_offset, indices, values, scores, **attributes):
        import torch

        from flashinfer.operators.attention.general_indexer.operator import _meta

        topk = int(indices.shape[-1])
        arguments = {
            "query": query, "key": key, "weights": weights, "k": topk,
            "layout_q": attributes["layout_q"], "layout_k": attributes["layout_k"],
            "mask_mode": attributes["mask_mode"], "cmp_ratio": attributes["cmp_ratio"],
            "max_seq_len_k": attributes["max_seq_len_k"],
            "cu_seqlens_q": cu_seqlens_q, "cu_seqlens_k": cu_seqlens_k,
            "seqused_q": seqused_q, "seqused_k": seqused_k,
            "cmp_residual_k": cmp_residual_k, "block_table": block_table,
            "output_idx_offset": output_idx_offset,
            "return_value": attributes["return_value"],
            "return_score": attributes["return_score"],
        }
        meta = _meta(arguments)
        device = query.device
        batch, tq, n1, d = meta.batch, meta.tq, meta.n1, meta.d
        s2 = meta.s2
        is_paged = int(meta.layout_k == "PA_BBND")

        if meta.layout_q == "BSND":
            query_flat = query.contiguous().reshape(tq, n1, d)
            weights_flat = weights.contiguous().reshape(tq, n1)
            if meta.layout_k == "BSND":
                key_work = _squeeze_n2(key).contiguous()
                key_flat = key_work.reshape(batch, meta.s2, d).reshape(-1, d)
            else:
                key_flat = _squeeze_n2(key).contiguous()
            cu_q = torch.arange(0, batch * meta.s1 + 1, meta.s1,
                                dtype=torch.int32, device=device)
            if is_paged:
                cu_k = torch.arange(0, batch * s2 + 1, s2,
                                    dtype=torch.int32, device=device)
            else:
                cu_k = torch.arange(0, batch * meta.s2 + 1, meta.s2,
                                    dtype=torch.int32, device=device)
        else:
            query_flat = query.contiguous()
            weights_flat = weights.contiguous()
            key_flat = _squeeze_n2(key).contiguous()
            cu_q = cu_seqlens_q
            cu_k = cu_seqlens_k
            if is_paged:
                s2 = int(attributes["max_seq_len_k"])
                if cu_q is None:
                    cu_q = torch.tensor([0, tq], dtype=torch.int32, device=device)
                cu_k = torch.arange(0, batch * s2 + 1, s2,
                                    dtype=torch.int32, device=device)

        tk_native = batch * s2 if (meta.layout_q == "BSND" or is_paged) else s2
        zero_page = None
        if is_paged:
            block_size = int(key.shape[1])
            zero_page = torch.zeros((block_size, d), dtype=query.dtype, device=device)

        native_indices = indices.reshape(tq, meta.k)
        if attributes["return_value"]:
            native_values = values.reshape(tq, meta.k)
        else:
            native_values = torch.empty((tq, meta.k), dtype=torch.float32, device=device)

        public_flat_scores = (
            attributes["return_score"]
            and meta.layout_q != "BSND"
            and (not is_paged or batch == 1)
        )
        if public_flat_scores:
            native_scores = scores.reshape(tq, tk_native)
        else:
            native_scores = torch.empty((tq, tk_native), dtype=torch.float32, device=device)
        native_scores.fill_(float("-inf"))
        if device.type == torch._C._get_privateuse1_backend_name():
            # Scratch initialization and the native launch must be ordered even
            # when the caller uses a non-default TorchNPU stream/queue.
            torch.npu.current_stream().synchronize()

        native_arguments = (
            query_flat, key_flat, weights_flat, block_table, cu_q, cu_k,
            seqused_q, seqused_k, cmp_residual_k, output_idx_offset, zero_page,
            native_indices, native_values, native_scores,
        )
        self.native(*native_arguments, k=topk, is_paged=is_paged,
                    mask_mode=attributes["mask_mode"], cmp_ratio=attributes["cmp_ratio"],
                    max_seq_len_k=s2 if is_paged else 0)

        if meta.layout_q == "BSND":
            batch_ids = torch.repeat_interleave(
                torch.arange(batch, dtype=torch.int32, device=device),
                torch.full((batch,), meta.s1, dtype=torch.int64, device=device),
                output_size=tq,
            )
            _localize_indices(torch, indices, batch_ids, s2)
        elif is_paged and batch > 1:
            _localize_indices(
                torch, indices, _row_batch_ids(torch, cu_q, tq, device), s2
            )

        if attributes["return_score"] and not public_flat_scores:
            if meta.layout_q == "BSND":
                public = scores.reshape(batch, meta.s1, s2)
                _copy_block_diagonal(torch, native_scores, public, cu_q, cu_k, batch, s2)
            else:
                _copy_block_diagonal(torch, native_scores, scores, cu_q, cu_k, batch, s2)
        if device.type == torch._C._get_privateuse1_backend_name():
            # Hidden values/score scratch is owned by this eager runner. Keep it
            # alive until the enqueued native operation has consumed it.
            torch.npu.current_stream().synchronize()


def admission(recipe, signature):
    from dataclasses import replace

    from flashinfer.ascendc._runtime import native_admission

    names = recipe.tensor_parameter_names
    positions = {name: index for index, name in enumerate(names)}
    try:
        specializations = dict(signature.specializations)
        is_paged = specializations.get("attribute.is_paged") == "1"
        layout_q = specializations.get("attribute.layout_q", "").upper()
        layout_k = specializations.get("attribute.layout_k", "").upper()
        hidden = positions["score_zero_page"]
        dtypes = list(signature.dtypes)
        shapes = list(signature.shapes)
        physical = list(signature.physical_layouts)
        contiguity = specializations.get("tensor_contiguity", "")
        q_dtype = dtypes[positions["query"]]
        query = shapes[positions["query"]]
        key = shapes[positions["key"]]
        if layout_q == "BSND":
            batch, s1, n1, d = query
            shapes[positions["query"]] = (batch * s1, n1, d)
            shapes[positions["weights"]] = (batch * s1, n1)
        if layout_k == "BSND":
            batch, s2 = key[0], key[1]
            shapes[positions["key"]] = (batch * s2, key[-1])
            key = shapes[positions["key"]]
        elif layout_k == "PA_BBND" and len(key) == 4 and key[2] == 1:
            shapes[positions["key"]] = (key[0], key[1], key[3])
            key = shapes[positions["key"]]
        dtypes.insert(hidden, q_dtype if is_paged else "none")
        shapes.insert(hidden, key if is_paged else ())
        physical.insert(hidden, "ND" if is_paged else "")
        if contiguity:
            contiguity = contiguity[:hidden] + ("1" if is_paged else "x") + contiguity[hidden:]
            specializations["tensor_contiguity"] = contiguity
        signature = replace(
            signature,
            dtypes=tuple(dtypes),
            shapes=tuple(shapes),
            physical_layouts=tuple(physical),
            specializations=tuple(sorted(specializations.items())),
        )

        indices = positions["indices"]
        values = positions["values"]
        scores = positions["scores"]
        tq = signature.shapes[positions["query"]][0]
        topk = signature.shapes[indices][-1]
        if is_paged:
            batch = 1
            cu = positions["cu_seqlens_q"]
            if signature.dtypes[cu] != "none":
                batch = signature.shapes[cu][0] - 1
            tk = batch * int(specializations.get("attribute.max_seq_len_k", "0"))
        else:
            tk = signature.shapes[positions["key"]][0]
        native_shapes = list(signature.shapes)
        native_shapes[indices] = (tq, topk)
        native_shapes[values] = (tq, topk)
        native_shapes[scores] = (tq, tk)
        signature = replace(signature, shapes=tuple(native_shapes))
    except (IndexError, KeyError, TypeError, ValueError) as error:
        return ("general_indexer requires complete native Tensor shapes: %s" % error,)

    reasons = list(native_admission(recipe, signature))
    if reasons:
        return tuple(reasons)
    if tq * tk * 4 > (1 << 30):
        reasons.append("Tq*Tk*4 exceeds the 1 GiB score workspace limit")
    if is_paged:
        max_seq = int(specializations.get("attribute.max_seq_len_k", "0"))
        if max_seq <= 0:
            reasons.append("paged execution requires max_seq_len_k")
    return tuple(reasons)


def make_runner(module, spec, recipe):
    return GeneralIndexerRunner(package_tensor_runner(module, recipe, spec=spec))
