# Sparse Indexer 接口与实现范围

面向集成者：本页说明 `sparse_indexer` 逻辑 domain 的当前成员、公共接口、与 sparse attention 的组合边界，以及尚未实现的未来变体。

## 当前 family

公共 domain 为 `sparse_indexer`。本次只实现一个成员：

| logical op | public API | 状态 |
| --- | --- | --- |
| `sparse_indexer.general_indexer` | `flashinfer.general_indexer` | 当前 A950 eager AOT 实现 |

`sparse_indexer.lightning_indexer` 仅作为未来同 domain 的命名空间预留，本次不创建、不注册、不构建。

## GeneralIndexer

公共入口为 `flashinfer.general_indexer`，固定返回 `(indices, values, scores)`；未请求的可选输出返回形状 `(0,)` 的 FP32 Tensor。计算为按 head 加权 `ReLU(Q @ K^T)` 后执行稳定 TopK，支持 SND/TND/BSND 与 PA_BBND 布局。

支持 Q/K 的 FP16/BF16 同 dtype 组合，W 支持 FP16/BF16/FP32；Ascend950 上 D=16..128、N1=1..64、B≤16、k≤8192、Tq/Tk≤16384，并要求连续 ND 输入。PA_BBND 必须提供 `block_table`、`seqused_k` 和 `max_seq_len_k`。

该实现为 eager AOT-only，图能力保持 `NOT_CAPTURABLE`；未做性能声明。源码：[公共定义](../../flashinfer/operators/sparse_indexer/general_indexer/operator.py)、[AOT/runner](../../flashinfer/ascendc/sparse_indexer/general_indexer/)、[设备实现](../../csrc/kernels/sparse_indexer/general_indexer/)、[common blocks](../../csrc/kernels/sparse_indexer/common/op_kernel/block/)。

## 与 Sparse Attention 的契约

Indexer 与 sparse attention 是两个 domain：

```text
sparse_indexer.general_indexer
        ↓ indices / values / scores
attention.sparse_flash_attention
```

GeneralIndexer 当前输出 token TopK 形状 `[Tq,k]`；SparseFlashAttention 的 `sparse_indices` 要求 `[B,Sq,1,Ns]` 或 `[Tq,1,Ns]` 的 KV block 编号，尾部用 `-1` 填充。因此二者不是直接同形接口，必须显式执行 token→block 转换、去重和尾部填充。

当前 general_indexer 目标是 Ascend950，现有 sparse_flash_attention 生产目标是 Ascend910B/Ascend910C；同 SoC 的生产级端到端真机组合尚未验证。当前只固化 CPU/API 契约测试，不声称端到端支持。

## 验证与限制

- 公共契约、AOT 路径和 block 无 concrete tiling 依赖来自 GeneralIndexer 定向测试。
- `test_sparse_indexer_contract.py` 固化 token TopK→sparse block 的转换和 SparseFlashAttention metadata 契约。
- 目标 SoC 真机端到端、图 capture/replay、性能和多卡并发未验证。
## 运行时 metadata 与校验模式

`cu_seqlens_q/k`、`seqused_q/k`、`cmp_residual_k` 属于 kernel 运行时 metadata。
host tiling 只记录 shape、presence flag 和静态调度参数；kernel 在执行时从对应 GM Tensor
读取实际值。默认 eager 路径因此不再为这些值执行 host D2H。

shape、dtype、device、optional 输入组合以及 mask 属性仍会在公共入口做结构校验。
需要额外检查 metadata 值域时，可显式传入 `validate_metadata=True`；该调试模式会保留
原有 cu/seqused/residual 范围检查，但会产生 host 读取开销。
