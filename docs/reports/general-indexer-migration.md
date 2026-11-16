# general_indexer 迁移与发布验收记录

> 日期：2026-11-14；upstream 同步复核：2026-11-16
> 分支：`wip/general-indexer-integration`
> 上游：CATLASS `wip/general_indexer`，`examples/99_ascend950_general_indexer/`，提交 `8e6ca8658e6683defd175aaed09a0f9b4c449ad4`
> 本地 CATLASS：`6c0a3242b2effd861ac91aab1999f2226b643614`（迁移源码来自上游 `8e6ca865...`）

## 1. 交付范围

`flashinfer.general_indexer` 已接入逻辑算子、Ascend C AOT、ACLNN/Torch runner、dispatcher 和生产发布清单。固定三输出为 `(indices, values, scores)`，`graph_capability=NOT_CAPTURABLE`，不声明性能优势。

当前保守支持域：

- SoC：Ascend950。
- Q/K：FP16、BF16；W：FP16、BF16、FP32。
- 布局：SND/TND/BSND/PA_BBND；PA 必须显式提供 `max_seq_len_k`。
- D=16..64、N1=1..64、B≤16、k≤8192、Tq/Tk≤16384。
- 完整 `Tq×Tk` FP32 score scratch 上限 1 GiB。

关键实现位置：

- 公共契约与校验：`flashinfer/operators/attention/general_indexer/operator.py:47`、`:243`、`:282`、`:331`。
- AOT 声明：`flashinfer/ascendc/attention/general_indexer/implementation.py:28`、`:68`。
- layout/scratch/PT 适配：`flashinfer/ascendc/attention/general_indexer/runner.py:43`、`:133`。
- Host tiling/schema：`csrc/kernels/attention/general_indexer/op_host/general_indexer.cpp:179`、`:214`。
- Device entry/tiling registration：`csrc/kernels/attention/general_indexer/op_kernel/general_indexer.cpp:17`、`:21`。

生产登记仍保持精确集合：

- Ascend C 生产项：6。
- descriptor：87。
- 逻辑 op：49。
- stack：Ascend C 6、CANN 41、reference 38、Triton 2。
- 新增 kernel id：`ascendc/attention/general_indexer/catlass/op_package/Ascend950`。

## 2. 环境与设备自检

实际执行：

```bash
source /home/void_ptr/Ascend/cann-9.3.0/set_env.sh
/home/void_ptr/miniconda3/envs/py311/bin/python -B -c \
  "import torch,torch_npu;print(torch.npu.is_available(), torch.npu.device_count())"
```

实际输出：

```text
True 8
```

CATLASS checkout 自检：

```text
git -C /tmp/catlass-pinned-plan.Oq9ZND rev-parse HEAD
41bf90da655bba3c66d0acd7e00abe33960ecfd6
```

## 3. 提交记录

本阶段相关提交：

```text
c554d6e test(general_indexer): lock process-local release environment
afa7fd8 test(general_indexer): isolate real edge cases by process
f3b73e5 fix(general_indexer): restore release fixture environment
08fdaf5 test(general_indexer): make real edge tests self-contained
8f39f48 fix(general_indexer): close arch35 source header contract
bcbd8bc test(release): update exact logical operator count
76af687 chore(release): add general indexer to exact production set
a6edc81 test(release): expect general indexer production inventory
79a7de1 refactor(general_indexer): remove dead tiling state and lint findings
cbde243 test(general_indexer): cover P3 contracts and P4 edge matrix
20ce8cd feat(general_indexer): add AOT runner and ACLNN contract
cf83f1b feat(general_indexer): migrate Ascend950 kernel and host tiling
9e47eed feat(general_indexer): add logical operator contract
e06a37d test(general_indexer): define public contract red tests
```

## 4. 定向真机验收

实际执行：

```bash
export FLASHINFER_CATLASS_ROOT=/tmp/catlass-pinned-plan.Oq9ZND
export FLASHINFER_AOT_CACHE_DIR=<本轮全新 AOT cache>
python -B -m pytest -p no:cacheprovider -q \
  tests/test_general_indexer.py \
  tests/test_general_indexer_real.py \
  tests/test_operator_packages_real.py \
  -k general_indexer
```

实际结果：

```text
37 passed, 12 deselected, 14 warnings in 173.62s
PYTEST_EXIT=0
```

覆盖：

- 公共契约、输出形状/dtype、FakeTensor、精确生产 inventory。
- TND 非均匀、非 8 对齐、mask3、seqused。
- PA 乱序页表、非法页、尾页。
- S2=8448、稳定 tie-break。
- TopK=8192、D=64/N1=64。
- AOT 命中、queue 0/1/2、out 身份、非默认 stream、缺包/损坏产物明确失败。

full 回归首次运行暴露了测试隔离问题：父进程此前已有 NPU 调用时，CANN 已冻结 custom OPP 搜索路径，之后再设置 `ASCEND_CUSTOM_OPP_PATH` 太晚。四组真机 case 因此改为由 `tests/test_general_indexer_real.py:28` 启动全新解释器，实际 case 位于 `tests/general_indexer_acceptance.py:40`、`:65`、`:94`、`:116`。随后用“先跑其他 NPU 用例，再跑 general_indexer”的同进程顺序复验：

```text
tests/test_a5_npu.py + tests/test_general_indexer_real.py
20 passed, 1 warning in 162.88s
```

## 5. AOT 与 wheel

生成与冻结：

```bash
python -B -m flashinfer.operators generate
python -B -m flashinfer.aot --freeze --all \
  --dependency-root catlass=/tmp/catlass-pinned-plan.Oq9ZND
```

两条命令退出码均为 0；generate 无 diff，冻结后的 general_indexer spec 为：

```text
e856e24e7794bb4654e44b3c53037111c3d6ae1522c5af6e3e8d876926fc0aaa
```

全新 cache 构建全部生产 AOT：

```bash
python -B -m flashinfer.aot --all \
  --dependency-root catlass=/tmp/catlass-pinned-plan.Oq9ZND \
  --cache-dir /tmp/fi-p5-aot-cache.Crgvld \
  --output-dir /tmp/fi-p5-aot-release.AiwV5J
```

实际汇总：

```text
{"summary": {"requested": 6, "built": 6, "failed": 0, "skipped": 0}}
AOT_ALL_EXIT=0
```

正式 wheel：

```text
/tmp/fi-p5-wheel.B7Tyct/flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl
size=6120590
sha256=3c27907afaf43dec8c2ed3f98a4b17119ce9693a2f947bac2dd62a1bbdb33a0c
```

wheel 内检查：

- 6 个 op-package manifest。
- general_indexer spec `e856e24e7794bb4654e44b3c53037111c3d6ae1522c5af6e3e8d876926fc0aaa` 存在。
- `flashinfer/data/aclnn/libflashinfer_aclnn.so` 与 manifest 存在。

## 6. 干净安装与 selftest

使用全新 venv，不使用 `--system-site-packages`；只通过纯路径 `_deps.pth` 提供基础依赖。实际导入位置：

```text
exe=/tmp/fi-p5-venv-final.dFnnNZ/bin/python
flashinfer=/tmp/fi-p5-venv-final.dFnnNZ/lib/python3.11/site-packages/flashinfer/__init__.py
npu=True 8
```

A950 四项完整测试：

```bash
FLASHINFER_NPU_JIT_DISABLE=1 python -B -m flashinfer selftest \
  --kernel-id ascendc/t4/flash_attention/catlass/op_package/Ascend950 \
  --kernel-id ascendc/gemm/matmul/native/op_package/Ascend950 \
  --kernel-id ascendc/t2/matmul_bias_silu/native/op_package/Ascend950 \
  --kernel-id ascendc/attention/general_indexer/catlass/op_package/Ascend950
```

复跑结果：

```text
Summary: 4 PASS, 0 FAIL, 0 SKIP (total=4)
SELFTEST_A950_RERUN_EXIT=0
```

general_indexer 的单项证据：

```text
PASS attention.general_indexer: bundled AOT hit; compilation=disabled;
CPU reference matched (author CPU reference matched;
max_abs=[0.0, 2.9802322387695312e-08, 2.9802322387695312e-08]);
spec=e856e24e7794bb4654e44b3c53037111c3d6ae1522c5af6e3e8d876926fc0aaa
```

无过滤运行全部六项时，实际结果为 `4 PASS, 2 FAIL, 0 SKIP (total=6)`：两个 Ascend910B 目标在 Ascend950 上按预期因 SoC 不匹配被拒绝。该命令退出码为 1，不作为 A950 发布的绿色门槛。

首次执行 A950 四项时曾观察到 `gemm.matmul` 子进程在打印 PASS 后以 `-11` 退出；单独复跑和随后四项复跑均通过。该瞬态退出未完成根因定位，保留为未解释风险。

## 7. 回归

`quick` 最终结果：

```text
2306 passed, 143 deselected, 15 warnings, 404 subtests passed in 43.35s
{"exit_code": 0, "errors": 0, "failures": 0, "passed": 2710,
 "profile": "quick", "skipped": 0, "tests": 2710}
```

`full` 实际执行结果为：

```text
2337 passed, 85 failed, 6 skipped, 21 errors, 404 subtests passed in 976.13s
{"exit_code": 1, "errors": 21, "failures": 85, "passed": 2741,
 "profile": "full", "skipped": 6, "tests": 2853}
```

该次 full 不能作为 P5 绿色验收证据：

- 当次 general_indexer 的 4 个失败已按第 4 节修复并完成定向复验；full 未在修复后重跑。
- 其余失败未逐项归因，范围包含与本迁移无关的真机/编译项；日志中至少可见缺少 `triton` 的 JIT 用例。
- 6 个 skip 按仓库规则不计通过。

## 8. 未验证项

- Ascend910B 真机上的 GeneralIndexer：不适用；当前目标仅为 Ascend950。
- 图 capture/replay：未验证，能力声明保持 `NOT_CAPTURABLE`。
- 性能：未做同口径测量，不声明性能优势。
- 多进程/多卡并发：未验证。
- D=128、N1>64：未支持；不得由当前保守 envelope 外推。
- 修复测试隔离后的全量 `full`：尚未重跑；当前只有定向真机与 quick 通过。

## 9. eager 边界补强与 full 复核（2026-11-15）

本轮按 TDD 补齐 eager 路径边界，修复以下问题：

- flat/SND/TND 只传 `seqused_q` 而不传 `cu_seqlens_q` 时，request query length 使用 `Tq`：`flashinfer/operators/attention/general_indexer/operator.py:292`。
- `k <= 每个 request key length` 检查覆盖 SND/TND/BSND/PA，而非仅 PA：`operator.py:316`。
- PA_BBND 4D key `[pages, block, 1, D]` 在 native admission 前归一化：`flashinfer/ascendc/attention/general_indexer/runner.py:189`。
- 零长度 key request 也发出 TopK tile，并写 `indices=-1`、`values=-inf`：`csrc/kernels/attention/general_indexer/op_kernel/block/block_scheduler.hpp:62`、`:128`，`block_topk.hpp:317`。

新增 eager 真机用例：

- flat `seqused_q` 无 cu。
- PA_BBND 4D key + batch offset。
- 默认 backend + hidden outputs + out identity。
- 零长度 key request。

定向验证：

```text
tests/test_general_indexer.py \
  tests/test_general_indexer_real.py \
  tests/test_operator_packages_real.py \
  -k general_indexer
46 passed, 12 deselected, 0 skipped
```

零长度 request 另在重建 wheel 的干净安装环境中通过；运行时未设置
`FLASHINFER_AOT_PATHS`、`FLASHINFER_ACLNN_ADAPTER_PATH`，且
`FLASHINFER_NPU_JIT_DISABLE=1`。

当前 general_indexer spec：

```text
326a2ba71d2a97dfee4e425338784c5ec6d48ab65de9b6f200fa03daf4144700
```

候选 wheel：

```text
/tmp/fi-gi-step3-wheel.8yHrbE/dist/flashinfer-0.1.0.dev0-py3-none-linux_x86_64.whl
sha256=e0431272b9eb27a3683f471835af653021571cf6d690e839652a85ecbaee471b
```

仓库级 full 已重跑，日志目录
`build/regression/20261115-031927-073380/`，实际结果：

```text
2403 passed, 28 failed, 21 errors, 6 skipped, 404 subtests passed in 1216.22s
exit_code=1
```

full 内全部 46 个 general_indexer 用例通过，无 general_indexer 失败或 skip。
其余 49 个 failure/error 的分类合计与总数一致：

- target 不适用：17。Ascend910B 的生产实现或 sparse A2 用例在 Ascend950 上按目标筛选失败。
- ACLNN adapter 路径缺失：10。legacy AOT/flash/matmul 子进程未继承
  `FLASHINFER_ACLNN_ADAPTER_PATH`；补设该路径后，抽验
  `tests/test_flash_attention_real.py::test_flash_attention_golden_multiple_shapes` 为 `1 passed`。
- `torch_npu.npu.set_task_queue_enable` 不存在：9。
- `triton` 未安装：8。
- Inductor 不支持 NPU：5。

6 个 skip 全部为 `could not import 'triton': No module named 'triton'`。

因此 P5 的 full 门槛仍未关闭，但当前证据不支持把剩余失败归因为
general_indexer 迁移回归。未验证项仍包括图 capture/replay、性能、多进程/多卡并发，
以及 Ascend910B 真机 general_indexer。

## 10. upstream 同步后的 rebase 复核（2026-11-16）

同步目标：

```text
main / upstream/main: 7bfb179 Sync upstream/infer at 4ff3c4b
CATLASS submodule: 6c0a3242b2effd861ac91aab1999f2226b643614
```

本地 general_indexer 分支已 rebase 到该基线，并解决：

- production inventory 从 upstream 的 `9/90` 与分支的 `6/87` 合并为 `10/91`。
- logical op / descriptor 精确计数更新为 `49/91`。
- 删除 upstream 已移除的旧文档和 generated coverage 冲突文件。
- Ascend C 入口改为直接的 `fi_general_indexer`。
- 移除测试对已删除 `enable_aclnn()` 的调用。
- 更新支持矩阵、wheel 范围、第三方来源声明和精确 release 断言。

生成与冻结：

```text
python -m flashinfer.operators generate: PASS
python -m flashinfer.operators check: PASS
python -m flashinfer.aot --freeze --all: PASS
general_indexer spec:
9e8a18650e06b1b24f3327b8ce7888cd4c34c3c8f830bcd14ea74e3a259f469c
```

AOT 冷构建：

```text
requested=1, built=1, failed=0, skipped=0
```

定向验证：

```text
tests/test_general_indexer.py + tests/test_operator_layout.py + tests/test_release_checks.py:
43 passed

general_indexer real + operator package:
13 passed，随后单个 topk_upper_bound_and_d64_n64 在独立复跑中 1 passed
```

合并矩阵第一次运行中 `topk_upper_bound_and_d64_n64` 子进程达到 300 秒超时；当时机器上有其他用户的大量 bisheng/CANN 编译任务。该 case 独立复跑通过，未观察到数值断言失败。

尚未在 rebase 后的新打包结构上重建 wheel，也尚未重跑仓库级 full。当前结论不包含性能、图 capture/replay 或多卡并发。

## 11. sparse_indexer family 重构验收（2026-11-16）

本次将 general_indexer 从 attention 逻辑域迁移到 `sparse_indexer` family：

```text
logical op: sparse_indexer.general_indexer
public API: flashinfer.general_indexer
kernel_id: ascendc/sparse_indexer/general_indexer/catlass/op_package/Ascend950
```

`flashinfer.general_indexer` 名称、签名、固定三输出和数值语义保持不变。
`lightning_indexer` 只作为未来同 domain 的命名空间预留；本次没有创建、注册或构建。

目录：

```text
flashinfer/operators/sparse_indexer/general_indexer/
flashinfer/ascendc/sparse_indexer/general_indexer/
csrc/kernels/sparse_indexer/general_indexer/
csrc/kernels/sparse_indexer/common/include/sparse_indexer_align.hpp
csrc/kernels/sparse_indexer/common/op_kernel/block/
```

Block 解耦：

- Scheduler、KeyLoader、BlockMmad、BlockReduceHead、BlockTopK 均定义运行时 `Params`。
- general_indexer kernel 入口负责把 `GeneralIndexerTilingData` 投影为 block Params。
- `sparse_indexer/common` 不包含 `GeneralIndexerTilingData` 或 concrete tiling header。
- common namespace 统一为 `SparseIndexer::Common`。
- BuildConfig 支持 `shared_native_roots/shared_includes`；公共头进入 recipe fingerprint、staging 和 wheel。
- 修复 textual `#include` 的 `.cpp` 文件未进入 native closure 的回归。

生成与 AOT：

```text
generate/check: PASS
freeze --all: PASS
general_indexer spec:
275c240f5e4ead3a49072cc69ae4b1183216d99f03f2eb932f92b0ff436caa31
AOT cold build: requested=1, built=1, failed=0, skipped=0
```

定向验证：

```text
test_general_indexer.py + test_operator_layout.py +
test_sparse_indexer_blocks.py + test_shared_native_inputs.py: 50 passed
test_general_indexer_real.py: 8 passed
test_operator_packages_real.py -k general_indexer: 6 passed
test_ascendc_entries.py: 20 passed
test_sparse_indexer_contract.py: 2 passed
```

Producer-consumer 契约：

- general_indexer 输出 token TopK `[Tq,k]`。
- sparse attention 需要 KV block 索引 `[B,Sq,1,Ns]` 或 `[Tq,1,Ns]`，尾部 `-1`。
- 两者必须显式 token→block 转换、去重和填充，不是直接同形接口。
- 当前 general_indexer 为 Ascend950，现有 SparseFlashAttention 为 Ascend910B/Ascend910C；
  同 SoC 生产级端到端真机尚未验证。

Quick：

```text
2373 passed, 170 deselected, 0 skipped, 0 failed
exit_code=0
```

Ascend950 wheel：

```text
/tmp/fi-sparse-indexer-wheel.nyRpQE/dist/flashinfer_ascend950-0.1.0.dev0-py3-none-linux_x86_64.whl
sha256=7aae8903af41b310487c11db85203da6316d3fbe55051caa38e69ebdffbd928a
wheel.json kernel_ids=4，包含 sparse_indexer/general_indexer
```

干净 venv 安装：

```text
flashinfer.__file__=<venv>/site-packages/flashinfer/__init__.py
Scope: Ascend950; 4 of 4 packaged kernels; other SoCs are not runtime-validated
Summary: 4 PASS, 0 FAIL, 0 SKIP (total=4)
PASS sparse_indexer.general_indexer: bundled AOT hit; compilation=disabled
```

Full：

```text
build/regression/20261116-122209-387812/
2485 passed, 31 failed, 21 errors, 6 skipped, 404 subtests passed
exit_code=1
```

full 内全部 56 个 sparse_indexer/general_indexer 相关用例通过，无失败、无 skip。
其余 52 个 failure/error：

- target 不适用：17。
- `torch_npu.npu.set_task_queue_enable` 不存在：9。
- `triton` 未安装：8。
- ACLNN adapter 路径缺失：7。
- Inductor 不支持 NPU：5。
- 上游 CATLASS golden fixture digest 与冻结常量不一致：6。

6 个 skip 全部为 `No module named 'triton'`。

因此本次 sparse_indexer family 重构本身无已知回归，但 full 绿色门槛仍受环境、torch_npu/Inductor 能力和目标不适用问题阻塞。
未验证项：lightning_indexer、同 SoC indexer→sparse attention 真机端到端、图 capture/replay、
性能、多进程/多卡并发，以及修复环境后的 full 绿色结果。

## 12. kernel layout cleanup 验收（2026-11-16）

参考 `attention/sparse_flash_attention` 后完成以下结构收敛：

```text
op_kernel/
├── general_indexer.cpp
└── arch35/
    ├── general_indexer.h
    └── general_indexer_kernel.h

op_host/
├── general_indexer_def.cpp
├── general_indexer_infershape.cpp
├── general_indexer_host.h
└── arch35/
    └── general_indexer_tiling.cpp

common/
├── include/sparse_indexer_common.hpp
└── op_kernel/block/
```

清理内容：

- 删除只做转发的 `general_indexer_arch35.h`。
- 删除未使用的 `GeneralIndexerEntryContiguous/Paged` 模板。
- 删除 entry 中重复的 `block/key_loader.hpp` include。
- host 按 OpDef、InferShape/Type、arch35 tiling 拆分。
- `sparse_indexer_align.hpp` 改为 `sparse_indexer_common.hpp`。
- common block include guard 改为 `FLASHINFER_SPARSE_INDEXER_COMMON_*`。
- tiling data 删除不再使用的 CATLASS macro include。
- BuildConfig 的 host 闭包同步为三个 host source。

最终 spec：

```text
92e13b6978d4e553a00a5bd07093a386b8737bbc267dd644b2af8152412f86ff
```

验证：

```text
generate/check/freeze: PASS
general_indexer AOT cold build: built=1, failed=0, skipped=0
test_sparse_indexer_blocks.py: 5 passed
test_general_indexer_real.py: 8 passed
quick: 2375 passed, 170 deselected, 0 skipped, 0 failed
```

最终 Ascend950 wheel：

```text
/tmp/fi-si-cleanup-wheel.BGMTFA/dist/flashinfer_ascend950-0.1.0.dev0-py3-none-linux_x86_64.whl
sha256=5dc5ebb89b58ea6edd50fe19553c3152dd33baff27025c694aea966c561c1ada
```

干净 venv 安装后 selftest：

```text
Scope: Ascend950; 4 of 4 packaged kernels
Summary: 4 PASS, 0 FAIL, 0 SKIP (total=4)
PASS sparse_indexer.general_indexer: bundled AOT hit; compilation=disabled
spec=92e13b6978d4e553a00a5bd07093a386b8737bbc267dd644b2af8152412f86ff
```

本次清理未重跑 full；最近一次 full 为 `2485 passed / 31 failed / 21 errors / 6 skipped`，
其失败项已归因为环境、目标不适用和上游 fixture 问题；所有 sparse_indexer 相关用例通过。
