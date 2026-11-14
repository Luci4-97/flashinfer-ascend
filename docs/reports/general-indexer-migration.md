# general_indexer 迁移与发布验收记录

> 日期：2026-11-14
> 分支：`wip/general-indexer-integration`
> 上游：CATLASS `wip/general_indexer`，`examples/99_ascend950_general_indexer/`，提交 `8e6ca8658e6683defd175aaed09a0f9b4c449ad4`
> 本地 CATLASS：`41bf90da655bba3c66d0acd7e00abe33960ecfd6`

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
