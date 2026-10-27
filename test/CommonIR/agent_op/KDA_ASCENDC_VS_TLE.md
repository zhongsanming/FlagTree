# KDA：AscendC 实现 vs TLE(FlagTree/Triton) 实现

对比对象：

* AscendC：`01-glm53-ascend/sgl-glm53-ascend-ops/csrc/kda/`（`kda_varlen_recurrent.cpp`
  主机侧 + `prefill_h4/h8` + `decode.h`/`decode_h16/h32/h64` + 遗留的 `prepared_recurrent.cpp`）
* Triton：`test/CommonIR/agent_op/target/glm_kda_varlen_recurrent_npu_v0.py`

> 先纠正一个前提：**本仓库里的 KDA 移植不是 TLE**，`glm_kda_varlen_recurrent_npu_v0.py`
> 只 import 了 `triton` / `triton.language`。同目录的 `causal_conv1d_fn.py`、
> `causal_conv1d_update.py`、`swiglu.py` 都写了 `import triton.experimental.tle as tle`。
> KDA 是这批移植里唯一没走 TLE 的，`agent_op/origin/` 里也没有 KDA 的 origin。
> 所以下面的对比实际上是“AscendC vs 纯 Triton”，而“TLE 化”本身仍是一个待做的动作。

当前结论基于**阅读代码**，不含任何在目标机上的实测（本地无 NPU）。文中引用的所有
性能数字都注明了出处，均来自注释里的既有测量。

---

## 0. 目标机与优先级

* **8 卡 910B 节点**，单卡 40 个 AIV，与 `kda_varlen_recurrent.h:95` 的
  `kNumVectorCores = 40` 一致；CANN 9.1.0；FlagTree triton 3.5.x。
* KDA 是单 device 算子：一次 launch 只作用于一张卡。**“8 卡”只影响选卡
  （`ASCEND_RT_VISIBLE_DEVICES` / `npu:N`），不影响 tiling**，所以下文一律按
  “单卡 40 AIV”推导。
* 优先级：prefill H=4/8；不要求与上游位精确。

上一轮把“8 核”误读成 8 个 AIV，因此有两条结论必须撤回：

1. **AscendC 的融合 prefill 内核（40 block = 32 consumer + 8 producer）在这台机器上是可以
   跑的**，而且正好是一个 wave（40 block = 40 AIV）。它是真实可用的对照基线，
   `bench/kda_recurrent/bench_kda_varlen_recurrent.py` 的 AscendC 那一臂可以正常执行。
2. **“H=8 用 BV=128”是 8 核的产物，不适用于 40 AIV**。40 AIV、H=8 的正解是
   BV=32（32 block、一个 wave），推导见 §5.5。BV=128 在 H≤8、40 AIV 上永远不是正解
   （block 数 ≤8，占用 ≤20%），所以“BV=128 编译不过”对本机 prefill **不构成阻塞**。

---

## 1. 实现清单

| | AscendC | Triton (`_v0.py`) |
|---|---|---|
| prefill H=4 | `prefill_h4.cpp`：单 launch、40 block = 32 consumer（4 head × 8 V-tile，BV=16）+ 8 producer（2/head）提前算 gate/beta/qk 归一化；chunk=16 token；`IBSet/IBWait` 环形 bank 深度 4 | `_opt_recurrent_kernel`：grid `(D//BV, nseq, H)`，BV autotune {16,32,64,128}×num_warps×num_stages，prep 在**每个 V-tile program 内重算** |
| prefill H=8 | `prefill_h8.cpp`：同一 40-block grid，每个 consumer 把 token 循环跑两轮（两个 head 轮次），跨轮次预取下一轮首 token | 同一 kernel，`H=8` → 依 BV 得 64（BV=16）或 32（BV=32）个 block |
| decode / verify H=16/32/64 | `decode.h`：**每 (request, head) 一个 block**，BV=128，64 KB state 常驻 UB，prep 按 `kStepGroup=4` 批处理，checkpoint store 发出后不等、与 output 归约重叠 | **退回通用 kernel** `_glm_kda_varlen_recurrent_kernel`，BV=64，每 `(V-tile, seq, head)` 一个 program |
| 遗留 | `prepared_recurrent.cpp`：输入已归一化 fp32，BV=16，state 经 `[128,24]` staging + 2×`TransDataTo5HD` | — |
| 主机侧 | `kda_varlen_recurrent.cpp`：校验、launcher 表、fp32 scratch + 清零 sync workspace、int64→int32 收窄 | 沿用上游校验、`empty_like`、`T>=64` 启发式、静默回退 |

支持矩阵：

* AscendC：prefill H∈{4,8}（numReq==1）、decode H∈{16,32,64}、`intermediate_state` 仅 decode、
  tracking 仅 prefill；要求 H==HV、K==V==128、bf16 输入、state 为 fp32 且 **K 连续**
  （`stride(2)==1, stride(3)==128`）。
* Triton：优化路径**仅 prefill**、H==HV、K==V==128、T≥64、不支持 `intermediate_state`
  （`_can_use_optimized_path`，289-317 行）。其余情况全走通用 kernel。

---

## 2. 逐项差异

### 2.1 工作划分

AscendC prefill 把“token 间独立”和“按 token 串行”拆成同一个 grid 的两类核，让 prep 跑在
本来空闲的核上（注释实测：prep 占 prepared 路径约 19%，H4/T4096 为 6.61 ms 中的 1.16 ms）。
40 AIV 上 H=4 用 BV=16 时正好是 32 block，**留下 8 个 AIV 空闲**——这就是 producer 的来源。
Triton 只有一类核，prep 塞进每个递推 program，那 8 个 AIV 只能空着。

decode 相反：AscendC 工作项是 `(request, head)`、整个 V 轴 128 宽，state 全程驻留 UB；
Triton 把 V 切半（BV=64），每个 program 重新加载 32 KB state tile。

### 2.2 内存与布局

* AscendC 手工在 UB 绝对地址上排布（`UbLayout`：`STATE_ADDR=0`、`Q0_ADDR=0x10100`…
  `UB_END≈0x21800`），区域复用（producer 住在 consumer map 下方 57 KB 的空洞里），
  且 repeat stride 由这些偏移推导。decode 用 192 KB 中约 116 KB。
* Triton 分配器装不下 `[128,128]` state + 活跃临时量：`decode.h:44-46` 记了准确报错
  `"requires 2383872 bits while 1572864 bits available"`，这就是搜索空间停在 BV=64 的原因。
  （对 prefill 而言 BV=128 本来也不是正解，见 §5.5，所以这条限制只在 decode 上真正咬人。）
* AscendC 的 state 是 V-major / K 连续，load/store 就是一次连续 `DataCopy`（`BV*128` 个 float）。
  Triton 用 `S0..SV` 表达；头文件断言另一种朝向对通用 kernel “数值上是错的”——
  该断言需要单独验证是误编译还是归约树差异（不要求位精确后影响变小，但仍影响布局选择）。
* AscendC 用 Scalar 单元读 `b`（每 token 一个 bf16、按 H 跨步），并用
  `DataCacheCleanAndInvalid` 回读 GM 中 producer 写的 `beta`，因为 Scalar 的 DCache
  与其他核的写入不一致。Triton 没有 DCCI，但 Triton 侧不存在“别的核写、本核 Scalar 读”
  的模式，所以这一条在 Triton 上不构成问题。

### 2.3 同步

AscendC：手工 event id 的 `SetFlag/WaitFlag`、`PipeBarrier<PIPE_V>`、
按 (producer, consumer lane) 配对的深度 4 `IBSet/IBWait` 信号量、以及必须存在但绝不能
`InitBuffer` 的 `TPipe`。
Triton：核内同步由编译器注入，唯一开关是 `disable_auto_inject_block_sync`；
TLE 额外提供 `sync_block_set/wait`（仅 CV、event id 0-15、计数语义）与
`sync_block_all`（整 grid 栅栏）。**没有 AIV↔AIV 的成对信号量。**

### 2.4 算术与数值

AscendC 刻意把 `sqrt` 与 beta 的倒数放到 **Scalar** 单元，因为本硬件 Vector `Div/Sqrt`
是 1 ulp 而上游是 IEEE 精确；它还对齐了 `tl.sum` 的精确归约树（注释：13 种候选树只有
一种每行都吻合）。Triton 无这些约束。

**既然不要求位精确，这些约束在 TLE 侧可以全部放弃**，并反过来变成优化机会：
使用 Vector 的 `sqrt`/倒数、把 `q/√s*scale` 折叠成 `q*(scale/√s)`（省掉每 token 一次
128 通道除法）、用 `tl.sigmoid` 等。

### 2.5 接口与契约

AscendC 拒绝它无法安全索引的东西（state stride、H 宽度、H==HV、prefill numReq==1）。
Triton 保留上游校验 + `T>=64` + 静默回退，回退面更宽（GQA、任意 K/V、intermediate、
tracking），这是它的真实优势。

---

## 3. 为什么 AscendC 的 producer/consumer 收益搬不过来

AscendC 的 19% 由三部分构成：(a) 独立 launch 的 prep 是纯串行延迟；(b) prep 走 Scalar，
每次 sqrt/倒数都是一次 V↔S 往返；(c) prepared 路径的 prep 结果以 fp32 写回 GM，
递推再读 8 份。

Triton 侧这三项都不成立：

* prep 全部落在 Vector 流水线（无 V↔S 往返）；
* prep 的算术量很小。以 BV=16、H=4 估算，每 token 递推约为
  `state*decay` + 两次 `state*k`/`muladd` + 归约 + `Brcb`(广播) + 两次 outer + 输出归约，
  约 8 个 state 级操作 × (16×128)/128 ≈ 130 条 repeat 指令量级；而 prep 是
  `q*q` 归约 + `k*k` 归约 + 若干 128 通道逐元素，约 10 条指令量级。即 P/R 大致是
  **5% 量级**，不是 19%；
* prep 在 8 个 V-tile program 间是并行重复，wall-clock 上只多算一次 prep，不是 8 次。

**结论：为了省 5% 去引入 producer/consumer + 核间信号量，在 TLE 侧不划算。**
真正可迁移的是 AscendC 提交历史里 `double buffered` 那一步——把每 token 的 DMA 延迟
藏到递推计算后面。这也正是 v1 的 `KDA_PREFETCH` / `KDA_STAGES` 选项。

残留差距（诚实记账）：H=4 与 H=8 在 40 AIV 上都只能填到 32/40 = 80% 的占用率
（§5.5）。理论上还能用最后 8 个 AIV 做 producer，但需要 AIV↔AIV 信号量，TLE 没有，
而收益按上面的估算只有 5% 量级，所以不做。

---

## 4. AscendC 相对 TLE 的能力缺口

| AscendC 原语 | TLE 对应 | 结论 |
|---|---|---|
| `IBSet/IBWait` 成对环形 | 只有 CV `sync_block_set/wait` 与整 grid `sync_block_all` | **真缺口**，但按 §3 当前不需要 |
| 手工 UB 映射、别名/重叠 | `tle.dsa.alloc`（编译器管理、无别名）+ `copy/to_tensor/to_buffer` | **部分**：可压小活跃集，不能重叠 |
| `SetFlag/WaitFlag`、event id、`PipeBarrier` | 自动注入 + `disable_auto_inject_block_sync` | **部分**：无法指定“checkpoint 的 MTE3 与下一 token 的 Vector 重叠” |
| `Muls/Add/Brcb/MulAddDst/WholeReduceSum` 带 repeat stride | `tle.dsa.add/sub/mul/div`；`tle.dsa.ascend.raw` + custom op 注册表 | **大部分可补**：`duplicate_bitwise_mask`、`pair_reduce_sum_continuous_mask`、`gather_mask_*` 已证明该模式；`MulAddDst`/`Brcb`/`WholeReduceSum` 未注册但补齐是机械工作 |
| Scalar `sqrt/div`、`GetValue/SetValue` | `tle.dsa.extract_element`，无可选标量数学 | **真缺口**，但已放弃位精确，不需要 |
| `DataCacheCleanAndInvalid` | 无 | **真缺口**；Triton 侧目前没有该使用模式，暂不需要 |
| `sub_vec_id`（910C 双 vector 子核） | `tle.dsa.ascend.sub_vec_id()` | **TLE 更强** |
| cube / MIX（矩阵乘） | `tl.dot`、`tile_cube_launch` | **TLE 更强**，是后续 chunked 算法的前提 |

---

## 5. prefill H4、H8 的优化路线（按性价比排序）

### P0 标定（先做，否则后面全是猜）

1. `bench_kda_tle.py calib`：固定每 block 工作量、扫 grid=1..64，延迟第一次跃升前的最后一个
   grid 即有效 AIV 数。预期是 40；若明显不是 40（分核/虚拟化/共卡），所有 BV 决策都要重算。
2. `bench_kda_tle.py sweep`：把 v0 的 `BV × num_stages × multibuffer` 扫一遍，确认：
   * autotune 实际选了哪个 BV；
   * `num_stages>1` / `multibuffer=True` 是否真的把 token 循环的 DMA 叠起来；
   * v0 是否已经等价于 v1 的默认配置（若如此，收益只剩 P1/P3）。

### P1 每 token 的 DMA 双缓冲（预期收益最大，可迁移自 AscendC 的 `double buffered`）

v0 在循环体里 `tl.load` q/k/a/v 四份，若编译器没有做软件流水，每个 token 都要等一次
MTE2 延迟。AscendC 明确做了“先把下一个 token 预取到 `buf[nxt]`”这一步。
Triton 侧两条路：
* (a) `KDA_STAGES` / `KDA_MULTIBUFFER` 交给后端流水线（零代码改动，先测）；
* (b) 手写一拍预取：循环外先取 `bos`，循环内先用当前值、再取 `min(i+1, seq_len-1)`
  （索引钳位、无分支）。已实现于 `target/glm_kda_varlen_recurrent_npu_v1.py` 的
  `PREFETCH` 选项。

### P2 grid 与 BV 对齐到“一波”

`tiles = pow2_floor(max(1, AIV // H))`，`BV = 128 // tiles`。推导与表格见 §5.5。

### P3 收益小但零风险的调整

* 折叠归一化：`q * (scale / sqrt(s+1e-6))`、`k * (1 / sqrt(s+1e-6))`，省掉每 token
  两次 128 通道除法（不要求位精确的前提下方可使用）。**v1 已默认开启。**
* state tile 是**连续**的（K==V==128、BV 整除 128），可用一维 `tl.arange(0, BV*D)` 取回再
  `reshape`，或 `tle.dsa.copy` 一次搬入 UB，替掉带 mask 的二维 gather（v1 的
  `FLAT_STATE` 选项；v0 注释警告无 mask 的 load 在 BV>=64 会误编译，故默认关闭，需实测）。
* 清理 autotune：`num_warps` 在 Ascend 上无语义（后端默认 32，作为 `--num-warps` 透传）；
  BV=128 若编译不过仍会被尝试，拖慢首次调用；`restore_value=["STATE"]` 会为每个 config
  快照整个 state pool（H=32 时数十 MB）。v1 已经改成无 autotune。
* 删掉死开关 `GLM53_KDA_ENABLE_FUSED`（测试 docstring 说生产走 BV=64，但 kernel 从不读它，
  autotune 会静默选任意 BV），以及未使用的 `_OPT_MIN_HEADS`。

### P4 之后的两个方向（不在本轮）

* decode/verify 的专用 kernel：`_can_use_optimized_path` 在 `prefill=False` 直接返回 False
  （289 行），所以所有 decode/verify 都在通用 BV=64 路径上。verify 是 profile 最热的场景，
  这是功能上最大的洞——但当前优先 prefill，故后置。BV=128 在 decode 上是真收益
  （`decode.h:44` 的 BV 16/32/64 → 85.2/52.6/40.3 µs），届时需要用 `tle.dsa.alloc`
  显式管 UB 来绕过分配器的 2383872 vs 1572864 bits 溢出。
* chunked delta rule（WY/UT 变换）把串行 token 循环变成矩阵乘，在 token 轴上并行。
  这才是数量级收益的来源，且依赖 cube，TLE（`tl.dot`）比手写 AscendC 容易达到得多。
  代价是数值必然改变（已可接受）以及 verify 逐步 checkpoint 语义需重新处理。

### 5.5 BV、block 数与 wave 的关系（“BV=128 编译不过时怎么选”）

模型：一个工作项 = (head, value-tile)；block 数 = `H × (128/BV) × num_seqs`；
一波最多 40 个 block；单个 block 的时间 ∝ BV。于是

```
T ≈ ceil( blocks / 40 ) × (per-token work at BV)
```

因为 `per-token work ∝ BV`，把 `ceil(...)` 压到 1 是唯一关键——**BV 越小、block 越多，
直到刚好填满一波为止**；再小就溢到第二波，时间翻倍。

H=4（block 数 = 512/BV）：

| BV | blocks | waves | 占用率 | 相对时间 |
|---|---|---|---|---|
| 16 | 32 | 1 | 80% | **1×**（正解，与 AscendC 的 32 consumer 一致） |
| 32 | 16 | 1 | 40% | 2× |
| 64 | 8 | 1 | 20% | 4× |
| 128 | 4 | 1 | 10% | 8× |

H=8（block 数 = 1024/BV）：

| BV | blocks | waves | 占用率 | 相对时间 | prep 重复份数 |
|---|---|---|---|---|---|
| 16 | 64 | 2 | 80% | 2×（= 1× 的 roofline 下界） | 8 |
| 32 | 32 | 1 | 80% | **1×** | 4 ← **正解** |
| 64 | 16 | 1 | 40% | 2× | 2 |
| 128 | 8 | 1 | 20% | 4× | 1 |

H=8 的 BV=16 与 BV=32 在 roofline 上等价（都是 80% 占用），但 BV=32 的 prep 重复份数
只有一半、block 数也只有一半（少一半 prologue：`exp(A_log)`、dt_bias 载入、state 载入），
所以 **H=8 选 BV=32**。这也解释了 AscendC 为什么在 H=8 上用 BV=16 + “每 consumer 跑两轮”：
它需要把 block 数压到 32 以内，才能把剩下的 8 个 AIV 留给 producer；而 Triton 侧既然
没有 AIV↔AIV 信号量、做不了 producer，就直接用 BV=32 换取同样的 32 block，不需要两轮。

因此：

* **不存在“BV=128 编译不过”的取舍问题**——在 H≤8、40 AIV 上 BV=128 只会把占用率打到
  10–20%。等做 decode（H=16/32/64、每 (req,head) 一个 block）时 BV=128 才是正解，
  那时才需要显式 UB（见 P4）。
* 如果标定发现有效 AIV 不是 40（比如 24 或 32），上表按同一公式重算即可，
  `pick_bv` 已经把它参数化为 `KDA_CORES`。

---

## 6. 验证与基准计划（由远程执行）

脚本：`target/bench_kda_tle.py`（自包含，不依赖 sglang）

```bash
cd test/CommonIR/agent_op/target
python3 bench_kda_tle.py calib
python3 bench_kda_tle.py check  --out v1_check.json
python3 bench_kda_tle.py sweep  --out v1_sweep.json
```

* `calib`：固定每 block 工作量、扫 grid=1..64，延迟第一次跃升前的最后一个 grid 即有效
  AIV 数（预期 40）。
* `check`：与纯 PyTorch 串行参考实现对比（T∈{64,128,256}、H∈{4,8}），给 max|Δ|、cos 与
  rtol/atol 判定。
* `sweep`：T∈{1024,2048,4096,8192,16384} × H∈{4,8}，对
  `{v0_autotune, v1(默认/BV16/32/64/128/prefetch/stages/multibuffer/flat)}` 取中位数。

需要回传：`calib` 的 suggested AIV 数、`sweep` 的 JSON、`check` 的 JSON，以及任何编译/运行
报错原文。

判定标准（先写死，避免事后解释）：

* `check` 必须 PASS（rtol=3e-2，atol=2e-3，bf16 输出）；
* `sweep` 里 v1 至少要在 (T=8192,H=4) 与 (T=8192,H=8) 上不慢于 v0，否则回滚对应选项；
* 与 AscendC 的对比用现有的 `bench_glm_kda_varlen_recurrent_npu.py` 那一套
  （需要 sglang），在 8 卡机上可以正常跑；对照口径取 `prepared` 列。

---

## 7. 本轮交付的文件

| 文件 | 作用 |
|---|---|
| `test/CommonIR/agent_op/KDA_ASCENDC_VS_TLE.md` | 本文档：对比、能力缺口、路线图、验证计划 |
| `test/CommonIR/agent_op/target/glm_kda_varlen_recurrent_npu_v1.py` | v1 kernel：core-aware BV、可选预取/扁平 state/`num_stages`，归一化倒数折叠；复用 v0 的校验与回退 |
| `test/CommonIR/agent_op/target/bench_kda_tle.py` | 自包含标定 / 扫点 / 正确性脚本 |
| `test/CommonIR/agent_op/target/test_kda_tle.py` | pytest 形式的正确性与状态语义检查 |

说明：开发环境没有 Python 也没有 NPU，上述代码**未经过任何执行验证**。第一次远程运行
同时充当编译检查，出现的 API 级报错（例如 `multibuffer=` 关键字、`tl.reshape` 形状列表）
请把 stderr 原文回传，一次修掉。
