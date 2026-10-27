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

> **第一次实测后本节已重排（见 §6）。** P1（DMA 双缓冲 / 软件流水）被证伪：预取、
> `num_stages`、`multibuffer` 全部 ±1%。真正的开销是**每 token 一条 ~1400 周期的
> 串行依赖链**（与 BV 无关），所以现在的大头是 P1.5（批量 prep），而不是 DMA。
> 下面的 P0/P1/P2/P3/P4 保留原编号以便和上一轮对话对应，新增项插入为 P1.5。

### P0 标定（先做，否则后面全是猜）

1. `bench_kda_tle.py calib`：固定每 block 工作量、扫 grid=1..64，延迟第一次跃升前的最后一个
   grid 即有效 AIV 数。预期是 40；若明显不是 40（分核/虚拟化/共卡），所有 BV 决策都要重算。
2. `bench_kda_tle.py sweep`：把 v0 的 `BV × num_stages × multibuffer` 扫一遍，确认：
   * autotune 实际选了哪个 BV；
   * `num_stages>1` / `multibuffer=True` 是否真的把 token 循环的 DMA 叠起来；
   * v0 是否已经等价于 v1 的默认配置（若如此，收益只剩 P1/P3）。

### P1 每 token 的 DMA 双缓冲 —— 实测证伪，不要再做

原假设：v0 在循环体里 `tl.load` q/k/a/v，每个 token 要等一次 MTE2 延迟，
应该像 AscendC 那样“先把下一个 token 预取到 `buf[nxt]`”。

实测（T=16384）：`PREFETCH=1` H4 17.57 → 17.84 ms、H8 23.38 → 23.51 ms（**更慢**）；
`num_stages=3` H4 持平、H8 −1.0%；`multibuffer=1` 无变化。三条路都没效果，说明 DMA
要么已经被编译器重叠，要么根本不是关键路径。`PREFETCH` 选项保留（默认关）以免以后再试一次。

### P1.6 规约调度 —— 实测失败（§6.6），但借它找到了真正的成本

baseline 的顺序让两次 `tl.sum` 串行：`v ← sum(state·k)` → `state += k vᵀ` →
`out ← sum(state·q)`，所以第二个规约必须等外积。我用恒等式 `out = sum(sd·q) + (k·q)·v`
把两者变成独立（`KDA_SCHED=overlap`），并试了 `tl.join → [BV,K,2] → sum → tl.split`
的多结果规约（`fused`）。

结果：**`overlap` 慢 13–16%，`fused` 编译不过。** 因为摘依赖的代价是多一次 `k·q` 规约，
而跨 lane 规约是吞吐受限、不能重叠的。结论反过来更有用：**一个 token 里的 4 次跨 lane
规约就是那 ~1400 周期的主要构成，能动的只有“每 token 几次”这个数字。**
代码保留（默认仍是 `base`）作为这条路的记录，不要再试它的变体。

### P1.5 批量 prep —— 已被 §6.6 降级：不做

（保留作为记录。）它只能批 token 本地的两个规约（`s_q`/`s_k`，~17%）与 gate（~26%），
上限 ~1.3×，且需要在 TLE 里重写内核；chunked 会把同样的工作连同 state 规约一起吃掉，
所以除非 `probe_dot.py` 证明 cube 不可用，否则不做这一步。

§6 的拟合把每 token 拆成 `0.79 µs 固定 + 0.018 µs × BV`；BV=16 时固定部分占 **74%**，
而其中真正的向量吞吐只有几十条指令——也就是说这 0.79 µs 几乎全是**依赖链延迟**
（两次跨 lane 规约 + sqrt + gate 的 exp/div 链 + beta 链），与 tile 宽度无关。
注意：按 §6.5 的消融，这里能用批量解决的只是 gate + norms（~43%），两次 state 规约
已经由 P1.6 处理（失败），两者的收益不可相加：它们共享同一条链。

对策：把一个 chunk（BT=8/16）的 prep **按 token 批量算一次**，让 BT 倍的独立工作
填满流水的深度，把“每 token 一次长链”变成“每 chunk 一次长链 + BT 次短的 state 更新”。
这与 AscendC 用 producer 把 prep 挪出关键路径是同一个目的，但**不需要跨核信号量**。

实现障碍（正是 TLE 的价值所在）：批量算出的是 `[BT, D]`，而串行 state 更新每次只需要
其中的第 r 行。纯 Triton 要么让 BT 成为最后一维再做递归 `tl.split`（等价于一次转置，
Ascend 上很可能更贵），要么用 mask+reduce 取行（把省下的延迟又还回去）。
TLE 的 `tle.dsa.alloc(UB)` + `tle.dsa.copy` + `tle.dsa.extract_slice` 正好可以
“批量算 → 落到 UB → 循环里按行取”，`causal_conv1d_fn.py` 已经用了同一套写法。
**这是“TLE 相对纯 Triton”的第一个可量化能力优势，不是纸面差异。**

预期：把 74% 里的大部分延迟摊掉，1.5–2.5×（上限是把固定部分降到 0，H4/T16384
17.6 → 4.6 ms；实际受 state 更新本身的依赖限制）。

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

## 6. 第一次实测结果（单卡 910B / 40 AIV，CANN 9.1.0，triton 3.5.x）

`bench_kda_tle.py sweep`，中位数（ms），warmup=5 / reps=20，单序列、2 个 state slot：

| case | v0 | v1 | v1_pf | v1_st3 | v1_flat | BV16 | BV32 | BV64 | BV128 |
|---|---|---|---|---|---|---|---|---|---|
| H4 T1k | 1.439 | 1.400 | 1.413 | 1.400 | **1.381** | 1.396 | 1.714 | 2.293 | ERR |
| H4 T2k | 2.543 | 2.455 | 2.486 | 2.462 | **2.411** | 2.448 | 3.078 | 4.212 | ERR |
| H4 T4k | 4.793 | 4.597 | 4.651 | 4.591 | **4.499** | 4.583 | 5.842 | 8.105 | ERR |
| H4 T8k | 9.224 | 8.856 | 8.963 | 8.842 | **8.660** | 8.838 | 11.359 | 15.903 | ERR |
| H4 T16k | 18.361 | 17.571 | 17.839 | 17.599 | **17.252** | 17.597 | 22.604 | 31.500 | ERR |
| H8 T1k | 1.800 | 1.733 | 1.725 | 1.713 | 1.760 | 2.464 | **1.713** | 2.280 | ERR |
| H8 T2k | 3.252 | 3.101 | 3.115 | 3.096 | 3.181 | 4.592 | 3.095 | 4.225 | ERR |
| H8 T4k | 6.171 | 5.871 | 5.896 | 5.854 | 5.998 | 8.844 | 5.849 | 8.106 | ERR |
| H8 T8k | 12.138 | 11.535 | 11.621 | 11.495 | 11.783 | 17.617 | 11.464 | 15.925 | ERR |
| H8 T16k | 24.718 | 23.378 | 23.514 | **23.150** | 23.735 | 36.827 | 23.155 | 31.493 | ERR |

（`v0` 为 autotune；`v1` 即 `KDA_BV=auto`；`v1_st2`、`v1_mb` 与 `v1` 差 <0.3%，未列。）

### 6.1 v1 只赢了 4–6%

T=16384：H4 18.36 → 17.57（auto，**−4.3%**）→ 17.25（flat，**−6.0%**）；
H8 24.72 → 23.38（auto，**−5.4%**）→ 23.15（st3，**−6.3%**）。
收益主要来自归一化倒数折叠（每 token 少两次 128 通道除法）和 BV 选择，不来自任何
结构改动。**v0 的 kernel 已经接近它这个形状的局部最优。**

### 6.2 BV / wave 模型方向对、幅度错

BV=128 全部 `ERR`（UB 溢出复现）；BV=32/64 比 BV=16 只慢 28%/79%，不是模型预测的 2×/4×。
原因：单 block 时间不是 ∝ BV，而是**仿射**：

```
per_token(µs, H4) = 0.79 (固定) + 0.0177 × BV      # 拟合 R²≈0.999，T≥4096 稳定
per_token(µs, H8) = 0.90 (固定) + 0.0170 × BV
```

| H | T | per-token µs | 固定部分 | 固定占比 |
|---|---|---|---|---|
| 4 | 1024 | 1.363 | 1.071 | 78.6% |
| 4 | 16384 | 1.074 | 0.791 | 73.7% |
| 8 | 1024 | 1.673 | 1.119 | 66.9% |
| 8 | 16384 | 1.413 | 0.904 | 64.0% |

固定部分 ≈ 0.3 ms 的 launch/调度开销 + 0.77 µs/token（T 越大占比越高；T=1024 时那
0.3 ms 约占 22%，含 Python/dispatch，不计在内核对 kernel 无意义）。

**关键推论：BV=16 时 74% 的时间是与 tile 宽度无关的每 token 串行链。** 而这一链的实际
向量吞吐只有几十条 128-lane 指令（完美流水下 T=16384 应 ≈0.4 ms），所以它是
**纯延迟受限**，不是吞吐受限。state 那部分反而是健康的：`0.0177 µs/BV` ≈ 每条 128-lane
指令 3 个周期，已经接近硬件吞吐极限。

这同时修正了 §3 的错误估计：prep 的**吞吐**占比确实只有 ~5%，但它在**串行关键路径**上，
按 wall-clock 占约 74%。AscendC 的 producer 买到的不是“省掉 prep 的算力”，而是“把这段
延迟挪出关键路径”——所以它的 19% 是保守的。

### 6.3 预取 / 软件流水全部无效（P1 被证伪）

`v1_pf` 比 `v1` **慢** 1%，`v1_st3` 在 H8 上只快 1%（H4 持平），`v1_mb` 无变化。
说明 DMA 不是关键路径（或编译器已重叠）。不要再在这条路上投入。

### 6.4 `FLAT_STATE` 不对称：H4 +2%、H8 −1.5%

H4（BV=16，8 KB tile）稳定快 2–4%，H8（BV=32，16 KB tile）稳定慢 1.5%。
推测是 1-D 无 mask 搬在更大 tile 上丢了按 datablock 切分，但收益太小不值得为它分叉，
保持默认关、按形状手动开。

### 6.5 归因结果（第二次实测）与修订后的下一步

T=8192，`KDA_ABLATE` 探针（故意算错数，只用于定位那 1400 周期），相对 control 的降幅：

| 关掉 | H4 (base 8.856 ms) | 占比 | H8 (base 11.473 ms) | 占比 |
|---|---|---|---|---|
| `norms`（q/k 两次规约 + sqrt） | −1.510 ms | 17.0% | −1.080 ms | 9.4% |
| `gate`（a+bias / 2×exp / div / beta） | −2.339 ms | 26.4% | −2.884 ms | 25.1% |
| `vred`（`sum(state·k)`） | −1.893 ms | 21.4% | −2.778 ms | 24.2% |
| `outred`（`sum(state·q)`） | −2.476 ms | **28.0%** | −3.094 ms | **27.0%** |
| `load`（地址不随 token 变） | −0.010 ms | **0.1%** | −0.109 ms | 0.9% |
| `all`（只留 state 外积） | 剩 2.618 ms | −70.4% | 剩 3.112 ms | −72.9% |

四条结论：

1. **加载彻底洗清**（0.1%/0.9%）。P1 到此结案，文档里那条假设可以划掉。
2. **最大头不是 prep，是两次 state 规约**（`outred` 28% + `vred` 21% ≈ 49%），
   `norms`+`gate` ≈ 43%。四项单删加起来 93% > 一起删的 70%，说明它们共享同一条链。
   所以“批量 prep 就能救场”是错的：它最多覆盖 43%。
3. 这两次规约在 baseline 里是**串行**的（`out` 必须等外积更新），彼此无法重叠；而被删掉
   一个就省 20%+，说明它们是**延迟受限**而不是吞吐受限（BV=16 只有 16 行，填不满流水）。
4. `abl_all` = 2.618 ms 与仿射模型的预测 `0.0177×16×8192 + 0.3 = 2.62 ms` 吻合到 0.1%，
   两套独立方法互证，分解可信。（注意 `SKIP_GATE` 让 decay 变成常量 1，`state*decay`
   会被常量折叠，所以 2.62 是**下界**。）

因此最优顺序变成本文 §5 的 **P1.6（规约调度，纯 Triton）→ P1.5（TLE 批量 prep）→ P4（chunked）**，
而不是直接上 TLE。天花板是 `abl_all` 给出的 **3.4×**（8.86 → 2.62）；所有微优化都撞这个顶，
再往前只能靠 token 轴并行。

### 6.6 P1.6 实测：失败，而且给出了比“成功”更有用的结论

`KDA_SCHED=overlap` 实测 **更慢**，`fused` 直接编译不过（`tl.join`/`tl.split` 这条
多结果规约路径在这套 BiSheng 流水线里走不通）：

| case | base | overlap | Δ |
|---|---|---|---|
| H4 T8192 | 8.859 ms | 10.020 ms | **+13.1%** |
| H8 T8192 | 11.522 ms | 13.351 ms | **+15.9%** |
| fused | — | `MLIRCompilationError` | 不可用 |

原因不是“连接依赖”假设错了，而是**我对成本的归因错了**：

* `overlap` 为了摘掉第二个规约对外积的依赖，必须多算一个 `k·q` 点积——**每 token 多了一次
  跨 lane 规约**（在第 3 个规约）。四个规约变五个，代价 +13%，和实测完全吻合。
* 也就是说：跨 lane 规约是**吞吐/发射受限**，不是“延迟可重叠”。从消融反推，一次
  `[BV,K]` 规约在 T8192/H4 上大约值 2.2 ms（≈ 480 cycles/token/次）。
* **一个 token 里有 4 次跨 lane 规约**（`s_q`、`s_k`、`vred`、`outred`），这就组成了那
  个 ~1400 周期的固定预算的大部分。调 BV、预取、重排、合并——都改不了“每 token 四次”
  这个数字。

**只有一个数字值得改：每 token 的规约次数。** 只有两类办法能改它：

1. **按 token 批量**（把 4 次/token 变成 4 次/BT token）—— 但只能批 token 本地的两个
   （`s_q`/`s_k`），state 那两个是串行依赖，批不了；而且 `fused` 已经证明这套后端
   发不出多结果规约，所以需要 TLE 的 UB + `extract_slice` 才能把批量结果按行取回。
2. **把规约换成矩阵乘**（chunked delta rule）：`Σ_j` 变成 cube 上的 `tl.dot`，吞吐受限
   而非延迟受限，而且一次处理 BT 个 token。

所以路线图再改一次：

* **放弃 P1.5（TLE 批量 prep）。** 它只能动 4 个规约里的 2 个（~17%）加上 gate（26%），
  上限 1.3× 左右，而为了这些要在 TLE 里重写内核；chunked 把同样的工作连同 state 规约
  一起吃掉，性价比不在一个量级。
* **直接去 P4（chunked delta rule）**，但先花一次极小的实验确认前提：cube/`tl.dot` 在这套
  流水线上能不能用、能不能 bf16 输入配 fp32 累加器、小 tile（64³/128³）的吞吐够不够。
  这就是 `target/probe_dot.py`——**如果那里不成立，chunked 整条路线就不成立**，
  那时候才回头去做 P1.5 的 1.3×。

```bash
cd test/CommonIR/agent_op/target
python3 probe_dot.py                           # 全部 dtype × 64/128/256
python3 probe_dot.py --shapes 128 --dtypes bf16  # 先只要一个答案，几秒
```

探针本身踩过两个坑，记在这里以免再犯：

* **tile 大于矩阵 = 零 block launch = 硬崩。** `block=128` 对 64³ 会得到 grid=(0,)，
  在这套栈上直接报 aicore inner error。所以现在 tile 会被 clamp、且 grid>0 有断言。
  推论：将来 chunked 内核必须在 host 侧校验 `shape % tile == 0`，不能指望 kernel 容错。
* **一个 case 崩了会毒死整个进程。** 崩之后所有 tensor 回来都是 0，`rel.err` 就变成 0/0，
  于是后面八行全是 `ZeroDivisionError`——看起来像“全都不支持”，其实一个字都没测到。
  所以现在**每个 case 一个子进程**，一次崩只损失一行。
* **子进程不能走 `python -c`。** `@triton.jit` 要 `inspect.getsourcelines()`，而 `-c` 的
  `__file__` 是 `"<string>"`，直接报 "@jit functions should be defined in a Python file"。
  现在子进程是**重新调用本文件**。

#### 第一轮结果（bf16/fp16/fp32 × 64³/128³/256³）

九行全部 `ok`：

| case | ms | rel.err | per_dot |
|---|---|---|---|
| bf16→fp32 64³ | 0.1472 | 1.23e-07 | 0.147 µs |
| bf16→fp32 128³ | 0.1369 | 2.35e-07 | 0.137 µs |
| bf16→fp32 256³ | 0.1466 | 2.74e-07 | 0.147 µs |
| fp32→fp32 64/128/256³ | ~0.14 | **0.00e+00** | — |

两条结论：

1. **cube 门开了。** bf16 输入 + fp32 累加器可用且正确：`rel.err ≈ 2.4e-7`，恰好是
   “对 bf16 舍入后的输入做 fp32 累加”的舍入水平（参考值用的是 `a.float() @ b.float()`，
   即 bf16 值的精确乘积）。fp32 输入也能跑，且与 torch 完全一致。
   所以 chunked 的算力与精度前提都成立，三角求解既可留 fp32 也可 bf16+fp32。
2. **但这组数字量的是主机下限，不是矩阵乘。** 三个尺寸全是 ~0.14 ms，TFLOP/s 只有
   0.004/0.031/0.229 —— 0.14 ms 是 python + dispatch + launch 的地板（与 KDA 里拟合出的
   ~0.3 ms 启动开销同一个量级）。所以 256³ 的**内核本身 ≤ 6.6 µs → ≥ 5.1 TFLOP/s**，
   这只是个下界。

要拿到真正需要的两个数（chunked 该选多大的 C），得用 `--repeat`：

```bash
python3 probe_dot.py --shapes 64,128 --dtypes bf16 --repeat 256   # 链式 dot 的每次延迟
python3 probe_dot.py --shapes 1024,2048 --dtypes bf16             # 大 tile 的吞吐上限
```

`--repeat` 让同一个累加器被连续喂入 REPEAT 次（依赖链，不能流水），主机开销被摊薄 REPEAT 倍，
得到的 `per_dot_us` 才是“一个 chunk 里每做一次小矩阵乘要付多少”，可以直接和
“每 token 0.8 µs 固定 + 0.018×BV”那个预算对比。

顺带一个对 KDA 本身有用的事实：**单次 launch 的主机地板就是 ~0.14 ms**。这解释了为什么
T=1024 的 KDA 耗时（1.38 ms）里有一大块与计算无关，也意味着任何“拆成多次 launch”的
方案（包括 AscendC 的 prepared 两段式）都要先付这么一笔。

---

### 6.7 chunked 路线的实测经济账：每个 op 的成本决定一切

chunked 内核跑通且数值正确（§6.6），但**比它要替换的串行递推慢 2.4×**：

| 变体 | ns/token | |
|---|---|---|
| chunked C=16 BV=32 T=8192 | 1488 | |
| chunked BV=16 | 2671 | relgram 随 V-tile 重复 4→8 份 |
| chunked BV=64 | — | UB 溢出 1,671,168 vs 1,572,864 bits |
| 串行·完整 | 1082 | |
| 串行·prep-ablated | **612** | 同口径 |

归因探针（`kda_chunked.py --bench`，故意算错数）：

| 关掉 | ns/token | 省 | 推出单位成本 |
|---|---|---|---|
| solve（16 步列扫描） | 695 | 46.7% | **695 ns/步 = 1251 cycles/步** |
| dots（6 个） | 340 | 22.8% | **906 ns/每个 dot = 1631 cycles** |
| cumsum | 202 | 13.6% | 16 步串行扫描 |
| relgram（2 dots + mask） | 179 | 12.0% | |
| 全部（floor） | 96 | 93.6% | 循环 + 加载 + store |

四项单删相加 95.1% ≈ 一起删的 93.6% —— **几乎完全可加，即四块都是纯成本、互不重叠**。

两条单位成本是本项目最有复用价值的数字：

* **一次小矩阵乘（含操作数 staging）= 906 ns ≈ 1631 cycles**。cube 探针里链式 dot 只要 122 ns，
  差别就是每个 chunk 重新 stage 操作数（UB→L1 ND2NZ）+ 结果搬回。这是“小 matmul 在 Ascend 上
  全是 overhead”的定量版本。
* **一步跨 lane 规约 ≈ 1250 cycles**（与串行内核里反推的 480 cycles 同量级，这里是两次规约/步）。

### 结论：在 C=16 下 chunked 的上限只有 1.34×

把每一块都用 matmul 重写（求解换 doubling 的 7 个 dot、cumsum 换 `tril_ones @ g` 一个 dot、
staging 换 bf16、relgram 同步减半）：

```
floor 96 + solve 397 + cumsum 57 + dots 170 + relgram 89 = 808 ns/token
vs 串行·完整 1082 → 1.34x ；vs 串行·递推 612 → 0.76x（仍然更慢）
```

也就是说**在 C=16 这个尺寸上，chunked 无论怎么优化都追不上串行版**，因为：

1. 每个 chunk 至少需要 ~6 个 dot，而一个 dot 就是 906 ns；
2. 求解是 C 步/C 个 token = **每 token 一步**，不随 C 摊销；
3. C 被 `exp(-gc) ≤ exp(5C)` 钉在 16（§6.6）。

### 但 C 由 gate 决定，所以结论是有条件的

同样的 kernel 在 **C=64** 下：dot 成本摊薄 4×（340 → 85）、cumsum 摊薄、求解用 doubling 变成
log C 个 dot（0.156 dot/token = 140 ns），预算变成约 **420 ns/token → 2.6×**。

而 C 能否取 64 完全取决于线上 gate 的衰减速度：`exp(-gc) = exp(Σ|g|)` 超过 fp32 上限就不可用。
bench 的合成输入（decay≈0.082/token）把 C 锁在 17；若线上多数通道 decay≈0.99（长记忆），
C=64/128 完全安全。

**所以 chunked 路线的去留取决于一个数：线上 `exp(gate)` 的分位数。** 在那之前：

* 不做 chunked 的进一步优化（C=16 的上限 1.34× 不值得）；
* 先拿**无条件收益**：串行版的 prep 占 470 ns/token（1082−612）且是 token 局部的，
  批量化后可降到 ~100–150 → 总计 **~712–762 ns/token = 1.45×**，风险低、复用现有代码。

### 6.8 结论：chunked 路线关闭（C=32 测试失败，未定位）

C=32 的测试失败，chunked 路线到此终止。**需要如实记录的是：这次失败不能当作对算法的证据。**
那条路径上三个部件（doubling 求解、`tril_ones @ g` 的 cumsum、`--gate-offset`）都是
当轮新写的，在设备上只跑过 C=16 + `--solve sweep` 这一种组合（它通过，误差 1e-7）。
失败既可能是我的 bug（doubling 在 C=32 的 5 层展开、`tril_ones` 点积在 C=32 的
`[32,32]@[32,128]`、或者 UB 估算偏乐观），也可能是算法本身，本轮没有区分。

所以状态是：

| 配置 | 状态 |
|---|---|
| C=16, `--solve sweep`, fp32 | 数值正确（1e-7），1488 ns/token = 0.7× 串行完整 → 不划算 |
| C=32, `--solve double` + matmul-cumsum | **失败，未定位** |
| C=64, `--bf16` | 未运行（依赖 C=32 的结论） |

**重启条件**（写给未来的自己）：只有在同时满足两条时才值得再看 ——
(a) 有人拿到线上 `exp(gate)` 的分位数量级，(b) 有人愿意先把 C=16 + doubling 跑通
（那是能区分“我写错了”和“算法不行”的最小实验）。按 §6.7 的经济账，即便两条都满足，
C=16 也只有 1.34×，所以真正的前提其实是 (a) 里的答案是“存在长记忆通道”。

文件保留在 `target/kda_chunked.py`（算法与全部实测数字的来源）与 `target/chunked_ref.py`
（fp64 参考实现，推导本身已验证到 1e-15，这部分结论不受影响）。

### 6.9 两段式（批量 prep + 串行递推）也关闭：1.07×

数值正确（`out rel 2.6e-3` 是 bf16 输出对 fp32 参考，`state rel 1.3e-7`），但收益不够：

| 分段 | ms | ns/token |
|---|---|---|
| prep 一段 | 2.981 | 364 |
| 递推一段 | 5.336 | 651 |
| 合计 | 8.317 | 1015 |
| 一起跑（实测） | 8.324 | 1016 |
| 串行 fused（v1） | 8.856 | 1082 |

**两次 launch 完全不重叠**（合计 ≈ 一起），所以收益只能来自两段各自变快，而它没有：

* prep 从 fused 里的 470 ns/token 降到 364，**只降了 23%**，不是我预测的 ~80%；
* 递推从 5.010 涨到 5.336（+6%），因为两段式喂给它的是 **fp32** 的 `qn/kn/decay`：
  每个 V-tile 都要重复读这三行（BV=16 时每 head 8 次），字节数从 bf16 的 768B/token
  变成 fp32 的 1536B/token；再外加 50 MB 的 scratch 写回。

**为什么批量化只能降 23%：我把“规约是吞吐受限”这件事又忘了一次。** 一次跨越 C 行的
`tl.sum(axis=1)` 的 repeat 数是 C，不是 1，所以批量只摊薄"每 op 固定开销"，从不减少
每 token 的 repeat 数量。实测的每 op 固定开销反而是个大数：prep 一段 58 µs/block，
C=16 时约 **6550 cycles/token-head ÷ ~10 个 op ≈ 650 cycles/op**。同一个错误在 P1.6
（`KDA_SCHED=overlap`）那轮犯过并写在 §6.5 里，规划 prep 批量时没回看。

两个方向都关闭后的实际可交付状态：

| 版本 | T=8192/H4 | 相对 v0 | 状态 |
|---|---|---|---|
| v0（autotune） | 9.224 | — | 基线 |
| **v1（BV 选择 + 归一化折叠 + 可选 flat）** | **8.660–8.856** | **1.04–1.065×** | **建议交付** |
| 两段式 prepared | 8.324 | 1.11× | 不交付：多一次 launch + 50 MB scratch + 代码重复，换 6% |
| chunked C=16 | 12.193 | 0.76× | 归档 |

### 6.10 这个内核在这套栈上的成本模型（本项目的最终结论）

三次独立测量给出同一张表，它解释了为什么所有结构性重写都只得到个位数收益：

| 操作 | 实测单位成本 | 出处 |
|---|---|---|
| 一次跨 lane 规约 | ~480–650 cycles | §6.5 与 §6.9 两次反推 |
| 一次小 `tl.dot`（含 staging） | ~906 ns（1631 cycles） | §6.7 |
| 一次 prep 向量 op | ~650 cycles | §6.9 |
| 一次 launch | 140 µs | probe_dot |
| state 级算术（per BV 行） | ~3 cycles/128-lane repeat | §6.6 拟合 |

推论：**在这套 BiSheng/triton 后端上，性能由“op 个数”而不由“FLOPs”决定**，而 KDA 的
串行递推每 token 必然包含 4 次规约（2 个 norm + 2 个 state），这就是 1082 ns/token 的来源。
任何不改 per-token op 数的重排、批量、chunk 化（chunk 化的 dot 数量又反而上升）都只能
动个位数百分比。要真正突破，需要的是减少 op 数的原语（多结果规约、把 norm 折进 matmul），
那是编译器/后端层面的能力，不是内核层面能解决的。

### 6.11 多结果规约原语：做出来了、验证了，但无法盈利地接入

按上一节的推论，我给 TLE 加了一个 custom op `fold_reduce_rows`（折叠两半 + 一次 `vcadd`，
两个 dst），并逐步验证到每一步都成立：

| 验证项 | 结果 |
|---|---|
| `tl.join`/`tl.split` 多结果规约 | 编不过（§6.6） |
| 单 dst 版（`[repeat,2*mask]` → 1 个结果向量） | **rel = 0.00e+00** |
| 双 dst 版（两结果直接分开返回） | **rel = 0.00e+00**（rows 1/2/16） |
| 计算出来的 tensor 能否直接喂给 `raw` | **能**（不需要 `insert_slice`/buffer 构造） |
| `insert_slice` 拼接两个 tile | **设备侧 UB 越界**，不可用 |
| 用 `row % BV` + `where` 复制构造 KDA 需要的形状 | **PASS，rel = 0** |

到这里原语本身无懈可击。但把它接进 kernel 需要的**tile 构造本身就是 op**，而每个 op 在这个
后端要付 ~650 cycles（§6.10），所以拿 op 个数一算就否了：

| | 基座 v1 | 融合版 | 结论 |
|---|---|---|---|
| norms（`q*q`/`k*k` + 2 次规约） | 4 op | `q*q`+`k*k`+`where`+1 raw = **4 op** | 持平 |
| state（`state*k`+规约、`state*q`+规约） | 5 op | 2 broadcast + `where` + `reshape` + mul + 1 raw = **6 op** | **更差** |
| overlap 恒等式还要一个 `k·q` | — | +2 op（又一次规约） | 更差 |

那次 "1.35×" 的前提是 `[2,BV,K]` 这种拼装**免费**。它不是：`where`／`broadcast`／`reshape`
恰好就是成本模型里计的那 650 cycles。所以接入后净收益 ≈ 0，state 那一对甚至是负的。

> **上面这段有一个错误，已修正。** 我当时断言"拼 tile 要 where/broadcast/reshape 共 3–5 个 op"，
> 但 `tle.dsa` 里本来就有 `tile_concat(lhs, rhs, dim)`，它 lower 成 **`hivm.hir.vconcat`，
> 一个 op**；拆分侧也有 `extract_slice`（在 `causal_conv1d_fn.py:123` 已经被用于在循环里
> 从 UB tensor 取第 j 行，所以它本来就是可用的）。我没查这两个接口就下了结论，这是错的。

按真实成本重算（op 数，单位是 §6.10 模型里的 triton 级 op）：

| | 基座 v1 | 融合版（用 `tile_concat` + `extract_slice`） |
|---|---|---|
| norms | `q*q`+sum、`k*k`+sum = **4** | `q*q`、`k*k`、`concat`、1 raw = **4** → 持平 |
| state | `state*k`+sum+sub、`state*q`+sum = **5** | `state*k`、`state*q`、`concat`、raw、2×`extract_slice` = **5**（若 slice 是零成本视图则 4） |

**一次规约 ≈ 一次 concat**，所以融合在 op 数上是**持平**：既不是我之前说的 26% 收益，
也不是后来担心的变负。剩下的问题降级成"每条指令的单价"：一次 `vconcat` + 一次 `vcadd`
是否比 `tl.sum` 内部展开更便宜 —— 这只能在真实 kernel 里测，而它的上限是 **~13%**
（9 个 op 里的 1 个）。

顺带一个要更正的旧结论：§6.6/§6.9 里我用"Triton 取不出第 r 行"来论证 chunked 的批量
prep 必须依赖 UB 机制。**那句也错了** —— `extract_slice` 能做这件事。chunked 路线终止的
真实原因仍然是 C ≤ 17 与每 op 单价（§6.7 的经济账），与"取行能力"无关。

### 6.12 融合规约实测：+6.8%，自定义 op 比原生 `tl.sum` 更贵（收尾）

用 `tile_concat` + `fold_reduce_rows`（双 dst）把 norms 那一对合起来，实测反而更慢：

| | 基座 v1 | `KDA_FUSE_RED=1` | Δ |
|---|---|---|---|
| H4 T8192 | 8850.1 µs | 9449.4 µs | **+6.8%**（+73 ns/token） |
| H8 T8192 | 11457.5 µs | 12342.6 µs | **+7.7%**（+108 ns/token） |

norms 那一对占 1082 ns/token 的 17%（≈184 ns），融合后约 257 ns，**+40%**。

也就是说：**一次 `tile_concat` + 一次 `raw`（vadd + 2×vcadd）比它替换掉的那两次原生
`tl.sum` 更贵。** 贵不在算法（op 数持平、硬件指令几乎相同），而在这个自定义 op 调用本身
—— memref 构造、操作数 staging、同步。

这条把 "用 AscendC 原语补差距" 这条路彻底关掉了，而且和前面三次失败同源：

| 尝试 | 结果 | 真实原因 |
|---|---|---|
| P1.6 overlap 恒等式 | +13% | 多一次规约（吞吐受限，不可重叠） |
| 两段式批量 prep | +1.07× | 规约是吞吐受限，批量只摊薄延迟 |
| chunked C=16 | 0.76× | 每个小 dot ~906 ns staging，C 被 gate 钉在 16 |
| **融合规约（本行）** | **+6.8%** | **自定义 op 的固定开销 > 原生 `tl.sum`** |

结论（本项目的最终答复）：

* **"补更多 AscendC 原语" 不能补上这 20%**：每次跨语言调用要付 ~70–110 ns 的固定开销，
  而我们的算术只能省下 1–2 个原生 op（每个 ~92 ns）。要赢，一次调用必须替换掉
  **好几个** op，而 KDA 每 token 的 op 结构给不出这种机会。
* 真正的杠杆只剩**编译器自己**：把 per-token 的 op 单价降下来（~650 cycles 的固定开销），
  或者让编译器把 norm/规约序列直接生成成 AscendC 那样的紧凑序列。这是编译器组的活。
* 交付仍是 **v1**（比 v0 快 4–6.5%，数值已验证）。

产物保留：`fold_reduce_rows`（C++/registry/构建接线，双 dst，数值精确 0.00e+00）、
`probe_fold_reduce.py`、`probe_concat.py`、`chunked_ref.py`、以及 §6.5–6.12 的完整证据链。

---

## 7. 验证与基准计划（由远程执行）

脚本：`target/bench_kda_tle.py`（自包含，不依赖 sglang）

**磁盘**：脚本自身只写 `--out` 指定的一个 JSON。占盘的是 Triton 编译缓存
（`$TRITON_CACHE_DIR`，每个不同 kernel 一个目录）；`--configs default|ablate|all|<名字>`
控制编译次数，`GLM53_KDA_AUTOTUNE=0` 可把 v0 的 24 次 autotune 编译降到 1 次。

```bash
cd test/CommonIR/agent_op/target
python3 bench_kda_tle.py calib                                        # 期望 ~40 AIV
GLM53_KDA_AUTOTUNE=0 python3 bench_kda_tle.py check --out c.json      # 容差判定
python3 bench_kda_tle.py sweep --configs ablate --tokens 8192 --out a.json
```

判定标准（先写死，避免事后解释）：

* `check` 必须 PASS（rtol=3e-2，atol=2e-3，bf16 输出）；`v1` 不得比 `v0` 离参考更远；
* 没有实测证据就不加配置：`sweep` 里新配置必须至少在一半的形状上快于当前最优，
  否则回滚（`v1_pf` 就是因为这条被关掉的）；
* 本轮**不**把 AscendC 的 `prepared` 列作为门槛（需要 sglang，且是两套 tiling，
  混进同一张表容易得出错误结论）；AscendC 只作参考，见 §1/§2。

---

## 8. 本轮交付的文件

| 文件 | 作用 |
|---|---|
| `test/CommonIR/agent_op/KDA_ASCENDC_VS_TLE.md` | 本文档：对比、能力缺口、路线图、验证计划 |
| `test/CommonIR/agent_op/target/glm_kda_varlen_recurrent_npu_v1.py` | v1 kernel：core-aware BV、可选预取/扁平 state/`num_stages`，归一化倒数折叠；复用 v0 的校验与回退 |
| `test/CommonIR/agent_op/target/bench_kda_tle.py` | 自包含标定 / 扫点 / 正确性脚本 |
| `test/CommonIR/agent_op/target/test_kda_tle.py` | pytest 形式的正确性与状态语义检查 |
| `test/CommonIR/agent_op/target/probe_dot.py` | cube 前提探针：`tl.dot` 各 dtype、小 tile 吞吐 |

本轮追加：`KDA_ABLATE`（5 个计时探针，故意算错数）与 `--configs`（选列，控制编译次数），
以及 §6 的实测结论。v1 的默认行为与上一轮完全一致（所有 `SKIP_*` 为 False 时生成同一份代码）。

说明：开发环境没有 Python 也没有 NPU，上述代码**未经过任何执行验证**。第一次远程运行
同时充当编译检查，出现的 API 级报错（例如 `multibuffer=` 关键字、`tl.reshape` 形状列表）
请把 stderr 原文回传，一次修掉。
