# 极限 offload runtime：设计与实验记录

维护分支：`feat/runtime`。原始调研创建于 2026-09-05；本版本迁移到清理后的
framework root，并保留旧提交号作为历史证据。

本文件是用户要求的分支专用设计、开发顺序和实验记录。项目状态及通用约束由
项目级约束由主分支的公开接口和测试管理。当前 packed artifact 已可生成和逐 tensor
验证；
`src/runtime/` 仍只有客户端等既有能力，没有可运行的极限 offload engine。

## 1. 与泛用 runtime 的边界

| 项目 | 泛用 runtime / `main` | 本分支 |
|---|---|---|
| 设备目标 | 16GB 显存及更充裕系统内存 | 名义 16GB RAM + 6GB VRAM + SSD |
| 首要取舍 | 交互速度、较少 offload | 有界内存下完整运行，再优化速度 |
| 最终模型 | 选定的 merged Q3+OPD artifact | 使用同源模型，不另训或改变量化方案 |
| 上下文 | 128K 长上下文方向 | 128K 总长度的单请求容量目标 |
| FA KV | 由泛用路线独立验证其策略 | 本路线固定 BF16、保留完整历史 |
| SSD | 文件存储及可选容量补充 | 明确支持活跃权重/必要 KV 的分块流送 |
| 验收 | 泛用设备的质量与性能 | 最低配置的容量、质量和独立延迟记录 |

硬件标签的 GB 不是可用内存保证。测量采用 bytes/MiB/GiB，记录物理 RAM、
操作系统/WSL 限额、已有进程、显示占用及实际可用值。不能把大内存机器的
page cache 命中成绩当成 16GB RAM 结果。

开发代码仍放在 `src/runtime/`，维持逻辑 `latticerun.*` 包布局。模型形状、层分类、
控制参数策略由 `model/` 提供，不在通用 offloader 中硬编码 Qwen 名称。
不复制第二套 PTQ/OPD；不预建空模块。可复用的 packing/kernel 修复可按独立提交
验证后带回泛用路线，极限配置、SSD 默认值和成绩不自动带回 `main`。

## 2. 已锁定的约束

- 文本 dense Qwen3.8-27B，最终 adapter 已 merge，部署不保留 DoRA 动态分支。
- 真 INT3 G128 主权重、INT4 lm_head，沿用既定代码值和 clipping/scale 契约。
  将 packed artifact 准备视为可解决的输入条件；调度器可先用合成 payload 开发。
- FA KV 固定 BF16；不以 KV 量化、滑窗截断、稀疏 top-k 或漏算历史换取达标。
- GDN 当前 recurrent state 保留 FP32，conv/control/norm 保持既定精度。
  活跃 GDN 状态约 147MiB，应优先留 GPU，不当作长历史 KV 往磁盘流送。
- 最初 batch/concurrency=1，noMTP，无额外多前缀快照；128K 指 prompt + output，
  不是 128K prompt 之外另送未计费的输出空间。
- 完整 BF16 权重、完整 FP32 KV 副本、全部 offloaded 权重的 pinned 副本均不能
  成为加载期或稳态的隐含要求。读取/转换/计算 buffers 必须有容量上限。
- CPU/GPU 计算位置、prefill 次序、KV/weight 页大小和缓存比例是待实测选择；
  CPU attention 当前只是候选，不能记作已验证加速。

## 3. 内存分配基线

计数来自 [`qwen38-runtime-research.json`](../../manifests/qwen38-runtime-research.json)。
以下用 FP32 scales 估算 payload，未替代实际 allocator/driver 峰值测量。

| 内容 | 大小约值 | 初始放置 |
|---|---:|---|
| 16 层 FA 的 Q/K/V/O 权重 | 0.635GiB | GPU 优先 |
| lm_head | 0.629GiB | GPU 优先 |
| GDN 当前 recurrent + minimal conv state | 0.143GiB | GPU |
| protected 权重 | 0.049GiB | GPU；精确 dtype 由 loader 复核 |
| 全部 128K BF16 FA KV | 8GiB | GPU/RAM 分层，必要时 SSD |
| embedding | 0.481GiB | CPU 按行查表，文件可 mmap |
| 全部 FFN 权重 | 6.475GiB | RAM 缓存 + SSD 溢出 |
| GDN qkv/z/out 投影权重 | 2.095GiB | RAM 缓存 + SSD 溢出 |

GDN 投影权重不能与其小型递归状态混淆；FA decoder block 还包含 FFN，
不能将“FA 主投影常驻”误记为“整个 FA block 已常驻”。

一个 6GiB GPU 初始容量分配为：上述约 1.46GiB 固定内容 + 2.5GiB KV +
约 2GiB 的传输、activation/workspace、allocator/显示余量。2.5GiB KV 可对应
5 个 FA 层的完整 128K cache；其余 11 层约 5.5GiB，优先留 RAM。
具体哪些层由 planner 和测量决定，不能依赖请求执行时的偶然内存余量。

RAM 优先容纳 5.5GiB 活跃 KV、embedding、小型缓冲区及约 3--4GiB 权重缓存；
其余留给框架/OS。权重缓存须按真实可用 RAM 下调，不能把整机 16GB 全交给引擎。
剩余约 4.57--5.57GiB Linear 权重需要 SSD 流送。若 RAM 仍不足，允许部分活跃
BF16 KV 落盘，并将其新增读盘时间单独记录，不能静默依赖 OS swap。

SSD 保存所有不可变权重的源文件，但 GPU/RAM 命中的 payload 不再重复读盘。
GPU 常驻权重不要求一份额外的 CPU 常驻镜像；file-backed resident pages 仍计 RAM。

## 4. 三层存储与生命周期

建议的责任边界是 payload store、执行计划、hybrid state store。仅在实现需要时
创建对应代码，不为文档名提前建立抽象层。

### 权重

1. 按真实 quantized payload 注册 codes/scales、布局、shape/stride/dtype、文件位置
   和校验信息；registry 必须支持 buffers，不能只遍历 `named_parameters()`。
2. 显式区分 GPU 常驻、RAM 常驻/缓存、磁盘流送。避免循环扫描超额 mmap 工作集
   使操作系统反复淘汰；缓存命中、实际 SSD 读量和 page faults 都要能观察。
3. 固定数量的 pageable/pinned/GPU slots；记录 `read_ready`、`copy_done`、
   `compute_done` 所有权。CPU 源缓冲在 DMA 完成前不可覆写，GPU slot 在计算
   完成前不可复用。依靠队列背压限制预取，不是无限 future/线程。
4. 只读权重用完后复用 slot，无须 D2H 写回。repack 在离线/加载阶段完成，避免
   每 token 全层展开 BF16 或重新布局。热路径不依赖逐层 `empty_cache()`。
5. 短读、异常、取消必须传播；取消后等待在途 I/O/DMA、释放本请求资源，不能
   让下一请求消费半页数据或旧 slot 内容。

### KV 与 GDN 状态

- BF16 FA KV 以层/head/token 页寻址，记录有效 token 数及不可变旧页。
  每步全模型新增 KV 约 64KiB；只追加脏页/新 token，不重写整段历史。
- CPU 保有的 KV 可直接供 CPU attention，也可送 GPU；SSD 页都必须先读出才能
  精确参与计算。旧 token 不是可忽略的冷 token。
- 同一 head 若按 token 区间拆给 CPU/GPU，分别返回 softmax 统计量/LSE 并正确
  合并；不能简单平均结果。先按完整 FA 层分配可降低首版复杂度。
- CPU attention 保持 BF16 存储、tile 内转换与 FP32 累加，复用 6:1 GQA 的 KV，
  不复制六份 cache 或构造完整 FP32 K/V。
- 活跃 GDN 状态保持最新且与 FA KV token 位置一致。暂停/恢复时必须同时记录
  模型/artifact、token 前缀、位置和 GDN 快照；只恢复 FA KV 不构成完整恢复。
- 原生 BF16 存储不代表 CPU/GPU 规约顺序 bitwise identical，仍需误差与质量回归。

## 5. Prefill 与 decode 分开优化

### Prefill

GPU 负责主要 GEMM、FA 和 GDN 计算。先采用有界 token/query chunks + KV tiles，
融合 GDN preparation，保留 FLA chunk prefill 的可验证路径，避免全量 score matrix。
比较 token-chunk 优先与层优先：前者降低 activation 工作集，后者可复用当前层
权重与 0.5GiB KV，减少重复 PCIe/磁盘读取。层优先额外的 hidden buffers 必须计账：
一份 `[128K,5120]` BF16 hidden 已有 1.25GiB。必要时 prefill/decode 使用不同布局。

128K 首次 prefill 的全部输入处理计入 TTFT。不得用预先算好的 KV 或只测后续
decode 宣称完成最低配置端到端运行；前缀复用成绩另列。

### Decode

先比较三个可归因的实现：

1. packed 权重 + KV 送 GPU，计算全在 GPU，作为直观容量/传输基线；
2. GPU 做投影/GDN/主要 Linear，CPU 直接算其保有的 BF16 KV attention；
3. 部分 FFN/连续 block 在 CPU 计算，与向 GPU 搬同一份 packed 权重比较。

FA 投影常驻 GPU 与 FA attention 核心的执行设备是两个独立决定。
小于实际瓶颈的 kernel 优化不能代替磁盘/H2D 字节率分析。先有阶段计时，
再调整预取深度、shape buckets、fusion 与 graph；MTP 属于后续独立实验。

## 6. 开发顺序与验收

| 阶段 | 状态 | 工作与验收 |
|---|---|---|
| E0 分支与方案 | 完成 | 独立 worktree、与泛用路线的边界、本文及分支 Agent.md |
| E1 容量与 payload 规划 | 待开发，下一步 | 显式预算 GPU/RAM/SSD、buffers/峰值；注册所有 FFN/GDN/FA/embedding/head；不可行计划提前说明缺口 |
| E2 有界加载与权重执行 | 待开发 | 合成 payload 的磁盘→RAM→GPU 生命周期/异常测试；实际 packed Linear parity；不构造整模型 BF16 镜像 |
| E3 Hybrid 状态与 attention | 待开发 | BF16 KV 页存取、slot0/padding、prefill→decode、GDN 对齐；GPU 流送与 CPU attention 对照 |
| E4 完整短上下文 | 待开发 | 4K/16K/32K 同源模型 forward/generate；加载与运行均在预算内；noMTP、单请求 |
| E5 最低配置 128K | 待开发 | 从原始 prompt 完成 prefill 和约定输出，prompt+output<=131072；GPU/host/SSD 全程计量；不以隐藏 swap/额外内存达标 |
| E6 性能与质量优化 | 待开发 | 相同 artifact/input 下比较 placement、预取、chunk、CPU kernels/fusion；端到端质量与速度分别报告 |

packed artifact 还未就绪时，E1 和合成 I/O 工作可独立推进；只有取得同源实际 artifact
后才能声称 E2 的模型数值验证或 E4/E5 完成。旧 BF16 interoperability 质量得分
不能直接变成本分支的 packed/offload 质量结果。

验收至少覆盖：

- packing/scale/dtype/尾块、逐模块与块级输出、完整 logits/greedy 行为；
- 状态 slot0、padding、请求结束/取消/重开、KV/GDN 一致的 token 边界；
- 冷启动、JIT/warmup、prefill、decode、continuation 的峰值，而非只看稳态显存；
- 暴露物理/可用 RAM、WSL 配额、swap 活动、file-backed resident、pinned pool；
- 读取错误、缓存未命中与预取覆盖率，以及 SSD 实际 bytes，不能只报文件大小。

## 7. 实验记录规范与目前证据

每次记录：运行 ID/分支 commit、模型与 packed artifact hash、精度与 engine/kernel
版本、CPU/GPU/内存/驱动/OS/SSD/文件系统、资源限制、输入/输出长度、KV 页面布局、
常驻/缓存/流送 bytes、chunk/prefetch、TTFT、TPOT p50/p95、输出 tok/s、GPU/host
峰值、H2D/D2H、磁盘读写量和等待。多次试验写明 cold/warm 和汇总方法。

原始本地输出放 ignored `results/extreme-offload/` 或 `.local/research/`；
可复现参数放 `configs/`、脚本放 `benchmarks/`，无私密输入的精简结果可放 `manifests/`。
不提交权重、tokenizer、大日志、第三方源码或原始评测 prompt。

| 证据 | 状态/边界 |
|---|---|
| 模型几何与字节预算 | meta/config 已复核，不是实际 packed loader 峰值 |
| 上游 GDN preparation | 本机算子级 5.15x/3.17x，相对 eager；不是全模型成绩 |
| 上游 GDN recurrent | 64 步重复输入及 slot0/padding 小实验通过 |
| 本机 pinned H2D | 10.243GB/s，中位数；未同时做计算或磁盘 I/O |
| CPU BF16 attention、SSD 流送、完整 128K | **尚未实现/验证** |

详见继承的[源码研究记录](../../.codex/runtime-upstream-research-2026-09-05.md)和
[H2D 原始结果](../../manifests/bf16-kv-128k-h2d-20260905.json)。既有估算器
`benchmarks/runtime_memory_budget.py` 假设活跃 KV 全在 GPU，仅处理权重 offload；
它不是本分支 GPU/RAM/SSD 联合 planner，不能用它的容量判断替代 E1。

本机 WSL 上次只暴露约 11GiB RAM；16GB 物理内存、WSL 限额和 runtime 可用值
必须分开。扩大限额或采用 SSD 缺口模式是后续部署选择，本次未改系统设置。
吞吐目标尚未冻结，不能用一个固定 tok/s 数字将估算包装成验收结论。

## 8. 工作目录与分支操作

独立工作目录：`/home/txc_king/Project/LatticeRun-extreme-offload`。
原 `/home/txc_king/Project/LatticeRun` 继续留在 `main`。

```bash
cd /home/txc_king/Project/LatticeRun-extreme-offload
git status --short --branch
git pull --ff-only
```

此 worktree 没有复制原目录的 `.venv/`、`.local/`、模型或实验输出。新开发使用
分支自己的环境/输出路径；不要直接复用指向 main 源码的 editable install。
原目录 `.local/upstreams/` 可作为只读源码参考。运行已有 probe 时显式指定上游路径。

每个实现提交同步更新本文对应阶段和 `Agent.md`。在另一个 worktree 上进行的
main 训练/评估继续使用其自己的配置与状态，禁止把本分支的 6GB/SSD 默认值带入。
