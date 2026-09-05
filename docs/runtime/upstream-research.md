# Qwen3.8-27B Q3+OPD 的 kernel、显存与 dense offload 研究

研究日期：2026-09-05。项目基线：`ecccef9688e4c0df1be5735cd7aa01676d0f1b3e`。

本文是一次固定版本的源码研究与独立小实验记录，供 M6 使用；当前 M5 的混合 OPD/质量评估里程碑不变。没有运行 27B 权重、修改训练代码、启动远端 GPU 或宣称完成 packed runtime。`README.md` 保持不变。主要研究由父代理完成；dense offload 初筛按用户要求由 **gpt-5.6-terra / high** 完成，父代理随后亲自复查了 AirLLM、llm-layer-prefetch、llama.cpp、FlexLLMGen，并补查了 FLUTE/GemLite 的 W3 kernels。

## 1. 可以直接指导后续工作的结论

1. **先解决真 INT3 执行，再讨论 offload 加速。** 当前 `FrozenQuantLinear` 保存的是 BF16 fake-quant 权重，现有推理导出也明确标记 `packed_int3=False`。开启 CPU offload 并不会把这些权重自动变成 3-bit。
2. **weicj 的仓库最有价值的是 GDN 融合、压缩 KV 的执行路径、静态 workspace 和状态正确性经验。** 其主线路是双 22GB 2080 Ti + NVLink、TP=2、FP8/INT4/NVFP4；不是单张 8/16GB 卡的 dense INT3 offload 实现，也没有真 INT3 Marlin。
3. **Qwen3.8 的内存账需要分别计算权重、FA KV 和 GDN state。** 通过官方 config + Transformers meta 模型复核：26,895,998,464 个文本参数；真 INT3/INT4 + FP16 scale 约 9.972GiB，保留 FP32 scale 约 10.363GiB。128K、单序列的 16 层 FA KV 为 BF16 8GiB、该 fork 的 INT8 4.0625GiB；48 层 GDN 单份 FP32 recurrent state 共 144MiB。
4. **16GB/128K 首选 INT8 KV，尽量常驻权重；并非一开始就全模型流送。** 仅把约 0.48GiB embedding 放 CPU 按行读取，已经能改善边界。FP32 scale + INT8 KV 是否能完全常驻取决于真实 workspace/graph/显示占用；不足时只流送少量 FFN。FP16 scale 或更低比特 KV 属于单独质量消融。
5. **8–12GB 更值得比较“部分 CPU 计算”与“部分 packed 权重 H2D 流送”。** 前者只传激活但吃 CPU 算力/带宽；后者保留 GPU kernel 但每个 token 重传 offloaded 权重。没有一条路线在所有 PCIe/CPU 上都最优。
6. **低 RAM 不等于每 token 从磁盘重读全模型。** GPU 常驻部分不需要 CPU 镜像；CPU 仅保留 offloaded packed 部分、按需 embedding 和有界 staging pool。若这些活跃页也装不下，才会出现磁盘重复读取；这时延迟由 NVMe 下界决定。
7. **真 W3 原型优先看 FLUTE 与 GemLite experimental W3，Marlin 看流水线设计。** FLUTE 具有 W3G128、LUT 和独立 3-bit packing；GemLite 的 W3 实验代码提供 1-bit + 2-bit 双平面拼接。它们仍需要适配本项目的 scale 精度、模型形状、导出与数值契约。
8. **本机做了局部验证。** RTX 4050 Laptop 6GB 上，原仓库 post-conv Triton 融合对比等价 eager 操作链，512/2048 tokens 分别约 5.15x/3.17x；packed recurrent decode 的 64 步 FP32 state、slot 0、padding 验证通过。这是算子级证据，不是 27B 全模型提速。

## 2. 源码快照、位置和证据级别

所有上游源码在项目的 ignored `.local/upstreams/` 中，没有复制进 Git、安装其完整运行栈或下载模型权重。主仓库使用 shallow clone；其 GitHub API 大小约 196MiB，实际浅克隆工作树连 Git 约 54MiB。新增参考 clone 合计约 770MiB，已有 FreeToken/InstinctRazor 保留。

| 仓库与固定 commit | 本地目录 | 许可证 | 本次用途/审查深度 |
|---|---|---|---|
| [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be) `4d676458` | `.local/upstreams/vLLM-2080Ti-Definitive` | Apache-2.0，部分 FLA/FlashQLA 文件另含 MIT 归属 | 父代理主审；kernel 独立验证 |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/4d9176092d00586775af140581bb0b558ddc4389) `4d917609` | `.local/upstreams/llama.cpp` | MIT | 父代理复核模型图、GDN CUDA、loader、Q3_K |
| [lyogavin/airllm](https://github.com/lyogavin/airllm/tree/51d62f0c9ffbec201bb3cd3f08c02ce7a5276ff6) `51d62f0c` | `.local/upstreams/airllm` | Apache-2.0 | 父代理复核 Qwen3.8 wrapper、加载/释放、压缩限制 |
| [SergiuDeveloper/llm-layer-prefetch](https://github.com/SergiuDeveloper/llm-layer-prefetch/tree/3c7a28477347a636fee60a94456dbbb0f09ab9c0) `3c7a2847` | `.local/upstreams/llm-layer-prefetch` | MIT | 父代理复核完整流水线核心文件 |
| [FMInference/FlexLLMGen](https://github.com/FMInference/FlexLLMGen/tree/004ffef82b46e8dc8685c55d0cdda650bdaf1269) `004ffef8` | `.local/upstreams/flexllmgen` | Apache-2.0 | 父代理复核 placement policy、overlap loop、量化限制 |
| [HanGuo97/flute](https://github.com/HanGuo97/flute/tree/9eb83a12d56949bbe7fe9c836ba97a67bd1e3761) `9eb83a12` | `.local/upstreams/flute` | Apache-2.0 | 父代理补充：W3 packing、CUDA LUT GEMM、API 限制 |
| [mobiusml/gemlite](https://github.com/mobiusml/gemlite/tree/89d9bc705c5dfca9115d3a5620f97a17ba0111a7) `89d9bc70` | `.local/upstreams/gemlite` | Apache-2.0 | 父代理补充：experimental W3 和按 batch 分派 |
| [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers/tree/31985f40bcc40da08107efdb1f81bf88cb38c6b2) `31985f40` | `.local/upstreams/ktransformers` | Apache-2.0 | Terra 初筛；非本次重点复核 |

Terra 另保留了 `nano-llama.cpp` (`19794f31`) 和 `peterjweir/powerinfer` (`59df1750`) 的参考 clone。后者是历史 fork，**不能据此代表 PowerInfer 当前主线**。父代理另外查阅了 [现 canonical PowerInfer](https://github.com/Tiiny-AI/PowerInfer#faqs)（旧 SJTU-IPADS 地址重定向至此），确认其公开 FAQ 对稀疏激活模型的限制。Terra 完整原始报告留在 `.local/research/terra-high-dense-offload-survey.md`；其中概算、历史 fork 和初筛建议以本文的复核结论为准。

证据分类：

- **源码事实**：已读固定 commit 的函数、数据结构和分派分支。
- **上游自报**：README/benchmark summary 的成绩；未在同硬件复现，不当作 LatticeRun 结果。
- **本机实测**：只包括本文第 10 节的孤立 GDN kernels 和 scale 数值实验。
- **分析估算/建议**：显存公式、带宽下界、设备路线与实施顺序；均不是性能承诺。

## 3. 与当前 LatticeRun 的接口对照

### 3.1 已实现与未实现

| 当前代码 | 源码行为 | 对最终 runtime 的含义 |
|---|---|---|
| `src/quant/core.py:GroupwiseTensor` | codes 是 int8；scale 可指定 dtype | 训练/研究格式，不是 3-bit runtime |
| `src/quant/core.py:fake_quantize` | scale 为 FP32，最后转回原 weight dtype | 最终数值参照必须保留这一点 |
| `src/quant/core.py:pack_nibbles` | INT3 仍占 4-bit nibble | 不能作为真 3-bit 达成证据 |
| `src/quant/export.py:export_quantized_checkpoint` | 保存 int8 codes + 默认 FP16 scales | 当前 reference shard 不是最终 OPD packed artifact |
| `src/adapters/merge.py:FrozenQuantLinear` | 保存完整 BF16 weight，forward 调 `F.linear` | 名称含 Quant 不代表运行时读压缩权重 |
| `src/opd/export.py:export_inference_student` | merged/requant 后保存标准 BF16 safetensors | 适合既有质量评估，不适合 16GB 显存部署 |
| `src/runtime/` | 当前是 OpenAI-compatible client/concurrency smoke | 尚无 packed loader、planner、offloader 或 kernel |
| `src/model/qwen35.py` | 已分 FA/GDN/control、INT3/INT4/recovery policy | 应作为新 runtime 的模型策略来源 |

现有质量结果来自 INT3/INT4 **数值**的 BF16 runtime。保持这个结果作为质量 oracle，然后替换权重存储/执行；不能用 Q3_K/GPTQ/FP8 模型替换它并直接沿用 OPD 的质量结论。主参考 fork 自报 base vLLM 0.21.0，项目当前验证的是 vLLM 0.28；直接覆盖安装其 fork 会跨版本改变调度、cache、量化和模型 API。

### 3.2 导出需要避免两次量化

最终路径应为：

```text
同一个选定的 OPD adapter + 原始 base + 锁定 clip ratios
  -> materialize effective DoRA/LoRA weight
  -> 一次最终 quantize，产出 codes + scales
  -> 一路 dequantize 成当前 BF16 quality oracle
  -> 另一路 lossless pack 成部署 artifact
  -> 两路逐组/逐模块核对，再比较 forward
```

不要把已经 clipped/requantized 的 BF16 weight 再交给带原 clip ratio 的量化函数。这会再次缩窄范围，尤其 ratio=0.55–0.85 时，已经不是评估过的同一份模型。现有 `export_quantized_checkpoint` 会对输入 weight 再做 quantize，因此不能不加区分地拿来打包 frozen student。非 OPD 的 embedding/projection 也需要从正确的单次量化结果导出。

**scale 精度是一个真实差异，不是文件格式细节。** 本地 CPU 实验：seed=42，`randn(512,512,bfloat16)*0.1`，相同 codes/clip，比较现有 FP32-scale fake_quant 与默认 FP16-scale reference dequant。ratio=0.70 时 262,144 个 BF16 结果中有 23,387 个不同，max abs=0.0009765625；ratio=1.00 在这个样本中无差异。它证明不能假设 FP16 scale 天然 bit-exact，不代表整模型质量一定下降。

最稳的 M6 正确性基线：保留 FP32 scales，group 内 `(q.float()*scale.float()).to(BF16)` 的权重舍入语义；待端到端 parity 后单独评价 FP16/BF16 scales。后者每模型可省约 0.391GiB，但要经过同一质量 gate。

## 4. 官方架构与可复现内存账

官方 [Qwen3.8-27B 固定 config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)，revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`，Apache-2.0。只下载了 config 元数据，未下载 checkpoint。

用本地 Transformers 5.8.0 的 `Qwen3_5ForCausalLM` 在 `meta` 上构造模型，按现有 adapter 分类；总参数数与项目 8 月 31 日真实加载记录完全一致。冻结的几何/计数在 `manifests/qwen38-runtime-research.json`。

| 项目 | 配置/推导 |
|---|---|
| text layers | 64 = 48 GDN + 16 FA |
| hidden / FFN intermediate / vocab | 5120 / 17408 / 248320 |
| FA query / KV heads / head dim | 24 / 4 / 256 |
| FA q projection | 包含 output gate，输出维度是 `2*24*256`，不能漏算 |
| GDN key heads / value heads | 16 / 48 |
| GDN key / value dim | 128 / 128 |
| causal conv width | 4，最小持续历史 3 tokens |
| text parameters | 26,895,998,464；无 vision/MTP 参数加载 |
| quantized / protected parameters | 26,869,760,000 / 26,238,464 |

### 4.1 权重

真 INT3 G128 每组 codes 是 48 bytes；FP16 scale 再加 2 bytes，即 **3.125 bits/weight**。FP32 scale 为 3.25 bits/weight。INT4 lm_head 对应 4.125/4.25 bits/weight。这个有效位宽与 Q3_K 的 3.4375 不同。

| 部分 | 参数数 | 真 packing + FP16 scales，GiB |
|---|---:|---:|
| 所有 FFN | 17,112,760,320 | 6.2256 |
| GDN in_proj_qkv | 2,516,582,400 | 0.9155 |
| GDN in_proj_z | 1,509,949,440 | 0.5493 |
| GDN out_proj | 1,509,949,440 | 0.5493 |
| 所有 FA projections | 1,677,721,600 | 0.6104 |
| embedding | 1,271,398,400 | 0.4625 |
| INT4 lm_head | 1,271,398,400 | 0.6105 |
| protected tensors，按 2 bytes/param 估计 | 26,238,464 | 0.0489 |
| 合计 | 26,895,998,464 | **9.9720** |

FP32 scale 合计 **10.3631GiB**；表中的 protected 小张量精确 dtype、文件 header、布局 alignment 要在真正 exporter 中复核，tiny FP32 例外通过余量覆盖。不能把这张字节预算表称为已导出文件的测量。最大 decoder 层仅量化 payload 约 148.28MiB（FP32 scales），两个可复用 weight slots 约 296.56MiB；不应每层都各留一组 staging buffers。

所有 Linear 基本每 token 都要读；**embedding 只是查 token 行，不需要每 token 读完整 0.46–0.48GiB 矩阵**。CPU embedding 每新 token 返回 BF16 hidden 仅 10KiB，本身 packed 行含 FP32 scales 约 2,080 bytes。反过来，lm_head 每次计算整词表，不具备 embedding 的稀疏读取性质，应优先常驻 GPU。Prefill 只需为真正要采样的位置计算 lm_head，避免生成 `[128K,248320]` 全量 logits。

### 4.2 KV / recurrent state

对一个长度 T 的单序列：

```text
FA KV bytes = 16 layers * T * 4 KV heads * (K bytes/head + V bytes/head)
GDN state bytes = 48 layers * 48 value heads * 128 * 128 * 4 = 144 MiB
GDN minimal conv history = 48 * (2*16*128 + 48*128) * 3 * 2 = 2.8125 MiB
```

| T | BF16 KV | INT8 KV（K/V 各带一个 FP32 token/head scale） | TQ K8V4 | TQ K4V4+norm |
|---:|---:|---:|---:|---:|
| 16,384 | 1GiB | 0.5078GiB | 0.3789GiB | 0.2559GiB |
| 32,768 | 2GiB | 1.0156GiB | 0.7578GiB | 0.5117GiB |
| 131,072 | **8GiB** | **4.0625GiB** | **3.0313GiB** | **2.0469GiB** |

TQ K8V4 使用每 KV head 256B FP8 key + 128B value codes + 4B scale/zero，共 388B；不是 INT8 key 的同义词。K4V4 是 key centroid index + norm，value uniform scale/zero。INT8 可选 stride 对齐会进一步增大实际内存。依据：[fork INT8 cache shape](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/vllm/v1/attention/backends/triton_attn.py#L551)、[TQ slot layout](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/vllm/model_executor/layers/quantization/turboquant/config.py#L130)。

这些只是 payload。Paged block 的向上取整、allocator、attention 分块临时结果、CUDA graphs、MTP draft state、prefix-cache 中间状态和并发序列均需另加。**144MiB 是一份完整 GDN 当前状态集**；如果为每个可重用 prefix block 都保留全量 snapshot，其成本会按 snapshot 数倍增，不能把 hybrid cache 永远当成常数 144MiB。需要有限的可恢复边界与 CPU snapshot 分层。

## 5. 主参考仓库如何加速 prefill / decode

### 5.1 两个阶段的瓶颈不同

对 Linear，`M` 是本次处理 token 数，`N/K` 是矩阵形状：约有 `2*M*N*K` FLOPs。Decode B=1 时权重读一次只贡献少量计算，通常受权重读带宽、launch/同步和长上下文 KV 扫描限制；prefill 可把一个 weight tile 用在多行 activation 上，更容易受 GEMM/attention 计算与 workspace 限制。短 decode、多请求 decode、MTP 多 token 验证和长 prefill 不应强制共用一个 kernel 配置。

该模型每 4 层有 1 层全注意力，长 prefill 仍有二次于序列长度的精确 attention 计算。GDN 的 recurrent state 不随 T 增长，并不意味着整个 Qwen3.8 都是线性复杂度。

### 5.2 GDN prefill：先融合准备步骤

核心：[fused_gdn_prefill_post_conv.py](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py#L20)。

原先逻辑链是 split/rearrange → 若干 contiguous → Q/K L2 norm → gate/beta。注意 split/view 本身不一定复制，真正需要消除的是其后的布局 materialization、独立 kernel 与中间结果读写。

融合 kernel 的 grid 是 `(ceil(T/16), Hk+Hv)`：

- Q/K 分支直接按 mixed QKV 地址取数据，在 FP32 中做平方和、归一化，写最终连续 Q/K。
- V 分支直接写最终 V 布局，同时计算 `g=-exp(A_log)*softplus(a+dt_bias)`、`beta=sigmoid(b)`。
- Q/K/V 使用 activation dtype，g/beta 输出 FP32。
- 这样少走多次 HBM 往返与 kernel launch；不是把所有数据都永远留在寄存器，也没有融合前面的 conv、projections 或后面的完整 GDN。

**可借鉴程度高。** 本项目同样有 mixed QKV、QK norm、control path，且 FP32 gate 正是锁定策略。不过本 fork prefill beta 输出 FP32，当前 HF 模型的 `b.sigmoid()` 可能按输入 BF16 舍入；需要核对整个 GDN 的 cast 顺序，不能用局部近似等价取代模型 parity。

### 5.3 SM75 FlashQLA legacy：实际是 persistent recurrent CUDA

入口：[gdn_linear_attn.py:flashqla_legacy_chunk_gated_delta_rule](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/vllm/model_executor/layers/mamba/gdn_linear_attn.py#L187)。实际 CUDA：[gdn_forward.cu](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/tools/flashqla_sm75_patches/gdn_forward.cu#L34)。

不能仅按函数名里的 `chunk` 推断实现是现代 FlashQLA 的 chunk Tensor Core 算法。这个兼容文件中：

1. 按 batch、value head、value column 分配 CTA/subgroup。
2. `state_shard[COLS][rows_per_lane]` 先读状态，每线程维护小块 FP32 state。
3. 在同一个 kernel 中 `for (t=0; t<tokens; ++t)` 顺序递推；用 warp shuffle 做点积规约与广播，使用 `fmaf` 更新 state。
4. D=128 时 `COLS=4, WIDTH=16`，提高一个 warp/subgroup 内复用；最后写一次 final_state。
5. 本文件没有 `wmma`/Tensor Core GEMM 的 chunk 并行。寄存器是否 spill 仍要看编译结果，不能只凭局部数组宣称零 local-memory traffic。

这一做法的优势是避免每 token 独立 launch，以及避免每 token 把完整 state 写回再读入；代价是 token 轴串行、并行规模受 head/列数限制。调用 wrapper 还将 q/k/v/g/beta/state 转 FP32/contiguous，并将 output 转回原 dtype，额外产生转换/临时 buffer。

`forward_flashqla_legacy` 只接受单连续序列的受限条件，否则回退 Triton/FLA；该扩展 forward-only。**对本项目现代消费卡/Blackwell，保留已验证的 FLA chunk prefill，优先移植布局融合；legacy 作为 SM70/SM75 兼容思路，不作为无条件替代，更不能用于 OPD backward。**

### 5.4 GDN decode：按 packed QKV 直接更新 state

核心：[fused_recurrent.py:257](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/vllm/model_executor/layers/fla/ops/fused_recurrent.py#L257)，分派：[gdn_linear_attn.py:1682](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/vllm/model_executor/layers/mamba/gdn_linear_attn.py#L1682)。这里的 **packed 指 QKV 拼接布局，不是 INT3 权重**。

存储采用 `[value_dim,key_dim]` 的 state 行，写成数学形式：

```text
q,k = L2_normalize(q,k); q *= 1/sqrt(K)
alpha = exp(-exp(A_log) * softplus(a+dt_bias))
beta  = sigmoid(b)  # 保留代码中的输入 dtype 舍入
S_bar = alpha * S
u     = beta * (v - S_bar @ k)
S_new = S_bar + u outer k
o     = S_new @ q
```

单个 Triton kernel 读取拼接后的 Q/K/V、a/b、小型 FP32 controls，完成 normalization/gating/recurrent update/output/state write。以 `BV<=32` 切 value 轴，K=128 在单个 block 内完成规约，避免额外 QKV 拆分/contiguous/gate tensor。Conv update 仍是前一个 kernel。非 speculative、decode-only 才走这个直接分支；mixed prefill/decode 和 speculative 分支必须保留独立 metadata/状态语义。

**最直接的正确性经验：slot 0 不等于 padding。** 仓库专门修了把合法 state slot 0 跳过的问题，调用传 `PAD_SLOT_ID`；测试位于 `tests/2080ti/test_gdn_causal_conv1d_update.py`。迁移时必须覆盖 slot 0、负 padding、重排后的 state index、多请求、prefill→decode 和 accepted/rejected speculative state。

### 5.5 Marlin：低位宽搬运 + tile 内解码 + MMA

关键文件：[marlin_template.h](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/csrc/quantization/marlin/marlin_template.h)、[dequant.h](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/csrc/quantization/marlin/dequant.h)、[marlin_mma.h](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/csrc/quantization/marlin/marlin_mma.h)。Marlin 是既有上游技术，本 fork 做硬件适配/集成；不把整个算法归为 fork 原创。

实际提速链路：离线/加载期 repack 成 MMA 友好布局 → coalesced 读取压缩 weight tile 和 scales → shared/register 双层流水线 → 在 fragment 中解码、应用尺度 → MMA → FP32 累加/可选 reduction → 写输出。不会为每个 Linear 先写一整块 BF16 W 到 HBM，再调用普通 GEMM。激活 tile 复用、规约拆分和形状选择同样重要，不能仅看“4-bit 带宽减半”。

**SM75 的特殊事实：** `marlin.cuh:62` 对 `__CUDA_ARCH__<800` 的所谓 `cp_async*` 实际实现为同步 load/store，fence/wait 是空函数；真正 `cp.async` 在 SM80+ 分支。`generate_kernels.py` 为 SM75 生成两阶段而非四阶段配置；`marlin_mma.h` 把部分 FP16 MMA 拆成 SM75 能用的 `m16n8k8`。不能把它描述成 Turing 新增了硬件异步 copy，也不能把 FP8 checkpoint 路由称为 2080 Ti 原生 FP8 Tensor Core 运算。

**INT3 不兼容：** `query_marlin_supported_quant_types` 列出 uint4/uint4b8、uint8b128、float8、float4；repack/dispatch 多处限定 4 或 8 bits，没有本项目 signed INT3-G128 格式。把 INT3 放 4-bit nibble 可以作为过渡性能对照，但每组 code 从 48B 变 64B，违反最终真 3-bit 存储目标。值得移植的是 tile/dequant/流水线方法，而不是只把 `bits=4` 改成 3。

### 5.6 全注意力、INT8 与 TurboQuant

Prefill 的首块可以直接用当下高精度 K/V 做 FlashInfer/FA attention，同时把缓存写成低精度；没必要先量化、再为同一块反量化。后续 prefix continuation 则已经面对低精度旧 cache，需要专门路径。压缩的存储类型和 attention 的计算类型可以不同。

TurboQuant decode 的 [Triton stage1](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/vllm/v1/attention/ops/triton_turboquant_decode.py#L98) 直接通过 block table 读取 packed KV，在 tile 内恢复 key/value、在线更新 softmax 最大值/分母/加权和；再规约不同 KV split 的局部输出/LSE。因此无需在每 decode step 先产生整个 prefix 的 BF16 KV 副本。固定 split 数便于 CUDA graph；SM75 当前默认 `BLOCK_KV=2`，应按新硬件/维度重测，不能当成通用最优值。

Long continuation 存在不同分支：小 q 复用 decode kernel，避免 O(prefix) 的额外全量 dequant workspace；大 q 则可先 dequant 旧 cache 再调用高吞吐 attention。后者并没有消除缓存读流量。

**prefix combine 很有价值，但不是零拷贝：** [turboquant_attn.py:2266](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/vllm/v1/attention/backends/turboquant_attn.py#L2266) 将旧 prefix 的 non-causal attention 与当前 chunk 的 causal attention 分别计算，按 LSE 合并：若两部分结果为 O1/O2，LSE 为 l1/l2，则权重为 `exp(li-logaddexp(l1,l2))`。这样避免再次构造拼接后的大 K/V，但旧 prefix 此分支仍已反量化。须保留 mask/position/滑窗前提。

**workspace 必须先扣除，剩下的才分给 KV。** `_reserve_continuation_workspace` 按 max_model_len、batch tokens、配置 reserve 的最大值预留整段旧 prefix 的 FP16 K/V，不能只按 prefill chunk 长度预留。单 GPU 本配置 128K 一层 FA 的这份 workspace 就约 0.5GiB。该 repo 的近期修复正是避免启动时“KV 容量足够”，真正长 continuation 却 workspace 不足。

**wrapper cache 的释放必须跟 graph 生命周期一致。** `_get_or_plan_flashinfer_prefill_wrapper` 在普通模式先 LRU eviction 再构造新对象，避免瞬时 maxsize+1 workspace；graph-safe wrapper 模式反而禁止 eviction，因为 graph 还引用 wrapper 内的 indptr buffer。这不是普遍有界缓存，后续应控制图/shape 数量和对象生命周期。

### 5.7 MTP、CUDA Graph、TP 通信不能混成 kernel 倍率

- CUDA Graph 缓解 CPU 逐算子 launch 和 Python 调度成本，不降低必须读的权重/KV 字节。固定地址、shape bucket 和控制数据更新顺序是前提。
- TP=2 的 all-reduce 走 GPU↔GPU NVLink/P2P；它与 CPU↔GPU 权重 offload 的 PCIe 完全不同。参考机器一个 x1、一个 x4 插槽“也能快”，是有 NVLink 承担 rank 间数据，不证明这种 PCIe 拓扑适合权重流送。
- `custom_all_reduce.py` 区分 full graph registered-input 和 piecewise/prefill staging，避免 CUDA IPC/graph-private buffer 问题；单 GPU LatticeRun 没有对应的 all-reduce 收益。
- MTP 是草稿+目标验证的算法级摊销，依赖接受率；它增加 draft compute、state/graph 显存，recurrent 拒绝回退尤其敏感。fork 将高风险 full-graph/nosync 行为放在显式 mode/env 下，不该把这些开关当成通用提速技巧。

上游 [KV sweep](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/docs/qwen36-kv-throughput-sweep.md) 的 Qwen3.6 GPTQ-INT4、PP65536/TG512 自报：

| 配置 | prefill tok/s | decode tok/s |
|---|---:|---:|
| noMTP / FP16 KV | 1364.3 | 42.1 |
| noMTP / INT8 KV | 1345.8 | 52.2 |
| MTP3 / FP16 KV | 1328.0 | 85.5 |
| MTP3 / INT8 KV | 1290.7 | 50.1 |
| MTP3 / TQK8V4 | 1309.1 | 40.2 |

它说明更小 KV 不自动等于更快 decode，MTP 和 KV backend 会改变最优组合。该表的 raw JSONL 在上游工作树之外，精确运行 commit/完整系统参数并未随这些行公开；后来代码又有长上下文优化，不能视为本次 HEAD 的重测结果。

[LongGen3 真实文本 MTP 测试](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/docs/mtp-task-sensitivity.md) 中同 Qwen3.6 INT4 从 noMTP 43.59 到 MTP3 60.62 tok/s，约 1.39x；并非宣传峰值所暗示的固定倍数。本项目应先 noMTP 完成最终 Q3+OPD parity，之后重测该目标模型的 draft 接受率，不沿用 BF16/INT4 接受率。

## 6. 到底减少哪一段数据搬运

| 数据路径 | 有效做法 | 不能混淆的限制 |
|---|---|---|
| GPU HBM→SM registers/shared | 真 packing、tile 内 dequant、weight/activation 复用 | CUDA/Triton fusion 主要首先改善此处 |
| GPU 中间 tensor→HBM→下一 kernel | GDN prep/update、RMSNorm/residual、SwiGLU 等融合；直接最终布局 | view 本身不复制；融合过大可能寄存器 spill/occupancy 下降 |
| CPU RAM→GPU | 只传 packed bytes、pinned 小环、copy stream、events、预取 | `non_blocking=True` 不等于 guaranteed overlap；仍受 PCIe 带宽 |
| CPU pageable→pinned | 固定 staging；可评估 readinto pinned；避免整模型 `pin_memory()` 副本 | pinning 占不可回收 RAM；file readinto 仍是实际 I/O/内存写入 |
| SSD→CPU RAM/page cache | GPU 常驻部分不重复读；CPU 仅保留 offloaded 工作集；顺序大块 I/O | mmap 不是无物理内存/无磁盘流量；低 RSS 不代表低系统内存占用 |
| GPU→CPU→GPU state | 当前 FA KV/GDN state 尽量在计算设备常驻；prefix snapshot 有界冷存 | weight offload、active KV offload、prefix cache offload 是三种不同东西 |
| GPU↔GPU | 合理 TP 与高速互联、少量 collective launch | 单 GPU 不存在该收益；NVLink 峰值不能作为 H2D 估算 |

该 fork 自带 [PrefetchOffloader](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/vllm/model_executor/offloader/prefetch.py) 是比其主服务 profile 更直接的 dense offload 参考。文件注明适配自 SGLang，不能全归为 weicj 新发明。

关键实现：

- 按 `(name,shape,stride,dtype)` 建 static pool，各 layer 使用循环 slot。name 必须纳入 key，避免同层两个同形矩阵错误共享；stride 也不能省。
- CPU 放最终处理后的 pinned 权重；`post_init/sync_cpu_storage` 修正 repack/quantization 后的旧引用，防止预取旧权重或已删除临时参数。
- 参数指向固定 GPU slot，执行时不重新创建 BF16 Parameter；下一次 copy 改的是 slot 内容。
- compute 等该层 `copy_done`，copy 在覆盖 slot 前等 compute fork event；warmup/eager 与 capture 的 event 合法期分开处理。
- `prefetch_ops.py` 用 custom op 的 `mutates_args` 建立 compile 依赖，避免编译器把 wait/start 移过对应 forward；不只是 Python 上下文中开一个 stream。
- `join_after_forward` 在 graph 结束前 join copy stream，避免跨 capture 的悬空工作。

**移植时必须改注册对象契约：** 该 offloader 遍历 `named_parameters()`；LatticeRun 当前 frozen weights 是 buffers，未来 packed codes/scales 也可能是 buffers。它不会自动识别本项目；应基于显式 payload/tensor registry，而不是把现成类包在 `FrozenQuantLinear` 外面就认为生效。

**低 RAM 需要进一步改造：** 上游会为所有被 offload 的参数保留 pinned CPU storage，这不适合极低 RAM；本项目应 mmap/pageable 保存 offloaded 权重，只保留两三个 pinned payload slot。`UVAOffloader` 的 zero-copy 仅表示不显式 DMA，GPU 仍通过 PCIe 读 pinned host memory；dense GEMV 不能因此享有 VRAM 带宽。

**prefix correctness：** [offloading scheduler](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/4d676458714f1c291cca6ae9ccc0c6d7ecb1e8be/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py#L321) 每次 group clamp 后重新对齐 Mamba hit boundary，保证 FA token prefix 与 recurrent snapshot 属于相同位置。它缓存/恢复跨请求 prefix，不意味着当前活跃 128K attention 可以每步只读少量 KV。对 agent 多轮前缀复用很有借鉴价值。

## 7. Terra 初筛后，哪些 dense offload 实现值得采用

### 7.1 llama.cpp：优先建立成熟的分层执行/状态对照

父代理核查了 [src/llama-model.cpp:1477](https://github.com/ggml-org/llama.cpp/blob/4d9176092d00586775af140581bb0b558ddc4389/src/llama-model.cpp#L1477)：`n_gpu_layers` 在加载期决定各层 buffer 的设备；未上 GPU 的层在 CPU 上计算。默认 input embedding 保留 CPU。这是 **静态 placement**，不是每 token 把所有 CPU 层搬进 GPU。

[src/models/qwen35.cpp](https://github.com/ggml-org/llama.cpp/blob/4d9176092d00586775af140581bb0b558ddc4389/src/models/qwen35.cpp) 有真实 GDN/FA 交替图、独立 control 权重和 MTP trunk 区分。[gated_delta_net.cu](https://github.com/ggml-org/llama.cpp/blob/4d9176092d00586775af140581bb0b558ddc4389/ggml/src/ggml-cuda/gated_delta_net.cu) 的 fused recurrent 算子保留 FP32 state，并支持用于回退的有限 snapshot slots；`llama-memory-recurrent.cpp` 不把 recurrent state 当成任意可截断的 KV 列表。

最有价值：CPU/GPU 静态层分配、较轻 C++ 控制面、mmap/shard 生命周期、hybrid state oracle。需要实现 LatticeRun 自定义 GGML weight type/CPU+CUDA kernels/转换器，才能读同一份最终 artifact。

格式核查：[ggml-common.h:311](https://github.com/ggml-org/llama.cpp/blob/4d9176092d00586775af140581bb0b558ddc4389/ggml/src/ggml-common.h#L311)。Q3_K 每 256 权重 superblock 含 32B high mask、64B low2、12B 六位子尺度和 2B 总尺度，共 110B，3.4375 bpw；它是 16x16 子组，取值/尺度结构不等于 signed `[-3,3]` G128。转换成现成 Q3_K 会引入新量化，不是 lossless repack。可以用作独立系统速度基线，不能直接拿它的得分代表 Q3+OPD。

### 7.2 AirLLM：最接近“先能放进小 GPU”的现成结构

当前 [AirLLMQwen3_5](https://github.com/lyogavin/airllm/blob/51d62f0c9ffbec201bb3cd3f08c02ce7a5276ff6/air_llm/airllm/airllm_qwen3_5.py) 显式写出 Qwen3.8-27B 的 GDN/FA 架构；利用真正的 Transformers forward/generate/cache 逻辑，自己用 hooks 负责权重生命周期，不再重写每种 attention。

[airllm_base.py](https://github.com/lyogavin/airllm/blob/51d62f0c9ffbec201bb3cd3f08c02ce7a5276ff6/air_llm/airllm/airllm_base.py#L869)：模型在 meta；前 hook 读取当前层并等待预取 future、搬到 device，同时线程预读下一层；后 hook 将本层还原 meta。小于等于 2GiB 的层尝试 pin，但每次都会创建副本。未见在这条普通 hook 路径中建立静态 GPU slots + 专用 H2D stream 的完整机制；主要重叠的是磁盘读与当前层计算。

不应照搬的部分：

- Qwen wrapper 将 vision tower 常驻，违反我们 text-only 目标；这里只借鉴模型名映射/状态委托。
- `compression` 参数非空会关闭 prefetch；不能期待开启旧 4bit compression 后仍获得原预取行为。
- `clean_memory()` 每层调用 GC、`malloc_trim`、`torch.cuda.empty_cache`，利于极端容量，但容易增加 CPU/allocator 开销，破坏热路径复用。
- 原生支持并不包含本项目 W3G128；部分 compressed-tensors 路径会在加载时展开权重。
- README 的 Qwen3.8 **3.33GB VRAM** 是作者在 3090 的自报，缺少可对照的上下文/RAM/延迟/质量完整条件，不能推成 3.33GB/128K 或可交互 tok/s。

它适合做短上下文的容量/功能对照，后续较快 prototype 可以“沿用模型 forward + 自己的 packed linear/payload loader”。生产热路径应采用有界复用，不能每层 empty_cache。

### 7.3 llm-layer-prefetch：最清楚的有界 pipeline 示例

核心 [layer_streamer.py](https://github.com/SergiuDeveloper/llm-layer-prefetch/blob/3c7a28477347a636fee60a94456dbbb0f09ab9c0/src/layer_streamer/layer_streamer.py)：

```text
固定 pageable slots -> 固定 pinned slots -> 固定 GPU slots
       disk thread          pin thread          H2D thread
                                               copy stream
                                                    |
                                             ready event
                                                    v
                                              compute stream
                                                    |
                                             compute_done
```

父代理核查：CPU/pinned 各有 `n+1` slots，GPU 也为 `n_gpu+1`；pinned slot 复用前等待 transfer-done，GPU slot 覆写前等待 compute-done，compute 等当前层 H2D 完成。`readinto` 减少 Python 临时 bytes 对象，但不是免除 SSD→RAM 传输。

可直接借鉴 buffer 所有权和队列背压。不能原封不动用：它假定 `model.layers.N`、所有层共享 keys/shape/dtype；Qwen3.8 有两类层、混合 3/4bit、FP32 control。当前示例仅 Qwen2.5；thread 异常传播、短读校验、取消/回收、graph capture 和长上下文状态也需要完善。`run_pass` 每 token 重新起线程、`load_static` 把整个静态 tensor 放 GPU，都应按部署负载改造。**选择其清晰机制，成熟度上更优先参考 vLLM static offloader。**

### 7.4 FlexGen/FlexLLMGen：借其分阶段策略，不继承 OPT runtime

[Policy](https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/flexllmgen/flex_opt.py#L34) 把 weights/KV/activations 的 CPU/GPU/disk 分配分别建模，支持 CPU cache compute；[overlap loop](https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/flexllmgen/flex_opt.py#L1011) 在当前层前向同时预取下一层 weights/cache，跨 GPU microbatch 复用权重。

适用于大 batch 吞吐的逻辑不等价于单请求低 TPOT。代码围绕 OPT，single-batch loop 每层 sync，compression 明确断言 4-bit asymmetric，与本项目都不匹配。保留其 W/KV/activation 独立预算、不同 prefill/decode 策略、CPU attention 候选。

[FlexInfer MLSys 2025 摘要](https://proceedings.mlsys.org/paper_files/paper/2025/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html) 支持的研究方向是按硬件/负载为 prefill 和 decode 选择不同 CPU 计算策略。Terra 未找到公开实现，本次也没有将它当作可直接集成的开源 runtime；不把论文机器上的收益拿来填本项目性能表。

### 7.5 暂不选作主线的方案

- **KTransformers**：Terra 核查的 Qwen3.5 路线以大 MoE CPU experts 为中心；不能把它的专家缓存/活跃参数量搬到 dense Qwen。未来可研究 NUMA/CPU kernel，但没有现成锁定 INT3/G128 的 shortcut。
- **PowerInfer**：canonical FAQ 仍明确普通机制要求 ReLU/ReGLU/Squared-ReLU 稀疏模型。当前 dense SiLU FFN 不能按“预测哪些神经元不算”保持原模型输出。稀疏化重训是另一条研究线，不是 Q3+OPD 无损 runtime 优化。
- **FreeToken**：已有本地参考可继续用于 mmap/host buffers/通信思路；expert cache 不用于 dense 每层必算的语义。
- **纯 NVMe 层流送**：保留容量兜底，不把“成功 generate”与“低延迟可用”混为一谈。

## 8. 真 INT3 kernel：比直接改 Marlin 更合适的起点

### 8.1 FLUTE：任意 scalar LUT 可以表示我们的 codebook

父代理核查 [flute/utils.py:_pack_3bit](https://github.com/HanGuo97/flute/blob/9eb83a12d56949bbe7fe9c836ba97a67bd1e3761/flute/utils.py#L137)、[qgemm_kernel.hpp](https://github.com/HanGuo97/flute/blob/9eb83a12d56949bbe7fe9c836ba97a67bd1e3761/flute/csrc/qgemm_kernel.hpp)、[packbits_utils.hpp](https://github.com/HanGuo97/flute/blob/9eb83a12d56949bbe7fe9c836ba97a67bd1e3761/flute/csrc/packbits_utils.hpp)。

它支持 3-bit LUT、G128、FP16/BF16 输入。在存储码 `u=q+3` 下，可选择 scalar table `[-3,-2,-1,0,1,2,3,0]`，禁止真实权重使用末项，不进行其 NFL/NF/HIGGS 再量化，也不引入 Hadamard 模型变换。这只是表达既有 Q3 codes 的一种方式。

Kernel 使用 SM80 风格 copy/MMA pipeline，压缩索引分块读入，借助 vectorized LUT/warp shuffle 还原 fragment 并乘 scales，然后做 GEMM/Stream-K。`utils._pack_3bit` 是专门重排 96-bit 段并分组的真实 3-bit 路径；通用 `packbits_utils.pack_integer_tensors(...,3)` 反而 `NotImplementedError`，集成应使用正确入口。

限制：`flute/ops.py` 要求 scales 与 activation dtype 相同，不能直接保存本项目 FP32 scale；形状/SM 数与 template/tuning 耦合，Qwen 的 5120、17408、6144、14336 等维度都要实际验证，尤其 BF16 W3。旧 vLLM integration 要求 multiprocessing fork、假定旧 `QUANTIZATION_METHODS` 字典，不能直接套到项目 vLLM 0.28。**适合挑 kernel/packing 与调优框架，不是现成 Qwen3.8 完整 loader。** 本次没有编译/跑 FLUTE 或宣称其在 4050/Blackwell 的速度。

### 8.2 GemLite experimental W3：更易读的 Triton 原型

主 API `SUPPORTED_BITS_TRITON=[1,2,4,8,16]` **没有 3**；真正 W3 在 [experimental/A16W3_gemm.py](https://github.com/mobiusml/gemlite/blob/89d9bc705c5dfca9115d3a5620f97a17ba0111a7/gemlite/triton_kernels/experimental/A16W3_gemm.py#L66)。

其 packing 将 u 分成 `u & 1` 与 `(u>>1)&3` 两个平面；总计 1+2=3 bits，不需要跨每个 32-bit word 拼不整齐的连续 3-bit 字段。kernel 两路载入后 `(b2<<1)|b1`，在 tile 内反量化，并直接 `tl.dot`，无需全 W 展开。

我们的变体可以令还原 `q=u-3`，scale 维持 row/input-group 的语义。这里的 3 是**存储编码偏移**，不会改变已锁定的数值 zero-point=0 对称量化。padding 要编码真实 q=0（u=3），不能随便填 packed zero bits（那代表 q=-3）。也可设计二补码，但 artifact 必须明确且两端一致。

移植前要处理：该实验默认部分入口使用 FP16 accumulation、依赖较早 Triton API、有未覆盖尾块 load、重型 autotune、测试/示例与可用 kernel 混放；不是生产接口。先给我们真实 shapes 做 masked tails 和 FP32 accumulation/parity。GemLite 主线按小 M 用 GEMV/reverse-split-K、中 M 用 split-K GEMM、大 M 用 GEMM 的分派思想可借鉴，但 threshold 要重新测。

### 8.3 建议的实现顺序

1. **先有正确的 W3 artifact + CPU unpack oracle。** 以 128 权重 48B、显式 scale dtype、valid K、tensor dtype、layout/version/hash 表达，不用每层运行时重排。Embedding 用可按行读取布局，Linear 用 kernel 友好 tile layout；同一逻辑 codes 不要求同一物理布局。
2. **decode 专用 W3A16 GEMV。** 合并 code load/unpack/scale/乘加，FP32 reduction，block 内复用 activation；候选 split-K 与按输出行并行。B=1 不能为了 Tensor Core 将 M=1 无条件补 16 行后还宣称获得相同比例收益。
3. **prefill W3A16 GEMM。** 以 GemLite W3 双平面/FLUTE 为候选，tile 内 dequant 后 MMA，按 M 与 N/K 选择配置。小模型先验证 exact codes + reference BF16 rounding，再看 matrix accumulation 的可接受误差。
4. **必要时可用单个 tile/单个 Linear 的 BF16 scratch + cuBLAS 作性能对照。** 它增加 unpack 写回流量但大 M 时可能有竞争力；必须有硬性 workspace 上限。不能为了 prefill 一次展开整个 27B 或在所有层永久保留 BF16 镜像。
5. **再融合邻接轻算子。** QKV/gate-up 共同输入的合并 projection、RMSNorm+residual、SwiGLU、GDN prep/update、最后 norm+lm_head/采样。保留每模块独立 clipping/scale；不能将 BF16 控制投影 a/b 与 INT3 主投影误合并到同一精度。
6. **最后才做 graph、跨设备并行、MTP。** 最初保持一个请求、固定 shape bucket、noMTP，避免 format、量化、cache 和 speculation 同时变动导致无法定位。

## 9. 更低显存与内存的部署方案

### 9.1 默认设计：静态常驻 + 最小化 offloaded 工作集

三个对象分开：`PackedWeightStore`、`ExecutionPlan`、`HybridStateStore`。这些是建议的责任边界，不是本次创建的空模块。前两者通用；模型层类型/受保护 tensor/state 形状由现有 model adapter 提供。先选一个运行宿主，避免同时维护两个完整引擎：

- 若优先延续现有 BF16/FLA 的质量 oracle：薄的 Transformers/vLLM custom linear 接入更易做逐层 parity，采用固定 payload slots；要限制 Python/compile 常驻内存。
- 若优先非常低 RAM 与成熟 CPU/GPU 层执行：llama.cpp 的 C++ 后端值得作为后续候选，但需要真正的自定义 INT3 type/kernels 和 Qwen prompt/模型数值验证，工作量不是一次格式转换。

默认分配：control/conv/norm + 当前 GDN state、活跃 FA KV 留在执行 GPU；lm_head 尽量 GPU；embedding 按行 CPU 查询；剩余 packed weights 尽量常驻，缺口主要从占比最大的 FFN 中找。若静态 CPU block 能减少跨设备往返，则按连续块分配；若采用 component streaming，则要同时考虑依赖间可覆盖的 compute 时间。不存在 dense expert hit-rate 这种收益。

流送路线中，copy stream 提前取下一 payload，compute 等 ready；slot 用完只标记可复用，**权重不需要再从 GPU 拷回 CPU**，因为 CPU 有只读来源。保存 CPU/GPU 双份的只有正在缓冲/流送的 offloaded 部分，不保留全模型 pageable+pinned+GPU 三份镜像。

### 9.2 设备矩阵：容量估算，而非速度承诺

以下用 **GiB**；单请求；CPU embedding；FP32 scales 为默认；GDN FP32；假设 workspace/activations/graph 合计 1GiB、额外 safety/allocator/display 1GiB；需要流送时再加两个最大 packed decoder slots。量化压缩 KV 的质量未验证。整层分配需要向上取整，表中只是连续字节近似下界。

| 设备显存 | 上下文与 KV | 估算需放 CPU 并反复使用的权重，不含 embedding | 路线判断 |
|---:|---|---:|---|
| 16GiB | 128K BF16 KV | 约 4.315GiB | 可保不量化 KV；但 offload 显著，非最快默认 |
| 16GiB | 128K INT8 KV / FP32 scale | 约 0.378GiB | 最值得首先测；实际开销更低可全常驻，反之少量 FFN 流送 |
| 16GiB | 128K INT8 KV / FP16 scale | 0，估算总显存 15.715GiB | 容量更好，但 scale 降精度须单独质量验证 |
| 16GiB | 128K TQ K8V4 / FP32 scale | 0，估算总显存 15.057GiB | 容量可观；TQ 未必比 INT8 快/准 |
| 12GiB | 32K INT8 KV | 约 1.331GiB | 少量 offload，适合先做消费卡落地 |
| 12GiB | 128K TQ K8V4 | 约 3.346GiB | 长上下文功能路线，需权衡质量/TPOT |
| 8GiB | 16K INT8 KV | 约 4.823GiB | 比较 CPU 部分计算与 H2D，系统 RAM 建议先 16GB 验证 |
| 6GiB | 16K INT8 KV | 约 6.823GiB | 容量实验可设计；本机还有显示占用，不能按6GiB全可用；先缩到4K/8K |

此表不说明某个现成 engine 已运行成功。特别是低位 KV、GDN 状态、packed loader 与图 buffer 同时生效还未做整模型测试。

估算器可复算/改参数：

```bash
python3 benchmarks/runtime_memory_budget.py \
  --vram-gib 16 --context 131072 --kv-format int8 \
  --scale-bytes 4 --cpu-embedding --workspace-gib 1 --safety-gib 1
```

prefill 分块从 256/512/1024/2048 做 sweep。更大 chunk 增强 GEMM/weight reuse，但增大 activation 和 attention 临时量，也可能增加 TTFT 尾部；不默认用最大值。流送 prefill 每个 token chunk 都可能重复过一遍 offloaded weights，代价约 `ceil(T/chunk)*W_off`；层优先、跨 chunk 权重复用可以减少搬运，却可能要额外保存整段 hidden/激活到 RAM/SSD，在“低 RAM”约束下需要一起算账。

### 9.3 Host RAM 预算与磁盘下界

用物理内存工作集而非仅 RSS 计：

```text
RAM ≈ CPU常驻的offloaded packed weights + 常用embedding页
    + pinned slots + 必要pageable read slots
    + CPU计算scratch/CPU KV(若选择)
    + runtime/allocator + OS/其他进程
```

file-backed resident pages 算物理 RAM；如果与 mmap 是同一批页不能再重复计为另一份 cache，但 `.clone()`/`pin_memory()`/load 后复制会形成真实额外副本。`mlock` 整个模型会抵消低 RAM 的目标。

16GiB GPU、权重大部分常驻时，设计上可将 host 模型工作集压到不到 1GiB，再加少量 pinned ring、运行时和 OS；因此 **8GB RAM 是值得实测的紧凑部署目标**，不是需要先在 host 装下 50GiB BF16。前提是 server 产出最终 packed artifact，消费机直接 meta/streaming load，加载期不构造完整 BF16、int8 codes 或 DoRA base/adapter 镜像。

8GiB GPU 的示例则需约 4.8GiB offloaded weights、0.48GiB embedding 工作集上界、约 0.3GiB pinned ring，再加运行时/OS，8GB RAM 会较紧；16GB RAM 更现实。6GiB GPU 示例约 6.8GiB offloaded 权重，8GB RAM 很容易触发反复 fault；需要更短上下文、更多 CPU/GPU placement 调整或接受磁盘路线延迟。

**带宽公式：** 对非 speculative B=1 decode，一步必须传 W_off，

```text
T_step >= W_off / BW_H2D
若每步都要从SSD读D_off，T_step >= D_off / BW_disk
理想充分流水：T_step >= max(T_GPU, T_CPU, T_H2D, T_disk)
实际还有依赖气泡、串行阶段与传输/计算争用，通常更慢。
```

例：3GiB / 24GB/s = 134ms，仅 H2D 就把上限压到约 7.45tok/s；3GiB / 12GB/s = 268ms，上限 3.73tok/s。带宽是**假设的持续有效带宽**，不是该机器测量或 PCIe 标称保证。GDN/FA/解包/CPU 同步只会使实际更慢。double buffering 只能隐藏可重叠部分，无法突破总链路字节率。

若极低 RAM 导致几乎全部 active Linear weights 每 token 从盘读，FP16-scale 模型扣除 embedding 后约 9.51GiB，3.5GB/s 需约 2.92s/token，上限 0.34tok/s；若已有 GPU 常驻、实际只缺 3GiB，就用 3GiB 计算，不能夸大成每步读整模型。冷启动、warm mmap、部分 page-cache thrash 必须分别测。

### 9.4 CPU 计算何时优于搬权重

对候选 FFN/整层分别测：

```text
GPU streaming成本 = packed H2D + GPU W3 kernel + 未被覆盖的等待
CPU计算成本       = CPU W3 kernel + 小activation边界传输 + 同步
```

Decode 的 hidden 通常只有 10KiB，CPU 算一组连续层再传激活，可能比重传几 GiB 权重更好；但 CPU W3 SIMD kernel、DRAM 带宽、线程/NUMA 决定结果。不能套用现代服务器 AMX 成绩到普通笔记本。基线应使用原 BF16 activation/FP32 accumulation 语义；若为了 CPU dot-product 改 activation INT8，属于新数值消融。

更远期可按 FFN intermediate 维度切分：CPU/GPU 各算自己的一组 `gate/up`、SwiGLU 和 `down` 部分，最后加和输出。所有 dense 神经元都计算，区别于预测跳过神经元；这需要真正的负载均衡和通信测量，先不作为 M6 首版。

128K 若 FA KV 仍放不进 GPU，**CPU 上计算部分 attention、只传 Q/结果**可避免每 token 搬整段 KV；但 CPU 每步仍要扫自己的 KV，内存与延迟都可能很高。把压缩 KV 往返 GPU、把 KV 放 CPU 算、缓存旧 prefix 是三种不同方案，分别与缩短 context/压缩 KV 比较。GDN 单份 state 仅约 147MiB 连 conv，不应先牺牲其 FP32 精度去省几十 MiB。

## 10. 本机验证与可复现证据

### 10.1 独立 GDN kernel probe

脚本：`benchmarks/upstream_gdn_probe.py`；结果：`manifests/gdn-upstream-probe-20260905.json`。脚本核对 upstream commit，只加载指定的原函数定义，保留原文件/行号与源码 SHA256；依赖由本地 Torch/Triton 提供，不安装整个 fork。上游文件没有修改，原 vLLM/FLA 归属与许可证保留。

环境：WSL2、RTX 4050 Laptop GPU（SM89，6141MiB 总显存，实验前显示等约占 1182MiB）；Torch 2.9.1+cu128，Triton 3.5.1；BF16 random inputs，GDN Hk=16、Hv=48、K=V=128；seed 42。时钟未锁，非空闲服务器隔离环境。每路径 warmup 10 次，5 轮各 30 次，CUDA events 取每调用中位数，包含 eager launch gap，无 CUDA Graph。probe 自身 peak allocated 265.88MiB。

| 算子 | fused 时间 | eager reference 时间 | 比值 |
|---|---:|---:|---:|
| post-conv preparation，T=512 | 0.0961ms | 0.4948ms | 5.15x |
| post-conv preparation，T=2048 | 0.6763ms | 2.1457ms | 3.17x |
| packed recurrent decode，2 live + 1 padding | 0.0517ms | 未做公平的优化 backend 对比 | 不报告加速倍数 |

Q/K 最大绝对误差 <=0.0009765625，V 完全相同，g/beta 接近 FP32 误差级。Decode 从相同初态重复同一组输入 64 步：FP32 state max abs 5.66e-7，output max abs 7.63e-6；slot0 正常更新，未用 slots 不变，padding 输出零。重复输入状态测试还不能替代随机长序列、完整 conv/模型、混合 batch 或 MTP 回退测试。

这支持“减少独立 launch 和中间内存操作值得做”，但没有对比 torch.compile/现有 FLA 已优化融合版本，也没有测完整 prefill、TTFT、27B decode 或端到端质量。若该 preparation 原先只占总耗时比例 f，即使局部加速 s，总上限仍是 `1/((1-f)+f/s)`。

复现命令：

```bash
.venv/bin/python benchmarks/upstream_gdn_probe.py \
  --upstream .local/upstreams/vLLM-2080Ti-Definitive \
  --output .local/research/gdn-probe-repeat.json
```

### 10.2 scale 数值实验

无需 GPU/权重：

```python
import torch
from latticerun.quant.core import quantize_symmetric_groupwise, fake_quantize

torch.manual_seed(42)
w = torch.randn(512, 512, dtype=torch.bfloat16) * 0.1
reference = fake_quantize(w, clip_ratio=0.70)
q = quantize_symmetric_groupwise(w, clip_ratio=0.70, scale_dtype=torch.float16)
actual = q.dequantize(dtype=w.dtype)
print((actual != reference).sum().item())  # 23387, 当前 Torch 2.9.1 CPU
```

这项验证提示保留 scale contract；本次没有改变已完成 OPD 的训练或导出数值。

## 11. M6 后续实验顺序与验收条件

| 顺序 | 最小实验 | 要解决的问题/验收 |
|---|---|---|
| 1 | 选定最终 mixed5k OPD adapter，同次量化分出 BF16 oracle 和 packed codes/scales | 单次 clipping；无 adapter；所有 tensor/policy、scale dtype、source revision/hash 可追溯 |
| 2 | CPU bitpack/unpack + GPU W3 GEMV/GEMM | 7 个合法 codes、unused code、跨字/组/尾维、embedding 行、lm_head INT4；逐组还原 oracle；不得整模型反量化 |
| 3 | 一个 GDN block 与一个 FA block | weight kernel 误差与 state/cache 误差分开；FP32 controls、Q gate、norm/conv、token 顺序和 dtype 一致 |
| 4 | 短上下文 full runtime，无 offload/noMTP | 与既有 BF16 interoperability 比较 logits、greedy tokens、quality；测 W3 实际带宽与 workspace |
| 5 | GPU 常驻/CPU 固定层/packed H2D 三种布局 | 相同模型/输入/context，实测 CPU、H2D、GPU 时间与 resident bytes，选 Pareto 点 |
| 6 | 4K→16K→32K→128K，单独切换 KV dtype | 测 prefill/continuation 峰值，而非仅初始化；INT8/TQ 的质量与速度各自可回归 |
| 7 | pinned ring、prefetch depth、CUDA Graph | ready/compute_done 生命周期、warmup/capture/replay、无意外全局 sync；证明有实际 overlap |
| 8 | 多轮 agent prefix 复用与可选 MTP | prefix位置与GDN snapshot一致、拒绝回退正确、收益基于最终 Q3+OPD 接受率 |

性能记录至少包含：模型/量化/engine/kernel commit、GPU/驱动/CUDA/CPU/RAM/PCIe/功率状态、prompt/context/generated tokens、batch/concurrency、cold/warm/JIT、TTFT、prefill tok/s、decode TPOT p50/p95、实际输出 tok/s、GPU allocated/reserved/process peak、host RSS+file-backed resident/pinned、disk bytes/page faults、H2D/D2H bytes 与等待占比。短任务还要单列启动/加载/首次 compile，不能用 warm microbench 掩盖它们。

诊断工具对应问题：CUDA events 做阶段计时；Nsight Systems 看 CPU launch、copy/compute overlap、同步和跨设备依赖；Nsight Compute 看 DRAM 吞吐、L2 命中、寄存器/溢出、occupancy、MMA 利用率。未 profile 前不预设所有慢都来自 kernel 算术。

后续 M6 第一项最具体的工作应是 **同源 packed export + W3 kernel parity**。GDN 融合、最小 offload planner 与有界 buffers 的源码已在本地备齐；128K/16GB 的潜在缺口和低 RAM 限制已用可复算公式明确，不需要靠继续堆更多上游仓库来决定第一步。

## 12. 追问补充：BF16 KV + 128K，在 8GiB / 6GiB 上单请求运行

用户明确将最终 adapter + 常规 Q3 merge/真 packing 视为可解决的前置工作。本节据此讨论容量与速度，不再把导出当作可行性的阻碍，也不以 INT8/TQ KV 替代用户要求的 BF16。这里 128K 是 **prompt + 已生成 tokens 的总有效长度约 131,072**；若 prompt 本身已满 128K，还要继续生成，预算需随总长度增长。

### 12.1 可行性判断与示例分配

**通过有界 weight/KV 流送，8GiB 和 6GiB 单请求在容量上均可设计实现。** 保留全部历史、BF16 KV，无需稀疏检索、滑窗截断或低比特 KV。但这不是当前某条现成启动命令已经跑通，也不能据此保证交互速度。

128K 的全部 FA KV 共 8GiB，单个 FA 层仅 0.5GiB；还可按 token 页切成更小块，通过在线 softmax 合并。模型从来不需要同时将所有 16 个 FA 层的 cache 放进 GPU。GDN 的约 147MiB 当前 state/conv、FP32 controls 优先常驻，不存整段 GDN 历史；暂不启用额外 prefix snapshots/MTP。

以下是一个容量示例，采用 FP32 scales（换成 FP16 scales只会略减权重容量），embedding 留 CPU。GPU 的 2.5GiB 预留**已经包含** protected weights、GDN state、weight/KV staging slots、activation/workspace、allocator/显示余量；不是另加 2.5GiB。常驻权重指量化 Linear payload，CPU 权重计数不含常驻 GPU 的副本。

| 分配项 | 8GiB GPU | 6GiB GPU |
|---|---:|---:|
| GPU 常驻量化 Linear 权重 | 4.5GiB | 2.5GiB |
| GPU 常驻 BF16 KV | 1GiB | 1GiB |
| 其他 GPU 预留合计 | 2.5GiB | 2.5GiB |
| CPU 量化 Linear 权重 | 5.333GiB | 7.333GiB |
| CPU BF16 KV | 7GiB | 7GiB |
| CPU embedding 全部驻留时 | 0.481GiB | 0.481GiB |
| CPU 权重 + KV 工作集 | **12.814GiB** | **14.814GiB** |

量化 Linear 总计 9.833GiB；加 embedding 和 protected tensors 后才是全模型 10.363GiB。表中尚未加入 host pinned/pageable ring、CPU scratch、框架、OS。整层分配也需取整。**建议先用 32GB 系统 RAM 验证两档显卡；16GB RAM 对 8GiB GPU 已很紧，对 6GiB 更容易分页。** 这并不证明 16GB 一定不可运行，只是不宜作为无换页的默认预算。

本机检查：RTX 4050 Laptop 总显存 6141MiB，检查时已有 808MiB 显示/其他占用；WSL 内 `free -h` 仅见约 11GiB 总内存、9.3GiB available。它不是 Windows 物理内存容量报告。在当前 WSL 配额内，上述 6GiB/128K 方案的 CPU 权重/KV 工作集已经装不下；不能把 8GiB swap 当作同等速度的 RAM。宿主若有足够物理 RAM，扩大 WSL 配额是后续部署条件之一，本次未修改系统配置。

### 12.2 全部算在 GPU：每 token 都要搬运历史 KV

对普通、非 speculative 的单 token decode，精确 dense attention 要使用全部历史 KV。CPU 保存 KV 只解决容量；若 attention 仍在 GPU，offloaded 部分每步都要送过 PCIe。缓存可以保存在 BF16 CPU 主副本中，每步只写回新增部分：全模型每 token 新增 K/V 共 **64KiB**。旧 token 的 post-RoPE K/V 不变，无须把整段旧 cache 再 D2H 写一遍。

[Transformers 官方 cache offloading 文档](https://huggingface.co/docs/transformers/main/en/kv_cache#cache-offloading) 描述了逐层异步预取、当前 cache 返回 CPU 的通用路线；不能直接把其完整 cache 往返策略当作本项目最小流量方案，也不能仅设置参数就假定 Qwen hybrid state 已适配。

上述示例的 H2D 每步最低有效 payload 约为：

```text
8GiB GPU: 5.333GiB weights + 7GiB KV = 12.333GiB/token
6GiB GPU: 7.333GiB weights + 7GiB KV = 14.333GiB/token
T_decode >= (W_off + KV_off) / BW_H2D
```

在“所有计算在 GPU、每步每个 payload 至少读一次、相同精度”的条件下，把 1GiB 常驻 KV 换成 1GiB 常驻权重，通常只是把搬运缺口从 KV 移到 weight，不会神奇地消灭每步这 1GiB 流量。权重/FA 层有不同计算密度、重叠机会和页开销，实际 placement 仍需调优；这不是说两种放法的速度永远相同。

本次补做了纯 H2D 基准：四个 64MiB pinned CPU buffers、两个 64MiB GPU buffers，一条独立 copy stream；16 次 warmup，5 轮各 64 次 copy，每轮 4GiB；CUDA events 计时。RTX 4050 / WSL 中位数 **10.243GB/s**，各轮 10.100–10.306GB/s。原始结果与完整协议保存在 `manifests/bf16-kv-128k-h2d-20260905.json`。此测试没有 CPU staging、attention 或 GPU compute 争用，因而是较理想条件下的实测参考，不是实际 serving 带宽保证。

按这个实测带宽推导：

| 示例 | 仅 H2D 时间下界 | 仅 H2D 的 tok/s 上限 |
|---|---:|---:|
| 8GiB 的分配，假设相同链路 | 约 1.29s/token | 约 0.77 |
| 本机 6GiB 的分配 | 约 1.50s/token | 约 0.67 |

更快设备若**实际持续**达到 24GB/s，两档也分别至少约 0.552/0.641s/token，上限 1.81/1.56tok/s。这里没有加 GEMV、GDN、attention、同步及 pageable→pinned 成本。即使某个 kernel 快 5 倍，copy stream 仍不能超过链路字节率。调大 prefetch 深度可以减少气泡，但多放 buffers 也挤占常驻容量。

复算本机下界：`GiB * 2**30 / (10.243019378919296 * 1e9)`。这只约束上述纯 GPU 逐 token 流送路线，不是所有 CPU/GPU 算法或 speculative 多 token 验证的普遍上限。

### 12.3 更值得优化：CPU 保管 BF16 KV，并计算对应的 decode attention

对于 CPU 上的 KV，让 CPU 直接算 `QK^T -> softmax -> AV`，GPU 只发送当前 Q/新增 K/V 并接回 attention 输出。全模型每步这些边界数据约几百 KiB，而不是几 GiB 历史 KV。GDN、低位 Linear 和尽可能多的 FFN 仍可在 GPU；剩余 Linear 再比较 CPU 计算和 packed H2D。

可以按完整 FA 层分设备，也可按 KV heads 分配；若按同一 head 的 token 范围拆给 CPU/GPU，必须分别返回 softmax 统计量/LSE，再按正确归一化合并，不能简单平均两个 attention 输出。保留所有 tokens/heads，不做 top-k 检索。BF16 cache 数据不降精度，累加/softmax 使用 FP32；设备/规约顺序改变仍需数值 parity，不承诺 bitwise identical。

已有源码依据是 FlexLLMGen 的 `cpu_cache_compute` 和 `pytorch_backend.py:422` 的 CPU dense-attention 分支。其 `_mixed_device_attention` 按 batch 分段，**不是单请求上下文切分+LSE 合并的现成实现**；CPU 分支还整体 `.float()` 展开 K/V，不能照搬到低 RAM 方案。我们需要 BF16 存储、CPU tile 内转 FP32、用后复用 scratch，禁止常驻一份完整 FP32 KV 镜像。

本模型每个 KV head 对应 6 个 query heads，CPU kernel 应复用同一 KV tile，不能为 GQA 显式复制六份 cache。CPU 仍需读自己的全部历史 KV，并完成计算；把 8GiB KV 留 CPU，每步至少有 8GiB 的 host 内存读量。例：有效 40GB/s 的**假设**下，仅这一遍读取就是约 0.215s；实际还有转 dtype、运算和权重 H2D 争用。i7-12700H 当前暴露 AVX2、没有 AVX512-BF16，不能套用 AMX 服务器成绩。该 CPU attention kernel 本次尚未实现或测量。

因此，这条路线是改善 128K decode 的优先候选，而不是保证高 tok/s。也可先测 CPU 完整 FA 层/连续 block 的较简单实现，再考虑更细拆分。

### 12.4 Prefill 必须另设调度

128K prefill 优先在 GPU 上做 attention 和 GEMM：按 token chunk、KV tile 控制峰值；CPU 保留 BF16 历史页；重叠下一层/下一页的 H2D；用 online softmax/FlashAttention 思路避免分配 `[heads,128K,128K]` score matrix。当前 chunk 结束后，只存新增 KV，48 层 GDN 只保留继续递推所需的状态。转到 decode 后再切换为 CPU KV attention，不要求两阶段执行位置一致。

分块没有消除 dense attention 的平方计算量。仅 16 个 FA 层的 causal QK/AV，128K 下约 **3.38×10^15 FLOPs**；再加不计算全部 prompt lm_head 的主要 Linear，粗算另有 **6.38×10^15 FLOPs**，尚未精确计 GDN/其他算子。若整体有效吞吐假设为 30TFLOP/s，这些算术量就约 325s；10TFLOP/s 时约 976s。这是 FLOP 等价算术估算，W3 解码、kernel 效率和 I/O 还会影响实际值，并非本机 TTFT 测量。

chunk 太小还会重复读取 CPU prefix：若 offloaded KV 在最终长度为 C，均匀生成 cache，N 个 token chunks 每次读取之前所有 offloaded prefix，流量约 `C*(N-1)/2`。例如 C=7GiB、chunk=512、N=256，约 892.5GiB；chunk=2048、N=64 时约 220.5GiB，均未加权重流量。实际层/head/token placement 会改变这一估计。

应比较两种次序：

- token-chunk 优先：activation 小，但权重和旧 KV 可能反复从 CPU 读取；
- 层优先或更大 query blocks：尽量让当前 FA 层的 0.5GiB KV、当前层权重常驻，跨 query chunks 复用；但一份 `[128K,5120]` BF16 hidden 已有 **1.25GiB**，输入/输出双缓冲就约 2.5GiB，会挤占 host/GPU 空间。

后者可在足够 RAM 时用 CPU hidden buffers、GPU 当前层 scratch，prefill 期间临时释放部分 decode 常驻权重，结束后再切回 decode placement。它减少权重/KV重复 PCIe 读取，但不是零激活搬运。低 RAM、低 TTFT 和较小 VRAM 之间需要选择；不能只靠把 chunk 缩小来同时达成。

**下一步验证顺序：** 先在 32GB host RAM 条件下建立 8GiB/6GiB 的完整容量路径；分别测纯 GPU 流送与 BF16 CPU-KV attention；从 4K/16K 增到 128K，逐段记录 TTFT、TPOT、GPU/host 峰值与真实 H2D/D2H。用户要求的 BF16 KV 在这些对照中保持不变。
