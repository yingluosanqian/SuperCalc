# GLM-V5 Baseline 计算方式 — 算子级闭式公式

> **定位**:[AI-infra-Advanced-Calculator](AI-infra-Advanced-Calculator.md) 的第一个验证 case。
> 用**轻量、个人推导的闭式数学公式**(非数据统计、非离散事件模拟)算出一个真实部署的 TTFT/TPOT,
> 再与压测 / profiling 数据对比,验证公式准确性、定位误差来源。
>
> **方法论转向(2026-05-28)**:计算器主体 = 算子级 roofline 闭式公式;profiling 数据降级为「校准 η 系数的数据源」,不进主计算路径;**砍掉 DES 模拟**(8 个输出无一强依赖模拟,仅放弃 P95/P99 尾延迟)。

---

## Case 规格

| 维度 | 取值 |
|---|---|
| 模型 | GLM-V5(MoE) |
| 硬件 | 昇腾 910B2 |
| 部署 | 双机 16 卡,**TP8 · DP2 · EP16**(DP-attention + EP-MoE,DeepSeek V3/R1 同款) |
| 输入长度 | S = 16k |
| 特性 | **零特性**(不开 MTP / chunked prefill / 量化 / prefix cache / 融合算子 / PD 分离…) |
| 目标 | 算 TTFT、TPOT,与压测/profiling 对比 |

**部署语义**:TP8×DP2=16 卡做 attention(TP 切 head,机内 HCCS allreduce);全部 16 卡重组为 EP16 做 routed experts(每卡 host E/16 专家,跨机 RoCE all-to-all);DP2 是两副本跑不同请求,FFN 层打破 DP 边界 16 卡一起 all-to-all。

---

## §0 约定、符号、原子公式

**单位**:一律 SI —— 秒 / 字节 / FLOP。峰值算力 FLOP/s,带宽 B/s。最后才转 ms/GB。

**符号**

**模型结构**

| 符号 | 含义 | 本 case / 备注 |
|---|---|---|
| `L` | Transformer 层数 | 待填;本地 `kv_meta` 中 GLM-5 KV 层数为 78 |
| `d_model` | hidden size / 模型宽度 | 待填 |
| `n_q` | Query head 数 | 待填 |
| `n_kv` | KV head 数 | 仅 GQA/MHA 公式使用;MLA 路径不主用 |
| `d_head` | 单 head 维度 | 待填 |
| `E` | routed expert 总数 | 待填 |
| `k` | 每 token 激活 routed expert 数 | top-k,待填 |
| `d_ffn` | 每个 expert 的 FFN 中间维 | routed/shared expert 使用 |
| `d_ffn_d` | dense FFN 中间维 | 仅 `first_k_dense` 层使用 |
| `n_sh` | shared expert 数 | MoE 层恒定激活 |
| `first_k_dense` | 前若干 dense 层数量 | 其后为 MoE 层 |
| `vocab` | 词表大小 | embedding / lm_head 使用 |

**MLA / KV cache**

| 符号 | 含义 | 本 case / 备注 |
|---|---|---|
| `d_kv_lora` | MLA latent KV 维度 | 本地 `kv_meta`:512 |
| `d_rope` | MLA RoPE 维度 | 本地 `kv_meta`:64 |
| `kv_per_tok_card` | 每 token、每卡 KV cache 字节数 | MLA: `L·(d_kv_lora+d_rope)·b_kv` |

**部署并行**

| 符号 | 含义 | 本 case / 备注 |
|---|---|---|
| `TP` | Tensor Parallel 度 | 8 |
| `DP` | Data Parallel 副本数 | 2 |
| `EP` | Expert Parallel 度 | 16 |
| `world` | 总卡数 | 16 |
| `B_per_dp` | 单个 DP 组内 decode 并发 | 压测口径待定 |
| `DP_moe` | MoE all-to-all 是否合并多个 DP 组 token | 空载单请求取 1;稳态压测常取 2 |
| `B_moe` | MoE 层看到的 token batch | `B_moe = B_per_dp · DP_moe` |

**硬件与效率**

| 符号 | 含义 | 单位 / 备注 |
|---|---|---|
| `C` | 单卡 BF16 峰值算力 | FLOP/s |
| `BW` | 单卡 HBM 峰值带宽 | B/s |
| `α_hccs`, `β_hccs` | 机内 HCCS 通信启动延迟、有效带宽 | 秒、B/s |
| `α_roce`, `β_roce` | 跨机 RoCE 通信启动延迟、有效带宽 | 秒、B/s |
| `η_c` | 算力效率 | 用 profiling 校准 |
| `η_m` | HBM 带宽效率 | 用 profiling 校准 |

**负载与精度**

| 符号 | 含义 | 本 case / 备注 |
|---|---|---|
| `S` | prompt 输入长度 | 16,384 tokens |
| `B` | decode 并发 | 默认指单 DP 组并发;必要时写成 `B_per_dp` |
| `S_ctx` | decode 当前上下文长度 | prompt + 已生成 tokens |
| `N_out` | 输出长度 | 用于 E2E latency |
| `b_w` | 权重 dtype 字节数 | BF16 baseline:2 |
| `b_kv` | KV cache dtype 字节数 | BF16 baseline:2 |
| `b_a` | 激活 dtype 字节数 | BF16 baseline:2 |

**原子公式**

| ID | 计算对象 | 公式 | 说明 |
|---|---|---|---|
| A-1 | 计算时间 | `t_compute = FLOPs / (C · η_c)` | `C` 为单卡峰值算力,`η_c` 用 profiling 校准 |
| A-2 | HBM 访存时间 | `t_mem = bytes / (BW · η_m)` | `BW` 为单卡 HBM 带宽,`η_m` 用 profiling 校准 |
| A-3 | 单算子时间 | `t_op = max(t_compute, t_mem)` | roofline 下界;baseline 默认计算与访存不重叠 |
| A-4 | 集合通信时间 | `t_comm = α + bytes / β` | `α/β` 按 HCCS、RoCE、集合通信类型分别标定 |

**DP 口径**:attention/dense 的单请求延迟按「单 DP 组、单卡视角」算;DP2 主要放大吞吐/并发(×2)。但若 EP 横跨整个 world,MoE 层会看到来自多个 DP 组的 token,则 routed expert GEMM / active expert 并集 / all-to-all 的负载应使用 `B_moe = B_per_dp · DP_moe`。空载单请求校验可取 `DP_moe=1`;生产稳态压测通常要取 `DP_moe=2`。

---

## §1 每卡权重显存

下表先按**每卡元素数**记账,最后统一乘 `b_w` 得字节。

| ID | 权重类别 | 每卡元素数公式 | 并行切分 / 说明 |
|---|---|---|---|
| W-1 | Attention 每层 | `(d_model·(n_q+2·n_kv)·d_head + n_q·d_head·d_model) / TP` | QKV projection + O projection,按 TP 切 |
| W-2 | Dense FFN 每层 | `3·d_model·d_ffn_d / TP` | 仅 `first_k_dense` 层使用;gated FFN 三矩阵 |
| W-3 | Routed experts 每层 | `(E/EP)·3·d_model·d_ffn` | 每卡 host `E/EP` 个 expert |
| W-4 | Shared experts 每层 | `n_sh·3·d_model·d_ffn / TP` | shared expert 按 TP 切 |
| W-5 | Router gate 每层 | `E·d_model` | 通常复制,占比小 |
| W-6 | Embedding + LM head | `vocab·d_model` | 是否乘 1 或 2 取决于 tie / 切分口径 |
| W-7 | Norms | `2·L·d_model` | 极小项 |

| ID | 计算对象 | 公式 | 说明 |
|---|---|---|---|
| W-8 | 每卡权重元素 | `L·W-1 + first_k_dense·W-2 + (L−first_k_dense)·(W-3+W-4+W-5) + W-6 + W-7` | 汇总 attention、dense 层、MoE 层、embedding/head、norm |
| W-9 | 每卡权重字节 | `W_card = W-8 · b_w` | baseline 中 `b_w=2` |

> 关键非对称:attention/dense/shared 按 **/8**(TP),routed experts 按 **/16**(EP)。这是 MoE 部署显存账的核心。

---

## §2 KV cache(每 token,每卡)

| ID | Attention 架构 | 每 token、每卡 KV 字节公式 | 本 case / 说明 |
|---|---|---|---|
| KV-1 | MLA | `kv_per_tok_card = L·(d_kv_lora+d_rope)·b_kv` | GLM-5 本地 `kv_meta.json` 显示为 MLA/DSA,优先用这条 |
| KV-1-gqa | GQA/MHA | `kv_per_tok_card = 2·L·(n_kv/TP)·d_head·b_kv` | 仅在确认 GLM-V5 不是 MLA 时替换 KV-1;`2` 表示 K 和 V |

GLM-5 本地元数据代入:

| 项 | 取值 |
|---|---|
| `L` | 78 |
| `d_kv_lora` | 512 |
| `d_rope` | 64 |
| `b_kv` | 2 |
| `kv_per_tok_card` | `78·(512+64)·2 = 89,856 B/token` |

> 这条直接决定 TPOT 的 KV 读取项([D-4])、最大并发([O-5])、最长上下文([O-6])。**MLA/GQA 的分叉就卡在这一条**。

> **zero-feature 口径提醒**:若"不开任何加速特性"字面包含不开 FlashAttention/SFA,则 prefill attention 的 IO 不能用 flash 的 `O(S·d_model)` 近似,要改成显式 attention score/prob 的 `O(S²·n_q/TP)` 读写。若实际服务框架默认走 fused attention kernel,即使业务未开额外特性,也应按真实 kernel 选择 IO 公式,否则无法和 profiling 对齐。

---

## §2.1 算子 bytes 记账规则

| ID | 计算对象 | 每卡元素数公式 | 用途 |
|---|---|---|---|
| B-1 | QKV projection 权重 | `W_qkv_card = d_model·(n_q+2·n_kv)·d_head / TP` | prefill/decode 读 QKV 权重 |
| B-2 | O projection 权重 | `W_o_card = n_q·d_head·d_model / TP` | prefill/decode 读 O 权重 |
| B-3 | Dense FFN 权重 | `W_dense_ffn_card = 3·d_model·d_ffn_d / TP` | dense 层 decode 读权重 |
| B-4 | Shared expert 权重 | `W_shared_card = n_sh·3·d_model·d_ffn / TP` | MoE 层 shared expert 读权重 |
| B-5 | 激活 routed expert 权重 | `W_expert_card(B_moe) = E_act(B_moe)/EP · 3·d_model·d_ffn` | MoE decode 中随 batch 变化的读权重项 |

> **防二次切分**:[W-1]/[W-4] 已经是每卡元素数;decode 读权重时直接 `W_*_card · b_w`,不要再 `/TP` 或 `/EP`。

---

## §3 TTFT —— prefill,S=16k,计算密集

单请求,token 数 = S。逐层 = (attention block) + (FFN block)。每个算子按 [A-1..4] 取 t_op。

**Attention block(每层)**

| ID | 算子 | FLOPs | bytes / 通信量 | 口径 |
|---|---|---|---|---|
| P-1 | RMSNorm in | 忽略 | `2·S·d_model·b_a` | memory-bound |
| P-2 | QKV projection | `2·S·d_model·(n_q+2·n_kv)·d_head / TP` | `W_qkv_card·b_w` | GEMM,按 A-3 取 max |
| P-3 | RoPE | 忽略 | `≈2·S·(n_q+n_kv)·d_head·b_a` | memory-bound 小算子 |
| P-4 | Causal attention | `2·S²·n_q·d_head / TP` | fused/flash:`O(S·d_model·b_a)`; naive:`O(S²·(n_q/TP)·b_a)` | causal 已含下三角系数;按实际 kernel 选 IO 公式 |
| P-5 | O projection | `2·S·n_q·d_head·d_model / TP` | `W_o_card·b_w` | GEMM,按 A-3 取 max |
| P-6 | Attention all-reduce | - | `2·(TP−1)/TP·S·d_model·b_a` | HCCS,大消息通常 β 主导 |

**FFN block(每层;dense 层走 [P-7d],MoE 层走 [P-8..P-12])**

| ID | 算子 | FLOPs | bytes / 通信量 | 口径 |
|---|---|---|---|---|
| P-7d | Dense FFN | `6·S·d_model·d_ffn_d / TP` | `W_dense_ffn_card·b_w + O(S·d_model·b_a)` | 仅 dense 层;gated FFN 三矩阵 |
| P-8 | Router gate | `2·S·d_model·E` | `E·d_model·b_w + O(S·d_model·b_a)` | MoE 层小项 |
| P-9 | All-to-all dispatch | - | `≈S·k·d_model·b_a·(EP−1)/EP` | RoCE,prefill 大消息;若合批多 DP 组再乘 `DP_moe` |
| P-10 | Routed expert FFN | `(S·k/EP)·6·d_model·d_ffn` | `(E/EP)·3·d_model·d_ffn·b_w + O((S·k/EP)·d_model·b_a)` | 理想均衡下每卡 token-expert 对为 `S·k/EP` |
| P-11 | Shared expert FFN | `6·S·d_model·d_ffn·n_sh / TP` | `W_shared_card·b_w + O(S·d_model·b_a)` | MoE 层恒定激活 |
| P-12 | All-to-all combine | - | `≈S·k·d_model·b_a·(EP−1)/EP` | RoCE,prefill 大消息 |

**组装**

| ID | 计算对象 | 公式 | 说明 |
|---|---|---|---|
| P-13 | 单层 attention 时间 | `t_attn_layer = Σ(P-1..P-6)` | 每个算子先按 A-1 到 A-4 求时间 |
| P-14d | dense 层 FFN 时间 | `t_ffn_dense = P-7d` | 前 `first_k_dense` 层使用 |
| P-14m | MoE 层 FFN 时间 | `t_ffn_moe = Σ(P-8..P-12)` | 其余 MoE 层使用 |
| P-15 | TTFT | `first_k_dense·(P-13+P-14d) + (L−first_k_dense)·(P-13+P-14m) + lm_head` | `lm_head` 仅末 token,近似 `2·vocab·d_model` FLOPs |

> prefill 主导项几乎全在 [P-4](S² attention)和 [P-10](expert GEMM)—— compute-bound。这是 TTFT 主要吃**算力 × η_c** 的原因。

---

## §4 TPOT —— decode,B 并发,单 token/请求,访存密集

每 step 出 B 个 token,上下文 S_ctx。逐层逐算子,几乎全 memory-bound(读权重 + 读 KV)。**TPOT 口径**:这里 [D-14] 是一个 decode step 的 wall time;同一批里的每个请求都等待这个 step,所以 per-request TPOT = step wall time,不是 `step_time/B`。吞吐再用 `B/step_time` 计算。

**Attention block(每层)**

| ID | 算子 | FLOPs | bytes / 通信量 | 口径 |
|---|---|---|---|---|
| D-1 | RMSNorm | 忽略 | `2·B·d_model·b_a` | 小项 |
| D-2 | QKV projection | 小 batch GEMM,通常不是主项 | `W_qkv_card·b_w` | decode 主要按读权重估算 |
| D-3 | RoPE | 忽略 | tiny | 小项 |
| D-4 | Attention decode 读 KV | `2·B·n_q·d_head·S_ctx·2 / TP` | `B·S_ctx·kv_per_tok_card/L` | FLOPs 为 GQA 近似;MLA 需要用真实 attention 结构细化 |
| D-5 | O projection | 小 batch GEMM,通常不是主项 | `W_o_card·b_w` | decode 主要按读权重估算 |
| D-6 | Attention all-reduce | - | `2·(TP−1)/TP·B·d_model·b_a` | HCCS,decode 小消息通常 α 主导 |

**FFN block(MoE 层)**

| ID | 算子 | FLOPs | bytes / 通信量 | 口径 |
|---|---|---|---|---|
| D-7 | Router gate | tiny | tiny | 小项 |
| D-8 | All-to-all dispatch | - | `≈B_moe·k·d_model·b_a·(EP−1)/EP` | RoCE,decode 小消息通常 α 主导 |
| D-9a | 激活专家并集 | - | `E_act(B_moe) = E·(1−(1−k/E)^B_moe)` | MoE TPOT 非线性核心 |
| D-9b | Routed expert FFN 读权重 | 小 batch GEMM,通常不是主项 | `(E_act(B_moe)/EP)·3·d_model·d_ffn·b_w` | MoE decode 主导项之一 |
| D-10 | Shared expert FFN 读权重 | 小 batch GEMM,通常不是主项 | `W_shared_card·b_w` | 每 step 必激活 |
| D-11 | All-to-all combine | - | `≈B_moe·k·d_model·b_a·(EP−1)/EP` | RoCE,decode 小消息通常 α 主导 |

**组装**

| ID | 计算对象 | 公式 | 说明 |
|---|---|---|---|
| D-12 | 单层 attention 时间 | `t_attn_layer = Σ(D-1..D-6)` | 每个算子先按 A-1 到 A-4 求时间 |
| D-13d | dense 层 FFN 时间 | `t_ffn_dense = (W_dense_ffn_card·b_w)/(BW·η_m)` | dense 层 decode 读权重为主 |
| D-13m | MoE 层 FFN 时间 | `t_ffn_moe = Σ(D-7..D-11)` | MoE 层使用 |
| D-14 | TPOT | `first_k_dense·(D-12+D-13d) + (L−first_k_dense)·(D-12+D-13m) + lm_head` | `lm_head` 读 `vocab·d_model·b_w/TP` |

> decode 三大主导项:**读 dense 权重**(QKV/O/shared,固定)+**读激活专家权重 [D-9b]**(随 B 涨,MoE 特有)+**读 KV [D-4]**(随 B·S_ctx 涨)。通信是小消息 → **α 主导**,别用 bytes/β 算。

---

## §5 从上面的量组装 8 个输出

| ID | 输出 | 公式 / 计算方式 | 说明 |
|---|---|---|---|
| O-1 | TTFT | `TTFT = P-15` | 单请求 prefill 到首 token |
| O-2 | TPOT | `TPOT = D-14` | decode 单 step wall time,也是 batch 内 per-request TPOT |
| O-3 | E2E latency | `E2E = TTFT + (N_out−1)·TPOT` | 输出首 token 后还需生成 `N_out−1` 个 token |
| O-4 | HBM 拆解 | `W_card`、`KV = B·S_ctx·KV-1`、激活/workspace、碎片 | 按每卡视角拆解 |
| O-5 | 最大并发 | `B_max(S) = (HBM − W_card − act_reserve) / (S·KV-1)` | 单卡/单 DP 组视角;集群容量再乘 DP |
| O-6 | 最长上下文 | `max_S = (HBM − W_card − act) / (B_target·KV-1)` | 给定目标并发反解 |
| O-7 | 瓶颈分析 | 在 `P-*` / `D-*` 中找最大分量 | 定位到 compute / memory / communication 以及具体算子 |
| O-8 | 推荐部署 | 扫描 `(TP,DP,EP)` 后重算 O-1 到 O-6 | 做 latency / throughput / card count 的 Pareto 筛选 |

---

## §6 内置近似 & 已知误差源(针对性提问/校准用)

| # | 近似 | 偏向 | 后续怎么修 |
|---|---|---|---|
| 1 | 串行求和,忽略计算-通信重叠([A-3] max 只在算子内) | **高估** TTFT/TPOT | 开 multistream/FlashComm 特性时做重叠扣减 |
| 2 | EP 理想负载均衡([P-10][D-9b]) | **低估** all-to-all 和慢卡等待 | 引入 imbalance_factor(L3 标定) |
| 3 | η_c/η_m 用默认值 | 最大不确定性 | 拿 profiling 回灌,逐算子标 η |
| 4 | flash attention bytes 用 O(S·d_model) 近似 | 轻微 | 需要时用精确 IO 公式 |
| 5 | all-to-all 用单段 αβ,没分层(机内+跨机) | decode 通信偏 | hierarchical αβ 两段 |
| 6 | prefill 全 compute-bound / decode 全 memory-bound 假设 | 边界 case 偏 | 一律走 [A-3] max,不预判 |
| 7 | DP 两组对称、all-to-all 合批因子简化 | all-to-all 量偏 | 明确 dispatch payload 是否含两 DP 组 |
| 8 | GLM-V5 MLA attention 的 Q/KV 投影仍用 GQA 近似 | attention 项偏 | 拿真实 config 拆 MLA q/down/up/rope 投影 |
| 9 | "零特性"与框架默认 fused attention/算子融合混用 | TTFT IO/launch 对不上 | 压测前导出实际 env/kernel 开关,选择 naive 或 fused 公式 |

---

## §7 待填 / 待定

**待填参数**(填进 §0 即可代数出数):

- **A. GLM-V5 config** — L / d_model / n_q / n_kv / head_dim / E / k / d_ffn(expert) / n_sh shared / first_k_dense / vocab / dtype / MLA(q_lora_rank,kv_lora_rank,qk_rope_head_dim)
- **B. 910B2 规格** — HBM 容量、HBM 带宽、BF16 TFLOPS、HCCS 带宽、RoCE 带宽

**待定口径**(影响公式选型/对比基准):

1. **PD 分离 / 一体** — 决定 TTFT/TPOT 是否互相拖累
2. **batch B + 压测口径** — TPOT 随 B 变;压测 TTFT 是否含排队/tokenize,要与「纯 GPU 时间」对齐(防假误差)
3. **零特性边界** — 是否真的关闭 fused attention / graph / multistream / MTP / W4A8;若压测环境开了这些,baseline 必须加相应修饰器后才能对比

---

## §8 Profiling 参考与校准入口

参考文件:[GLM-V5-w4a8-dp2tp8ep16-profiling.md](GLM-V5-w4a8-dp2tp8ep16-profiling.md)

**重要口径**:这份 profiling **不是零特性 baseline**。它开启了 `W4A8/W8A8`、`MTP(num=3)`、`chunked prefill(max_num_batched_tokens=32768)`、`CUDAGraph FULL_DECODE_ONLY`;KV cache 仍为 bf16。因此它不能直接验证本文的零特性公式,但非常适合校准真实 GLM-V5 结构、DSA attention、MoE 通信策略和固定开销。

| 类别 | profiling 观测 | 对计算器的用法 |
|---|---|---|
| 模型结构 | `78` 层;`3` 层 DSA+Dense MLP;`75` 层 DSA+MoE;额外 `1` 个 MTP 层 | 回填 `L=78`,`first_k_dense=3`;MoE 层数为 `75` |
| 部署 | `TP=8, DP=2, EP=16, PP=1`,总卡数 `16` | 与本文部署口径一致 |
| 量化 | routed experts 为 `W4A8_DYNAMIC`;attention/dense/shared 为 `W8A8_DYNAMIC` | 不能用 BF16 `b_w=2` 直接对账;需要 feature 修饰器 |
| KV | bf16;本地 MLA KV 为 `89,856 B/token/card` | 可复用 KV 容量公式与 decode KV 读量 |
| Prefill 负载 | 单设备处理 `252,302` tokens;8 个 chunk;总 prefill `108.6s` | 这是 chunked prefill 场景,不是单次 `S=16k` prefill |
| Decode 负载 | MTP decode batch=`4`(`1+3 speculative`);平均 `53.6ms/step` | TPOT 对账必须区分 step time 与有效 token time |

**Prefill 校准点**

| 项 | 实测 | 含义 |
|---|---|---|
| Attention(LI+SFA) | `76.75s`,占 `70.7%` | DSA/MLA 路径下 prefill attention 不能只用普通 flash attention 粗略项 |
| LightningIndexer | `52.67s`,占 `48.5%` | 随 KV 增长线性变慢,chunked prefill 下是第一瓶颈 |
| SparseFlashAttention | `24.07s`,占 `22.2%` | 单次约 `76ms`,累积成为第二瓶颈 |
| 通信 | `20.91s`,占 `19.3%` | Step Trace 显示通信-计算 overlap 为 `0`;当前公式串行求和与 profiling 一致 |
| MoE 通信策略 | prefill 走 AllGather / ReduceScatter | 不能用 decode MC2 dispatch/combine 公式替代 prefill MoE 通信 |

**Decode 校准点**

| 项 | 实测 | 含义 |
|---|---|---|
| 平均 step time | `53.6ms/step` | batch 内 per-request TPOT 是 step wall time;MTP 有效 TPOT 需再除以接受 token 数 |
| 代表 attention 层 | `315us/layer` | 核心 SFA 不是唯一瓶颈,前后辅助小算子链路很重 |
| 代表 MoE 层 | `553us/layer` | MoE 层比 attention 层更重 |
| GroupedMatmul | `13.2ms/step`,占 `24.6%` | routed expert 计算主项,但报告认为利用率已较高 |
| MC2 dispatch+combine | `10.45ms/step`,占 `19.5%` | 小 batch 下主要是固定通信延迟,不是带宽项 |
| MoE Pad+MemSet | `3.90ms/step`,占 `7.3%` | 需要在 decode 公式里作为 fixed overhead 或 calibration residual |

**公式修正方向**

| 位置 | 当前公式状态 | profiling 暴露的问题 | 后续修正 |
|---|---|---|---|
| P-4 | `Causal attention` 仍是普通 FLOPs/IO 近似 | GLM-V5 DSA prefill 中 `LightningIndexer` 是最大项 | 拆成 `LightningIndexer + SparseFlashAttention + cache/update` |
| P-9/P-12 | prefill MoE 通信用 all-to-all 近似 | profiling 显示 prefill 使用 AllGather / ReduceScatter | 给 prefill 和 decode 分开建 MoE comm strategy |
| D-8/D-11 | decode MC2 用 `α + bytes/β` | 小 batch 下 `α` 和 Pad/MemSet 固定成本主导 | 加 `t_mc2_fixed + t_pad_memset` 标定项 |
| D-4 | decode attention 只含 KV 读量 | attention 辅助链路可超过 SFA 本体 | 加 `t_mla_prepost` 或按算子拆 qkv/rope/index/cache |

> 离线约束:目标机器不能联网安装 pip 包。后续计算器和 profiling 对账脚本默认只使用 Python 标准库、Markdown/CSV/JSON 文本输入,不依赖 `numpy/pandas`。

---

## §9 算子粒度推导路径

基于 profiling,GLM-V5 不能再只按「attention + FFN」两块粗算。第一版算子级公式建议按下面的模块树累加:TTFT 算 prefill chunk 序列,TPOT 算 decode step。

### §9.1 先统一三个 batch / length 口径

| 符号 | 含义 | 用在哪里 |
|---|---|---|
| `q_i` | 第 `i` 个 prefill chunk 的 token 数 | TTFT / chunked prefill |
| `K_i` | 第 `i` 个 chunk 可见的累计 KV 长度 | `K_i = prefix_hit + Σ_{j≤i} q_j`;影响 LightningIndexer / attention |
| `B_seq` | decode 并发请求数 | 常规 TPOT |
| `B_tok` | decode 单 step 实际 token batch | 无 MTP 时 `B_tok=B_seq`;MTP 时 `B_tok=B_seq·(1+n_spec)` |
| `B_moe` | MoE 通信看到的 token 数 | `B_tok·DP_moe`,再考虑 pad 到 TP/EP 粒度 |

无 chunked prefill 的 16k baseline 可取 `q_0=S=16384`,`K_0=S`。profiling 文件里的场景则是 8 个 chunk,不能直接等价成单次 16k prefill。

### §9.2 TTFT 应计算的模块

TTFT 是 prefill 阶段到首 token logits 的时间。算子级结构:

| 层类型 | 模块 | profiling 对应算子 | 公式形态 |
|---|---|---|---|
| 所有层 | Norm / Quant | `AddRmsNormBias`,`RmsNorm`,`AddRmsNormDynamicQuant`,`DynamicQuant` | `t_norm(q_i) + t_quant(q_i)` |
| 所有层 | DSA/MLA projection | `QuantBatchMatmulV3`,`MatMulV2`,`BatchMatMulV2` | `max(FLOPs/(C·η_c), bytes/(BW·η_m))` |
| 所有层 | RoPE / KV 写入 | `KvRmsNormRopeCache`,`_triton_rope`,`ScatterNdUpdate` | `t_rope_cache(q_i)` |
| 所有层 | LightningIndexer | `LightningIndexer` | `t_LI(q_i,K_i)`;profiling 显示近似随 `K_i` 线性增长 |
| 所有层 | Sparse attention | `SparseFlashAttention` | `t_SFA(q_i,topk)`;profiling 中单次相对稳定 |
| 所有层 | O projection + TP comm | `QuantBatchMatmulV3`,`hcom_allReduce_` | `t_o_proj(q_i)+t_allreduce(q_i)` |
| dense 层(3层) | Dense MLP | `QuantBatchMatmulV3`,`SwiGLU` 等 | `t_dense_mlp_prefill(q_i)` |
| MoE 层(75层) | Router / top-k | `MoeInitRoutingCustom`,`MoeGatingTopK`,`MoeTokenUnpermute` | `t_router(q_i)` |
| MoE 层(75层) | Routed expert compute | `GroupedMatmul` | `t_grouped_gemm_prefill(q_i,k,EP)` |
| MoE 层(75层) | Shared expert compute | `QuantBatchMatmulV3`,`SwiGLU` | `t_shared_prefill(q_i)` |
| MoE 层(75层) | Prefill MoE comm | `hcom_reduceScatter_`,`hcom_allGather_` | `t_rs(q_i)+t_ag(q_i)`;注意不是 decode MC2 |
| 末尾 | LM head / sample first token | `lm_head` 相关 GEMM/通信 | 只在最后一个 chunk 末尾算 |

TTFT 组装:

| ID | 公式 | 说明 |
|---|---|---|
| T-1 | `t_dsa_prefill(i)=t_norm+t_proj+t_rope_cache+t_LI(q_i,K_i)+t_SFA(q_i)+t_o_proj+t_attn_comm` | 每层 DSA/attention 路径 |
| T-2 | `t_dense_layer_prefill(i)=t_dsa_prefill(i)+t_dense_mlp_prefill(i)` | 前 3 层 |
| T-3 | `t_moe_layer_prefill(i)=t_dsa_prefill(i)+t_router(i)+t_grouped_gemm(i)+t_shared(i)+t_moe_comm_prefill(i)+t_misc_moe(i)` | 后 75 层 |
| T-4 | `TTFT = Σ_i [3·T-2(i)+75·T-3(i)] + t_lm_head_last + t_sample + t_host_gap` | chunked prefill 对所有 chunk 求和;无 chunk 时只有 `i=0` |

第一版可以先把 profiling 里的大项拆成校准函数:

| 校准函数 | 初始来源 |
|---|---|
| `t_LI(q,K)` | 用 profiling 中 `Chunk 0/2/4/6` 的 LightningIndexer 耗时拟合 `q·K` |
| `t_SFA(q)` | 用 `SparseFlashAttention` 平均耗时按 `q` 缩放 |
| `t_moe_comm_prefill(q)` | 用 `hcom_reduceScatter_`、`hcom_allGather_` 的平均耗时做 αβ 或查表 |
| `t_host_gap` | 用 Step Trace 的 `Free` 时间按阶段分摊 |

### §9.3 TPOT 应计算的模块

TPOT 是一个 decode step 的 wall time。profiling 中 `53.6ms/step` 是 step time;若启用 MTP,有效每 token 时间再除以 `1+n_spec·accept_rate`。

| 层类型 | 模块 | profiling 对应算子 | 公式形态 |
|---|---|---|---|
| 所有层 | Decode norm / quant | `AddRmsNorm`,`DynamicQuant` | `t_norm_decode(B_tok)` |
| 所有层 | MLA 前后处理链路 | `qkv_a_proj`,`q_b_proj`,`q_nope_proj`,`kv_b_proj`,`wk_proj`,`weights_proj`,`Transpose`,`ScatterNdUpdate` | `t_mla_prepost(B_tok,S_ctx)` |
| 所有层 | LightningIndexer + SFA | `LightningIndexer`,`SparseFlashAttention` | `t_LI_decode(B_tok,S_ctx)+t_SFA_decode(B_tok,S_ctx)` |
| 所有层 | O projection + TP comm | `QuantBatchMatmulV3`,`hcom_allReduce_` | `t_o_proj_decode(B_tok)+t_attn_comm_decode(B_tok)` |
| dense 层(3层) | Dense MLP | `QuantBatchMatmulV3`,`SwiGLU` | `t_dense_mlp_decode(B_tok)` |
| MoE 层(75层) | Router / top-k | `MatMulV2`,`MoeGatingTopK` | `t_router_decode(B_tok)` |
| MoE 层(75层) | MC2 dispatch | `MoeDistributeDispatchV2` | `t_mc2_dispatch = α_mc2_dispatch + bytes/β_mc2` |
| MoE 层(75层) | Pad / buffer fixed cost | `PadV3`,`MemSet` | `t_pad_memset(B_tok)`;小 batch 下必须显式建模 |
| MoE 层(75层) | Routed expert compute | `GroupedMatmul` | `t_grouped_gemm_decode(B_tok,E_act)` |
| MoE 层(75层) | Shared expert compute | `QuantBatchMatmulV3` | `t_shared_decode(B_tok)` |
| MoE 层(75层) | MC2 combine | `MoeDistributeCombineV2` | `t_mc2_combine = α_mc2_combine + bytes/β_mc2` |
| 末尾 | LM head / sampling / graph overhead | LM head、采样、CUDAGraph replay | `t_lm_head + t_sample + t_graph_gap` |

TPOT 组装:

| ID | 公式 | 说明 |
|---|---|---|
| DSA-1 | `t_dsa_decode=t_norm+t_mla_prepost+t_LI_decode+t_SFA_decode+t_o_proj+t_attn_comm` | profiling 代表层约 `315us/layer` |
| MLP-1 | `t_dense_layer_decode=t_dsa_decode+t_dense_mlp_decode` | 3 层 |
| MOE-1 | `t_moe_layer_decode=t_dsa_decode+t_router+t_mc2_dispatch+t_pad_memset+t_grouped_gemm+t_shared+t_mc2_combine+t_result_add` | profiling 代表层约 `553us/layer` |
| TPOT-1 | `t_step = 3·MLP-1 + 75·MOE-1 + t_lm_head + t_sample + t_graph_gap` | batch 内 per-request TPOT 等于 `t_step` |
| TPOT-2 | `TPOT_effective = t_step / (1+n_spec·accept_rate)` | 仅 MTP 场景;无 MTP 时分母为 1 |

profiling 给出的 decode 初始校准值:

| 校准项 | profiling 值 | 放入公式的位置 |
|---|---|---|
| representative attention layer | `315us/layer` | `DSA-1` 总校准 |
| representative MoE layer | `553us/layer` | `MOE-1` 总校准 |
| `GroupedMatmul` | `13.2ms/step` | `t_grouped_gemm_decode` |
| MC2 dispatch+combine | `10.45ms/step` | `t_mc2_dispatch+t_mc2_combine` |
| Pad+MemSet | `3.90ms/step` | `t_pad_memset` |
| LI+SFA decode | `4.60ms/step` | `t_LI_decode+t_SFA_decode` |

**最小可实现版本**:先不拟合所有小算子,而是把 TTFT/TPOT 分成 `DSA attention`、`dense MLP`、`MoE compute`、`MoE comm`、`fixed overhead` 五个桶。等能对上 profiling 总量后,再把 DSA 桶拆成 `LightningIndexer / SFA / prepost`。

---

> **状态**:符号公式骨架(2026-05-29)。已接入 w4a8 profiling 作为校准样本,但它不是零特性 baseline。下一步:填入 GLM-V5 config + 910B2 规格,并把 §8 暴露的 DSA/MC2/chunked prefill 修正项落回 §3/§4。
