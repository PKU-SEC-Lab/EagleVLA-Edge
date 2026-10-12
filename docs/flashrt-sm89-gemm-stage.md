# RTX 4090：pi0 短序列 GEMM 适配与验收

宿主固定提交：`PKU-SEC-Lab/Jetson-PI-Edge@1271fa933c3f81e1cb3a71a9f09f13bdefa4461b`。
FlashRT 参考：`flashrt-project/FlashRT@e881e1424cf4fadee7dff121d01cc4e159e550e7`。
本增量接续已验收的 GGML-math LayerNorm、Gemma RMSNorm 和宿主 encoder KV 生命周期修复。
两个 GEMM 候选默认关闭，完整 SM89/pi0 kernel 移植仍未完成。

## 已有 GPU 证据与此次目标

用户提供的十次请求 CUDA kernel 统计中，
`mul_mat_f<__half2,32,11,8,false>` 共 12,600 次，占已记录 kernel 累计时间 31.69%。
其次是 `ampere_h1688gemm_256x64_ldg8_stages_32x1_tn`，630 次，占 18.80%。
这些是 kernel 累计时间占比，包含可能重叠的执行时间，不是端到端耗时占比。
本阶段只替换第一项对应的非 MoE、小 token 数 MMF 分派。

根据宿主 action-expert 图和用户 GGUF 权重清单，下面五种形状与调用数吻合。
这里的节点映射是源码推断；脚本必须通过运行时 TRACE 确认实际命中。

| M：token | K：输入维度 | N：输出维度 | 推定位置 | 十次请求推定调用次数 |
|---:|---:|---:|---|---:|
| 11 | 1024 | 2048 | Q 投影 | 1800 |
| 11 | 1024 | 256 | K、V 投影 | 3600 |
| 11 | 2048 | 1024 | O 投影 | 1800 |
| 11 | 1024 | 4096 | FFN gate、up | 3600 |
| 11 | 4096 | 1024 | FFN down | 1800 |

上述次数以 18 个 action-expert 层、10 个 denoising step 为前提：18×7×10×10=12,600。
TRACE 是主机 enqueue 记录；CUDA Graph 重放不会重新打印这些 enqueue，不能据 TRACE 条数统计 GPU 调用数。

## 两个候选的真实来源和数学

| 运行时 GEMM 模式 | 实现 | 数值边界 | 当前用途 |
|---|---|---|---|
| `0` 或不设置 | 原 GGML MMF | 原 32 行 CTA，8 份 K 部分和 | 对照 |
| `1` | 参考 FlashRT `GemmRunner::fp16_nn` 设计的 cuBLASLt 适配 | F16 权重；F32 激活按 RN 转 F16；FP32 累加和输出；归约次序可能变化 | 独立数值、Graph、速度诊断，不自动接入模型 |
| `2` | GGML-math MMA16 适配 | 每 CTA 改为 16 输出行，保留原 8 份 K 分组、MMA 及顺序 FP32 归约 | 要求实测逐位一致后再跑模型对照 |

FlashRT 的 `fp16_nn` 本身是 cuBLASLt 包装，输出为 F16，权重描述为 row-major K×N。
GGML 此处的权重是连续 `W[n*K+k]`，输出为 `D[m*N+n]` F32。
mode 1 使用 column-major A(K,N)、op(A)=T，B(K,M)、op(B)=N，D(N,M)，无需重排权重。
保持 F32 输出是明确的宿主接口适配，不能称为原接口逐位移植。
原 GGML MMF 已将 F32 激活转换为 F16 后使用 Tensor Core；mode 1 没有新增加这一步输入舍入。

mode 2 来自 MIT 许可的 GGML MMF 模板，属于本移植实验中的硬件适配候选，
不能把它记成 FlashRT 原生 GEMM 已迁入。
16 行 CTA 会增加部分形状的并行 CTA 数，也会增加激活加载次数；是否加速必须逐形状测量。

## 接口、回退与资源

- 编译开关沿用 `GGML_CUDA_FLASHRT`；运行时还需 `GGML_CUDA_FLASHRT=1`。
- `GGML_CUDA_FLASHRT_GEMM=1/2` 只对 SM89、以上五种形状、M=11、连续二维、F16/F32/F32、
  默认精度、同设备 CUDA buffer、满足对齐且无输入输出重叠的 MUL_MAT 生效。
- 入口放在原 MMF 条件分支内；其他 token 数、dtype、GPU、布局、显式高精度或不符合契约的张量保留宿主分派。
- mode 1 还要求权重不是 view，buffer 用途是 WEIGHTS；不复用全局 mutable handle 或全局权重缓存。
- mode 1 每 context/stream 自有 Lt handle、32 MiB workspace、90 KiB 激活 scratch 和五个 shape 计划。
  计划与地址在非 capture 阶段建立；每个 shape 必须先执行一次才能在 capture 中使用。
- 如果首次见到某 shape 已在 capture 内，会回退；这个已捕获实例可能持续使用原 GGML。
  因此验收同时检查实际候选 enqueue 和 capture 命中，不能仅检查环境变量。
- context 析构持宿主 capture 排他锁，同步后先释放上述资源，再销毁 streams。
  运行期 Lt/CUDA 错误直接报告；不能写了一部分输出后静默回退。
- `GGML_CUDA_FLASHRT_GEMM_GRAPH_TRACE=1` 仅为单 MUL_MAT 测试图记录成功的 `cudaGraphLaunch`，
  用于核实重放。它不是模型图 profiler，性能测试必须关闭。

## 一次执行的验收流程

`scripts/run-sm89-gemm-stage.py` 编译、检查主机契约，在物理 GPU 2 上运行独立进程微测试。
每种形状包含普通数据、相消数据和有界大动态范围数据。
保存原始 F32 输出、FP64 点积诊断、所有计时样本和日志。
FP64 参考使用实际舍入后的 F16 操作数；诊断误差界只排查明显布局/计算错误，不能替代模式间一致性验收。

mode 2 必须同时满足：

1. 五种形状实际命中新入口；15 个输入/形状组合均有限。
2. Graphs 关闭和允许两种设置都与原 GGML 输出逐位一致；Graph 重放不改变输出。
3. Graph 检查日志证实候选被捕获，并有足够实际 `cudaGraphLaunch`。
4. 关闭 TRACE 和 shadow，用独立进程 ABBA 测量；微测试指标是一次后端图执行并同步的墙钟时间，不能叫纯 kernel 时间。

mode 1 单独记录数值偏移和 ABBA 速度；自身 Graph 开/关应一致，不能自动获得动作误差豁免。
它的诊断失败与 mode 2 的门槛分别记录。

微测试通过后，`scripts/run-sm89-gemm-e2e.py` 验证 pi0 模型实际输出传播。
A 是已验收的 LayerNorm + RMSNorm 组合、GEMM=0；B 只增加 GEMM=2。
相同固定图像、固定噪声下切换提示词和状态，检查 Graphs 关闭/允许时的动作一致性，
然后关闭诊断做 ABBA。这个 A/B 边界隔离新增 GEMM 的影响。
脚本只管理自己启动的服务；18081 已被占用时报告忙并保留结果。

任何失败均保存实际阶段结果；只有已完成的阶段可以写成通过。
源码提交、二进制指纹和原始记录随报告保存。没有 CUDA GPU 的本地编译不能代替 4090 验收。

## 本地交付检查（2026-09-10）

- CUDA 13.0.88、GCC 13.3、SM89：两个新 CUDA 文件编译通过。
  MMA16 使用 64 个寄存器，无 spill、无 stack；PTX 检查保留顺序 FP32 归约。
- 完整 CUDA 后端、`llama-server`、微测试和主机 guard 程序已成功编译链接。
  本地无 GPU/驱动，链接检查使用 `GGML_CUDA_NO_VMM=ON`；服务器保留其既有 VMM 配置。
- 使用真实 GGML 源码执行 49 项张量契约检查通过。
- 微测试 C++ 严格语法、FP64 参考函数、JSON/原始输出协议检查通过；
  runner 的失败隔离、capture 证据和进程清理经过主机模拟测试。
- 增量补丁在包含旧归一化改动的模拟工作树上，通过完整 apply、reverse-check、reverse，旧文件内容恢复一致。
- 本地没有执行 4090 数值、CUDA Graph 或 pi0 模型测试；这些记录由终端流程实际生成。

## 完整移植剩余范围

已完成的 LayerNorm + RMSNorm GPU 验收范围仍是一张固定图像的三组输入。
先前归一化组合 ABBA 的 Infer HTTP P50 从 84.339 ms 到 79.549 ms，降低 5.68%；
这是组合结果，不能归因于本阶段 GEMM，也不能与本阶段微测试相加估算总收益。

FP8 权重打包、激活量化与 scale 生命周期、FP8 GEMM、残差/激活到 FP8 的融合、
patch、QKV/RoPE/KV 写入、attention、action 更新和全组合验证仍需按
`flashrt-sm89-coverage.md` 逐项推进。现有 GGML 回退不计作这些区域已完成 FlashRT 移植。
FP8 路径会改变数学精度，仍需代表性真实输入与明确的动作误差/任务成功率验收，不能用一张合成输入放宽标准。

参考源码：

- [FlashRT GemmRunner](https://github.com/flashrt-project/FlashRT/blob/e881e1424cf4fadee7dff121d01cc4e159e550e7/csrc/gemm/gemm_runner.cu)
- [宿主 MMF](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge/blob/1271fa933c3f81e1cb3a71a9f09f13bdefa4461b/ggml/src/ggml-cuda/mmf.cuh)
- [宿主 pi0 action-expert 图](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge/blob/1271fa933c3f81e1cb3a71a9f09f13bdefa4461b/src/models/pi0_ae.cpp)
