# FlashRT → Jetson-PI：RTX 4090 首个 kernel 移植

## 当前状态

- Jetson-PI 基线：`1271fa933c3f81e1cb3a71a9f09f13bdefa4461b`（master）。
- FlashRT 源码：`e881e1424cf4fadee7dff121d01cc4e159e550e7`（main）。
- 工作分支：`flashrt-sm89-layernorm`。
- 已实现：可选的 SM89 `NORM → MUL → ADD` 融合，适配 FlashRT LayerNorm 算法到 ggml FP32。
- 目标模型：先以 pi0 的 SigLIP 区域建立移植与验证流程；没有完成整个 pi0/整个 FlashRT 的移植。
- 当前环境没有 nvcc 和 GPU。CUDA 编译、4090 正确性、graph capture、真实模型 E2E 和性能结论均待验证。
- 不应把 FlashRT 原项目的 latency 或之前跑通 FlashRT 的结果视为这个补丁的加速证据。

本地检查结果：GCC 13.3、CPU Release 构建通过；新增 `FLASHRT_LAYERNORM`
图测试 12/12 通过；独立 benchmark 的 4 个 shape 均通过 double 参考检查，
最大绝对误差范围约 `2.49e-7..4.25e-7`。这些结果验证测试图与参考程序，
不验证被编译开关排除的 CUDA kernel。CUDA/GPU 状态仍为待验证。

## 代码审阅结论

Jetson-PI 的模型图仍由 `src/models/pi0.cpp`、`src/models/pi0_ae.cpp`
和 `tools/mtmd/` 构造。CUDA 分派在
`ggml/src/ggml-cuda/ggml-cuda.cu`。因此第一阶段保留 GGUF、ggml graph、
allocator、scheduler、KV cache 和 server API，只替换满足条件的 CUDA 计算区域。

“ggml 通用，因此一定慢”不能作为性能结论。当前代码已经包含
FlashAttention、RMSNorm affine 融合、部分 FFN/GEMV 融合和 CUDA graphs。
应该用这些已有优化开启时的基线进行比较。

| 区域 | FlashRT 来源 / 边界 | Jetson-PI 接口 / 差异 | 迁移状态 |
|---|---|---|---|
| SigLIP LayerNorm affine | `csrc/kernels/norm.cu::layer_norm_kernel`；FP16/BF16 `[rows, dim]` + 通道权重/偏置 | `tools/mtmd/clip.cpp::clip_graph::build_norm`；`NORM → MUL → ADD`；当前 ggml CUDA norm 为 FP32，SigLIP width=1152 | 首个已写入的候选，待 GPU 验证 |
| Vision/QKV/FFN GEMM | `flash_rt/models/pi0/pipeline_rtx.py::_fp8_gemm` / `_fp8_gemm_fused`；`fp8_gemm_descale_fp16` | ggml `MUL_MAT`；F16/Q8_0/Q4_K 等 GGUF 不能当作 E4M3 指针使用 | 下一阶段；需权重预打包、scale、workspace 与实际 shape profile |
| Encoder RMSNorm / residual | `pipeline_rtx.py::_encoder_layer`；`rms_norm_fp8_noweight_fp16` 等；FP8 量化可跨算子融合 | `src/llama-graph.cpp::build_norm_pi0` 实现 `norm(x)*w + norm(x)`；必须处理 `1+w` 及 FlashRT 的权重折叠 | 保留基线；不能简单替换普通 weighted RMSNorm |
| Decoder 小 M GEMM / GEGLU | `pipeline_rtx.py::_decoder_layer`；`gate_geglu_merged_fp16` / FP8 版本 | `src/models/pi0_ae.cpp`、`build_ffn`、`ggml_geglu_split`；M 由动作序列配置决定 | profile 后排优先级；分开核对 GELU 公式与 gate/up 顺序 |
| Attention / RoPE / QKV | `csrc/attention/fa2_wrapper.cu`、`csrc/kernels/rope.cu`；RTX FA2 路径 | ggml 已有 attention；Q/K/V 的 ne/nb、GQA、KV cache、mask/position 语义需要对齐 | 保留现有实现，先测 attention 与 layout 转换成本 |
| 图复用 / 缓存 / 预处理 | FlashRT pipeline 的固定 buffer 与图执行 | Jetson-PI 已有 ViT 图复用、decode 优化；见 `latency_optimize_docs/` | 保留宿主状态和生命周期；单独统计 host 开销 |

以上是源码区域映射，不是 nsys 热点排序。4090 第一轮 profile 后应按实际耗时调整后续顺序。
SM100/SM120 的 NVFP4、TMA/架构专用 GEMM 不在本次 SM89 移植范围。

## 首个补丁的契约

计算公式：`y = ((x - mean(x)) / sqrt(var(x) + eps)) * weight + bias`。

- 使用 FlashRT 的 256-thread block、packed-pair IO、两遍方差与 affine 融合。
- FP16/BF16 改为 FP32/float2，避免在 ggml 每层新增 cast。不是原 FP16 二进制直接调用。
- 编译开关 `GGML_CUDA_FLASHRT=ON`；运行开关 `GGML_CUDA_FLASHRT=1`。
- 只对 SM89、连续 FP32、偶数宽度 `2..8192`、正且有限的 eps、通道向量 affine 生效。
- `[dim, rows, batch, ...]` 连续输入可以按行展平；行数须可由 CUDA grid.x 表示。
- 8-byte 对齐；输出不得与 x/weight/bias 重叠。未参与 fusion 的节点不被修改。
- 使用 ggml 原有的 consumer/output 检查，norm/mul 有外部消费者时不消除中间结果。
- 不匹配时执行原 ggml CUDA 路径，不迁移到 CPU；其他平台与默认配置不变。
- 直接使用 `ctx.stream()`；kernel 路径不分配显存、不改 stream、不做设备同步。
- `GGML_CUDA_DISABLE_FUSION=1` 同样关闭本次融合。
- 运行开关按进程初始化；A/B 用独立进程，不能在已 capture 的 graph 上修改环境变量切换。

FlashRT 的两遍方差与原 ggml CUDA 的 `E[x²] - E[x]²` 不保证位级一致。
本补丁保留 affine 部分的分步 FP32 舍入；仍需参考误差和真实 action 误差双重验证。

## 4090 编译

从已应用补丁的仓库根目录运行；CUDA 12.4 环境可以作为初始验证环境。

```bash
cmake -S . -B build-flashrt-sm89 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_CUDA_FLASHRT=ON \
  -DLLAMA_BUILD_TESTS=ON

cmake --build build-flashrt-sm89 -j 8 \
  --target test-backend-ops test-flashrt-layernorm-bench llama-server
```

`flashrt-layernorm.cu` 由现有 CUDA source glob 编译；不需要安装 PyTorch/FlashRT Python 包。

## 正确性与实际命中

选择目标 4090 的物理编号；`CUDA_VISIBLE_DEVICES` 生效后测试程序使用逻辑 `CUDA0`。

```bash
export CUDA_VISIBLE_DEVICES=7  # 按机器实际编号修改
export GGML_CUDA_DISABLE_FUSION=0
mkdir -p results/flashrt-sm89

GGML_CUDA_FLASHRT=0 build-flashrt-sm89/bin/test-backend-ops test \
  -b CUDA0 -o FLASHRT_LAYERNORM \
  > results/flashrt-sm89/baseline-test.log 2>&1

GGML_CUDA_FLASHRT=1 GGML_CUDA_FLASHRT_TRACE=1 \
  build-flashrt-sm89/bin/test-backend-ops test \
  -b CUDA0 -o FLASHRT_LAYERNORM \
  > results/flashrt-sm89/candidate-test.log 2>&1

rg 'flashrt_layernorm: enqueued|FAIL|OK|passed' results/flashrt-sm89/candidate-test.log
```

新增 12 个图测试：4 个真实宽度/不同 row 数，非连续输入、norm 额外消费者、
逐行 affine、未对齐权重、eps=0、3D batch、奇数宽度、宽度越界。
正常连续案例应出现 enqueue；负例应走原路径且输出正确。trace 记录的是
host enqueue/capture，并不逐次统计 CUDA graph replay；profile 应进一步确认 kernel。

独立 benchmark 还有确定性输入和 double-precision 参考，逐元素检查
`abs_error <= 5e-5 + 5e-5 * abs(reference)`，拒绝非有限输出。这个阈值只用于
本地算子门槛，不能代替模型级精度评估。

## A/B 性能

```bash
for round in 1 2 3; do
  for mode in 0 1 1 0; do
    GGML_CUDA_FLASHRT=$mode GGML_CUDA_FLASHRT_TRACE=0 \
      build-flashrt-sm89/bin/test-flashrt-layernorm-bench CUDA0 \
      >> results/flashrt-sm89/bench-mode-$mode.csv
  done
done
```

每个 shape 100 次 warmup + 200 次采样，报告完整三节点图的同步 wall-clock P50/P95，
包含 backend dispatch/同步成本；不是纯 kernel CUDA-event 时间，也不是模型 E2E。
不要用 `test-backend-ops perf` 的单末节点复制逻辑替代整个融合区域的定时。
需要再用 nsys 查看 `flashrt_layer_norm_affine_f32`、baseline 三节点的 GPU 时间和 launch 数，
并在相同 CUDA graph 设置下比较。

真实模型继续沿用已经跑通的 server/client 命令，仅把 server 二进制改为
`build-flashrt-sm89/bin/llama-server`，分进程设 `GGML_CUDA_FLASHRT=0/1`。
复用相同 GGUF、mmproj、image/state/prompt、固定 action noise、相同步数和相机数量。
记录 encoder / action expert / total latency、action 的 max-abs / RMSE / cosine，
同时确认真实 ViT 图命中 kernel。不要在 debug dump/trace 开启时报告性能。

现有 `scripts/run_pi0_4090.sh` 自带 CUDA device、模型路径和构建目录假设，
运行前核对，尤其不要把其他卡编号误认为逻辑 CUDA0。

## 后续最值得做的步骤

1. 先在 4090 完成这个补丁的编译、正负例、捕获重放、模型精度和 profile。
2. 从 profile 抽取最耗时 GEMM 的真实 M/N/K、dtype、stride、每步调用次数。
3. 若 FP8 GEMM 值得做，优先从未量化权重建立一次性 E4M3 预打包与 scale 生命周期，
   复用 ggml stream/workspace，避免每次 inference 重打包。量化 GGUF 路径保持单独分派。
4. 再扩展 norm/activation→FP8 与 GEMM 的区域融合；用真实输入校准，独立评估精度损失。
5. attention 只有在同 shape、同 mask、同 cache 语义下超过现有 ggml 后端才替换。

首个 LayerNorm 的收益受它在整次推理中所占时间限制；减少两次 launch 不等于整模型已有明显加速。
