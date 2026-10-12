# RTX 4090 / pi0 完整移植范围与验收

范围：FlashRT `e881e1424cf4fadee7dff121d01cc4e159e550e7` 的
`flash_rt/models/pi0/pipeline_rtx.py::Pi0Pipeline`，接入 Jetson-PI
`1271fa933c3f81e1cb3a71a9f09f13bdefa4461b` 的 GGUF / ggml CUDA 后端。
保留宿主的输入预处理、状态、tokenizer、KV cache、动作步进和 API 边界。
一天是推进目标；只有下列所有区域有明确实现与 GPU 验收记录，才能称为完整移植。
源码覆盖表不代表性能热点排序，也不代表这些 kernel 已在服务器执行。

## 全部 native 区域

下表覆盖 pipeline 中所有直接 `fvk`、`gemm`、attention 调用及跨算子融合。
原 pipeline 的常规激活为 FP16，大矩阵乘法使用 E4M3；当前宿主 norm 激活为 F32。
符号名中的历史 `bf16` 命名不能作为真实 dtype 证据。

| 区域 | FlashRT 文件 / 符号 | 形状与边界 | ggml 对应 / 必要适配 | 当前状态 |
|---|---|---|---|---|
| 图像 patch | `csrc/kernels/patch_embed.cu::patch_im2col` | 每视角 224×224×3 → 256×588；FP16 | `IM2COL` + patch projection；核对布局和预处理 | 待移植与 profile |
| Patch / projector / 常规 GEMM | `gemm.fp16_nn` | vision D=1152，projector 1152→2048 | `MUL_MAT`；权重布局、输出 dtype、算法与 workspace | 待移植与 profile |
| ViT LayerNorm affine | `norm.cu::layer_norm_fp16` | 27×2+1 层，D=1152 | `NORM→MUL→ADD`；从宿主读取 eps | GGML-math 适配通过固定输入验证，ViT P50 -2.91%；原 FlashRT 数学路径最终动作偏移未验收 |
| Bias / residual | `elementwise.cu::bias_residual_fp16`, `residual_add_fp16`; `decoder_fused.cu::add_bias_fp16` | 通道 bias 广播、残差逐元素 | `ADD`；已有部分多 ADD 融合，需核对实际命中与舍入 | 已映射，待 profile 后逐区域处理 |
| ViT 激活 | `activation.cu::gelu_inplace_fp16` | D=4304，27 层 | `UNARY(GELU)`；核对 erf/tanh 版本、先后舍入 | 待移植与 profile |
| ViT QKV split | `rope.cu::qkv_split_fp16` | Q/K/V 各 16 heads ×72 | `VIEW/RESHAPE/PERMUTE/CONT`；不可破坏相机边界 | 待移植与 profile |
| ViT attention | `attn.run("siglip")` | 每视角 S=256，head_dim=72 | 已启用 `FLASH_ATTN_EXT`；比较相同 mask/layout 的后端 | 现用宿主后端，未完成 FlashRT attention 接入 |
| Gemma RMSNorm | `norm.cu::rms_norm_fp16` | encoder D=2048；decoder D=1024 | 宿主为 `n=rms_norm(x); n*w+n`；FlashRT 将 `1+w` 折入下游权重 | FP32 / GGML-math 融合已实现并通过 SM89 编译；GPU 数值与性能待验收；非原生 FP8 路径 |
| 动态 / 静态 FP8 量化 | `quantize_fp8_device_fp16`, `quantize_fp8_static_fp16` | FP16 激活→E4M3，设备端 scale | ggml 无等价的现成 E4M3 张量契约；需要受管理的辅助 buffer 和 scale 生命周期 | 待实现；缺真实输入校准集 |
| FP8 GEMM / descale | `decoder_fused.cu::fp8_gemm_descale_fp16` | ViT/encoder/decoder 的 QKV、O、gate/up、down | 不可把 F16/Q8_0/Q4_K 指针当成 E4M3；需要一次性权重预打包、矩阵布局、每设备计划与 workspace | 待实现与 GPU 验证 |
| RMSNorm→FP8 | `norm.cu::rms_norm_fp8_noweight_fp16` | encoder/decoder 输入；输出无 weight 的 norm 量化值 | 必须与已折叠 `1+w` 的矩阵权重配套 | 依赖 FP8 权重与 scale 接入 |
| Residual→RMSNorm→FP8 | `norm.cu::residual_add_rms_norm_fp8_noweight_fp16` | 双输出：更新 residual + 量化 norm | 保留残差外部消费者；不能消除必须写回的值 | 依赖 FP8 接入及图内生命周期验证 |
| GEGLU | `activation.cu::gate_geglu_fp16`, `gate_geglu_merged_fp16` | encoder H=16384，decoder H=4096 | `GLU(GEGLU)`；核对 gate/up 顺序及 GELU 公式 | 待移植与 profile；不能沿用误写为 SiLU 的源码注释 |
| GEGLU→FP8 | `activation.cu::gate_geglu_merged_fp8_fp16` | 合并 gate/up → H 个 E4M3 激活 | 依赖合并权重布局与 down GEMM scale；不能只替换激活算子 | 依赖 FP8 接入 |
| QKV→RoPE→KV cache | `rope.cu::qkv_split_rope_kvcache_fp16` | encoder/decoder：8 Q heads、1 KV head、HD=256 | `ROPE`、cache 写入、views；核对 position、stride、每层偏移与缓存覆盖 | 待移植与 profile |
| Encoder attention | `attn.run("encoder")` | S=视觉 tokens+文本 tokens；最后一层只生成 KV | 宿主非因果 encoder 与现有 attention | 现用宿主后端，未完成 FlashRT attention 接入 |
| Decoder attention | `attn.run("decoder", state_nk=...)` | S_dec=动作数+1；state 行仅看 prefix+自身 | 保留 state mask、GQA、prefix KV 与动作 token 范围 | 现用宿主后端，未完成 FlashRT attention 接入 |
| 状态 / 动作 / 时间 MLP | `gmm_fp16`, `fused_add_silu_fp16`, `add_bias_fp16` | state 32→1024；动作+时间 MLP；每去噪步 | `MUL_MAT/ADD/CONCAT/SILU`；预计算 time 部分须与宿主 timestep 对齐 | 待移植与 profile |
| 最终 norm / 动作投影 | `rms_norm_fp16`, `gmm_fp16(beta=1)`, `add_bias_fp16` | D=1024→32；只输出动作行；权重乘 -1/步数 | 宿主 scale/update 次序和 FP32 舍入不能被假定相同 | 待移植与误差验证 |
| 运行状态与 CUDA Graphs | `CudaBuffer`, `CUDAGraph`, `record_infer_graph`, `set_language_embeds` | buffer、观测更新、prompt 长度、capture | 使用 ggml allocator/stream/graph 生命周期；禁止 capture 内初始化计划或重打包权重 | LayerNorm 路径已验证；其余新增区域仍需独立验证 |

范围之外：SM100/SM120 的 FP4/TMA 专用代码、其他模型和真机控制；这些不属于
本次明确的 SM89/pi0 native 路径。整套 FlashRT 的所有硬件 kernel 不等同于这条路径。

## 必须保留的验收

1. 每个新增分派：正例实际命中；布局、dtype、宽度、额外消费者和内存别名反例正确回退。
2. 单算子：目标 GPU 与参考输出比较，拒绝非有限值；覆盖真实 shape 和边界。
3. 模型：同图像、状态、prompt、噪声、步数下比较 baseline/candidate；
   shadow 的原输出传播不能代替真实融合输出传播。
4. CUDA Graphs：关闭与允许两种设置；采集、重放以及重新输入正确。
5. 多个代表性真实输入：记录动作误差；FP8 的验收不能用单张合成图校准代替。
6. 性能：诊断关闭，独立进程交替 A/B，保存所有样本与长尾；profile 时间不能当正常请求时间。
7. 全部开启的组合检查：逐项通过不代表组合通过；最后以相同 E2E 边界复测。

## 现有证据与当下依赖

- LayerNorm 证据见 [flashrt-sm89-port.md](flashrt-sm89-port.md)。
- 4090、CUDA 13.0.88、GGUF 和构建产物位于用户的 gpu4；当前工作环境没有远端登录连接或 GPU。
- 下一份必要证据：预热后的 CUDA kernel / API 时间分布、模型实际 tensor dtype/shape，
  以及新 RMSNorm 分派的逐层和端到端结果。
- 此表中的“待实现”不能写成“已完成”，现有 ggml 回退不能计作 FlashRT kernel 已命中。

## 本次增量与本地检查

- 新增 `flashrt-rmsnorm.cu/.cuh` 与共用的 `flashrt-rmsnorm-guard.h`。
  编译开关沿用 `GGML_CUDA_FLASHRT`；运行时同时设置 `GGML_CUDA_FLASHRT=1`
  和 `GGML_CUDA_FLASHRT_RMSNORM=1` 才启用新分派，后者默认关闭。
- 保持 ggml 的归约线程分工和 FP32 中间舍入；融合三节点并处理 norm 的两个内部消费者。
  外部消费者、输出标记、非连续布局、别名、非 F32、eps 不符时回退。
- CUDA 13.0.88 / GCC 13.3 下，新 kernel 和修改后的 `ggml-cuda.cu`
  使用 `-arch=sm_89 -use_fast_math -extended-lambda -O3` 编译成功。
  当前环境没有 GPU；编译成功不能计作 CUDA 数值测试通过。
- 34 项主机图与别名契约检查通过；25 个新算子 fixture 在 CPU 后端通过。
- 自动流程的 10 项模拟 / 文件格式检查通过；4 项真实进程监督测试因当前环境
  procfs / PID namespace 不一致而跳过，打包在服务器流程中执行。它们不是 GPU 测试。

## GPU 执行入口

应用本次增量后，在原仓库执行：

```bash
PATH="/usr/sbin:/usr/bin:/sbin:/bin:$PATH" /usr/bin/python3 scripts/run-sm89-sprint.py
```

流程依赖原目录中的 `run-ggml-math-shadow.py`、已验证的 LayerNorm summary、A2 结果和
224×224 输入图像。仅停止身份与模型参数相符的本用户 18081 测试服务。

1. 读取两个 GGUF 的 tensor header，保存 dtype、名称与形状；验证测试流程。
2. 重新 CMake configure（新增 `.cu` 需要重新发现），编译并运行 34 项主机检查。
3. 在 CUDA0 / 物理 GPU 2 执行 baseline、candidate 各 25 项算子测试，验证实际命中。
4. graphs 关闭下比较真实 encoder / decoder RMSNorm 的 shadow 输出。
5. graphs 关闭 / 允许、原图 / 融合图的三组 prompt/state 输入对照；允许 graphs 时，
   每组输入至少六次，最后回到第一组检查状态恢复。所有动作要求与对应原图一致。
6. 关闭 shadow / trace，ABBA 四个独立服务各预热五次、测量二十次，保留全部长尾。
   A 为原 GGML，B 为 GGML-math LayerNorm + RMSNorm，报告的是组合收益。
7. 若已安装兼容的 Nsight Systems，启动独立 session，预热五次之后采集十次固定输入。
   保存 `.nsys-rep` 与 CUDA kernel / API / memcpy CSV；空采集不能视作成功。

任一数值检查失败即保存实际响应和日志。没有 nsys 时会报告 `profile_pending`，不会
安装系统软件或将未采集项目写成成功。summary 中的 `full_port_complete` 保持 `false`，
直到覆盖表中的全部 native 区域均完成适配和 GPU 验收。

本次固定图像的三组输入用于数值回归，不构成任务成功率评估，也不构成 FP8 校准集。
