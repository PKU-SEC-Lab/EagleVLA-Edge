# FlashRT bias + activation 的 GGML F32 适配

本增量在已验收的 GGML-math LayerNorm、Gemma RMSNorm、MMA16 GEMM=2 和
bias/residual 之上增加两条 activation 融合。新开关 GGML_CUDA_FLASHRT_BIAS_ACTIVATION
默认关闭；验收脚本只在其拥有的候选子进程中开启。

## 已取得的上阶段证据

来源是用户提供的 gpu4 终端结果，远端记录为
results/flashrt-sm89/bias-residual-stage-tbcuj047/summary.json。
status=bias_residual_model_stage_complete；
correctness=exact_to_accepted_gemm_for_three_input_cases_in_both_graph_settings。
设备为物理 GPU 4 / GPU-bc795c7e-702c-3874-ce74-babf32a5fa2c。

这个完成状态对应 bias/residual direct 与 CONT 两种模式、同一固定图像下的三个文本/状态案例、
Graph 关闭和允许，以及新组合真实输出传播的一致性；不代表完整 FlashRT 移植完成。

| 指标 | A P50 ms | B P50 ms | 本轮耗时降低 |
|---|---:|---:|---:|
| ViT | 6.342 | 6.202 | 2.20% |
| encode | 14.970 | 14.983 | -0.08% |
| decode | 53.567 | 54.915 | -2.52% |
| total | 68.663 | 69.810 | -1.67% |
| handler | 78.200 | 78.963 | -0.98% |
| infer HTTP | 79.998 | 80.860 | -1.08% |

A/B 都保持 norm 与 GEMM=2，仅切换 bias/residual。total 是 encode+decode，
不包含单列的 ViT；不能把这些列互相相加后再从 P50 推导总收益。

| 区块 | HTTP P50 ms | ViT P50 ms |
|---|---:|---:|
| A1 | 71.662 | 6.330 |
| B1 | 62.048 | 6.175 |
| B2 | 126.176 | 6.210 |
| A2 | 114.609 | 6.345 |

由显示值计算，ViT 两对都观察到小幅下降（约2.45%、2.13%）；HTTP 一对下降、一对上升。
80次 HTTP 样本中34次超过100ms，26次超过150ms。慢样本全部保留，既不能声称稳定
端到端收益，也不能从合并-1.08%直接断言稳定退化。尾延迟主要伴随 decode 波动，
其具体原因需读取已保存的 GPU/进程/CPU 遥测；采样范围不能提供单次请求的因果证明。

## 原生来源与接入范围

FlashRT 固定提交 e881e1424cf4fadee7dff121d01cc4e159e550e7：

- csrc/kernels/activation.cu 的 bias_gelu_kernel / bias_gelu_strict_kernel：
  上游导出 BF16 接口，模板采用 packed2 并融合广播 bias 与 GELU；本增量适配 F32
  输入、独立输出、GGML GELU 数学和图契约。不能写成原 pi0 FP16 路径已经使用 G7.11。
- 同文件 fused_add_silu_kernel / fused_add_silu_fp16：原生 Pi0 动作时间 MLP 的
  add+SiLU 融合；本增量增加 F32 行广播 bias 和独立输出适配。

宿主固定提交 PKU-SEC-Lab/Jetson-PI-Edge@1271fa933c3f81e1cb3a71a9f09f13bdefa4461b：

| 部分 | 原图 | 新目标 | 目标实际形状 |
|---|---|---|---|
| ViT FFN up | GEMM → ADD(bias) → GELU → CONT | 一个 bias+GELU kernel，省去连续复制 | F32 [4304,256] |
| 动作时间 MLP | GEMM → ADD(bias) → SiLU | 一个 bias+SiLU kernel | F32 [1024,10] |

ViT 目标来自 tools/mtmd/clip.cpp build_ffn，27个视觉层；DiT目标来自
src/models/pi0_ae.cpp，每个 denoise step 一组。不能把视觉 GELU 当成解码 GEGLU：
后者的 Gate*GELU*Up 已经是宿主 GLU kernel。宿主 SSM_CONV 的 bias+SiLU 特例也不等于
这里的通用 ADD→UNARY 已融合。模型的实际激活类型由 GGUF metadata 选择，
只有真实 GELU 节点可走此 GELU 路径；GELU_QUICK/ERF 必须保持回退，不改变模型语义。

## 数学、图与内存约束

- 保留先做一次 FP32 ADD 的舍入点，再调用宿主 unary.cuh 中相同的 GELU/SiLU 单值函数。
  不用代数上等价、浮点上不同的 sigmoid GELU 代换。
- 仅处理 SM89、F32、连续二维矩阵、8字节对齐、同设备标准 CUDA buffer 和白名单形状。
  GELU width=4304、rows=1/50/256/512；SiLU width=1024、rows=1/10/50。
- bias 为逐行广播通道向量。保留输入顺序，输出与输入同址时不使用错误的 restrict 承诺。
- 可选终端 CONT 必须保持完全相同的连续 layout；不能消去 transpose/reshape-copy。
- 中间 ADD/UNARY 不能有额外消费者、可观察输出或外部 view；省去的中间写不可被后续读取。
- 别名支持以正常 GGML allocator 和显式 view 元数据为边界，拒绝部分重叠或输出覆盖 bias；
  不支持绕过元数据手动设置任意隐藏指针别名。
- provenance/license 随源保留，上游 Apache-2.0 许可见已安装的
  vendor/flashrt/bias-residual-reference-LICENSE。

## 验收口径

新阶段独立构建并冻结源码、配置和二进制；旧 bias 报告仅提供动作回归参考。
A/B 都保持 norm、GEMM=2、bias/residual=1，只切换新 activation 开关。
测试图须检查原始 F32 输出一致、候选实际命中、反例回退、capture 与真实 cudaGraphLaunch。
CPU 的 tanh/exp 参考只作容差检查，不能假设它与 CUDA transcendental 函数逐位相同。
模型须在两种 Graph 设置中命中 GELU 与 SiLU，并观察到 ViT 的 CONT 被融合，
三个输入案例输出与已验收组合一致。性能保留全部 ABBA 样本、区块与合并统计。

GPU 空闲检查沿用已修复的两次连续采样标准，不因刚退出测试留下的单次利用率读数立即失败；
阈值仍为零计算进程、显存<=256 MiB、利用率<=5%。只有低显存无进程的高利用率读数可短暂重试。

## 本地交付验证

- CUDA 13.0.88、GCC 13.3、SM89：新 CUDA 后端、llama-server 和
  test-flashrt-bias-activation 整合编译与链接成功。
- 整合后的测试程序实际执行 122 项主机契约检查通过。
- 新 runner 的 9 项协议、负向门槛和自有进程清理 mock 检查通过；
  包括四个模型设置的22请求流程、44个fixture的严格协议以及缺失命中/capture的拒绝。
- 补丁在含旧norm/GEMM/bias接入的模拟工作树中可应用和反向恢复；终端Python和shell语法、
  内嵌压缩补丁SHA256均检查通过，打包新kernel/test源与实际编译的源完全相同。
- 本地无可用 GPU，尚未执行新候选的 CUDA 数值、真实回放、模型或性能验收。
  额外CPU构图/导出检查只验证数据流程；CPU后端的GELU半精度查表不能充当CUDA位模式参考。

## 完整迁移仍未完成的区域

FlashRT FP8 权重/scale 生命周期、量化及 GEMM/激活融合、QKV/RoPE/KV 布局、attention、
patch embedding 和 action update 仍需逐项接入与模型验收。当前 MMA16 是保持 GGML
算术的五形状优化，不应列为完整原生 FlashRT FP8 GEMM 已迁移。默认宿主 Pi0 逐步回传动作
并在 CPU 更新；它已有 PI0_DECODE_UNROLL 图内更新选项，因此单独贴一个 Euler kernel
不能被算作完成整条 GPU action 路径。所有阶段报告继续 full_port_complete=false。
