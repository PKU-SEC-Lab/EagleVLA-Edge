# FlashRT bias/residual 的 GGML F32 适配

这是已验收的 GGML-math LayerNorm、Gemma RMSNorm 和 MMA16 GEMM 之后的独立增量。
本阶段不改这三个已验收 kernel 的实现。新开关 `GGML_CUDA_FLASHRT_BIAS_RESIDUAL` 默认关闭。

## 上一阶段已取得的证据

以下来自用户提供的 gpu4 终端结果，而非本工作环境直接读取远端报告。

- GEMM micro：`results/flashrt-sm89/gemm-stage-p3zt54pe/summary.json`，`microtests_complete`。
  五种形状、三组数值输入，Graph 关闭/允许均通过原 GGML 与 MMA16 的二进制输出比较，
  且测试图记录了候选 capture 和实际 graph launch。
- 模型：`results/flashrt-sm89/gemm-mma16-e2e-9__3vz_m/summary.json`，
  `gemm_model_stage_complete`、`exact_for_three_input_cases_in_both_graph_settings`。
  同一张图像、固定/改变提示词/改变状态三用例，44 个候选和44个对照请求全部动作数值一致。
- 本轮 GPU UUID：`GPU-52904383-e783-198f-5b1b-9c7b79d34f87`。
- Graph 关闭时的 GEMM TRACE 27,720 / 22 = 每请求 1,260 次主机 enqueue；
  Graph 允许时756条是主机 capture/dispatch 日志，不代表GPU实际调用量骤减。
- 这些是五种形状的 GGML 算术保持优化；不能记成 FlashRT FP8 或完整原生 GEMM 已验收。
  FlashRT cuBLASLt mode 1 的独立诊断结果需要另从 micro 报告读取。

| K | N | M | GGML 微测试 μs | MMA16 微测试 μs | 观测耗时降低 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 256 | 11 | 9.146 | 7.735 | 15.42% |
| 1024 | 2048 | 11 | 9.864 | 9.419 | 4.51% |
| 1024 | 4096 | 11 | 11.226 | 10.401 | 7.35% |
| 2048 | 1024 | 11 | 11.503 | 10.147 | 11.79% |
| 4096 | 1024 | 11 | 15.800 | 13.143 | 16.82% |

微测试单位是同步 ggml graph 的墙钟时间，包含提交与同步，不是孤立 GPU kernel 时长。
表中每方数值是两块样本各自 P50 的平均值。

模型 A/B 均启用已验收 norm，仅比较增加 GEMM=2：
Infer HTTP 合并 P50 为85.190→62.672 ms，观测降低26.43%；decode P50为50.409→36.710 ms。
汇总计算没有问题，但运行波动明显，不能把26.43%写成稳定复现的收益。

| 区块 | HTTP P50 ms | HTTP P95 ms | 超过100 ms |
|---|---:|---:|---:|
| A1 | 72.419 | 394.150 | 8/20 |
| B1 | 60.369 | 240.573 | 7/20 |
| B2 | 75.507 | 243.458 | 7/20 |
| A2 | 102.799 | 284.129 | 10/20 |

80次测量中32次超过100 ms，28次超过150 ms。这里没有删掉慢样本，也没有仅凭时延推断
其他用户负载、降频、CPU抢占或kernel错误。下一阶段保留区块统计和运行期间的系统观测。

## 原生来源与融合目标

来源为 FlashRT 固定提交 `e881e1424cf4fadee7dff121d01cc4e159e550e7` 的
`csrc/kernels/elementwise.cu`，包括 `bias_res_strict_fp16_kernel` 和泛型独立输出
`bias_res_out_kernel<T>`。后者上游显式实例化 BF16；本增量为 GGML 增加 F32 接口适配，
保留每行一个 CTA、packed2 元素访问以及 bias/residual 融合结构。
这是明确的 dtype、输出接口和别名契约适配，不是声称原 FP16/BF16 接口原封不动可用。

原宿主：`PKU-SEC-Lab/Jetson-PI-Edge@1271fa933c3f81e1cb3a71a9f09f13bdefa4461b`。

| ViT 路径 | 原节点序列 | 本阶段目标 |
|---|---|---|
| Attention output | GEMM → ADD(bias) → ADD(residual) | 两次 ADD 合成一次 |
| FFN down | GEMM → ADD(bias) → CONT → ADD(residual) | 两次 ADD 和连续复制合成一次 |

SigLIP 的27个实际视觉层，理论上每张图各有27组目标。不能用权重矩阵统计中的28推定层数。
宿主已有 multi-ADD 融合，但要求链上每次 ADD 的第二输入 layout 相同；
bias `[1152,1]` 与 residual `[1152,256]` 不同，因此不覆盖此处目标。
原 Nsight ADD 的5.36%是全部 ADD 的总和，不能都归给本增量，也不能拿来承诺端到端收益。

## 数学和内存契约

- 保留先 `x+bias`、再与 residual 相加的两个 FP32 舍入点和原图的操作数次序。
  禁止重排成 `(x+residual)+bias`。
- 仅处理 SM89、连续二维 F32、width=1152、已测试 row 数和同设备常规 CUDA buffer。
  bias 必须为按行广播的通道向量。
- 中间 ADD/CONT 不能有外部消费者、可观察输出或外部 view。
  CONT 必须不改变 shape/stride，不能消去 transpose 或 reshape-copy。
- ggml allocator 可能原地复用完整输入。允许经过契约检查的同址、同 layout 的逐元素输出，
  每线程必须先加载所需输入，再写回；不作与这些别名相矛盾的 `__restrict__` 承诺。
- 拒绝 shifted/partial overlap、输出覆盖广播 bias、中间写入改变后续 residual/bias 等情况。
  不匹配时保留原 GGML 分派，不把 fallback 记作融合成功。
- 别名支持以正常 ggml allocator 和显式 view 为边界，不支持绕过 view/INPUT/OUTPUT 元数据
  手工把外部 tensor 指针绑定到中间存储并在未来读取的隐藏别名。

## 验收与性能口径

新 runner 独立构建并冻结本次源码/产物指纹。旧 GEMM 模型报告作为动作回归参考，
不会用旧二进制的 manifest 替新二进制作验收。

1. 主机契约和别名检查。
2. 原图/融合图各自在 Graph 关闭和允许下运行 CUDA fixture，比较全部输出，
   检查正例实际命中、反例回退，以及测试图真实 capture/replay。
3. 固定 LN/RMS/MMA16 开启，新 bias/residual 关闭/开启，进行三输入、两种 Graph 设置的实际模型对照。
   候选的 direct、cont 两条路径都必须有真实模型命中；Graph允许时两条路径均需进入capture。
4. 全部正确性通过后，关闭诊断做 A/B/B/A 性能测试，保存全部样本、逐区块和合并统计。
   GPU与主机遥测用于分析观测条件，不自动断言时延波动的原因。

本地已用 CUDA 13.0.88、GCC 13.3、SM89 完成新 CUDA 后端、llama-server 和测试程序的
整合编译/链接；实际执行 69 项主机契约检查通过。测试构图与顺序 F32 参考还在 CPU 后端
检查了 30 种案例，新 runner 的 25 项针对性 mock 检查通过。
本地环境无可用 GPU，这些检查不能替代 gpu4 的数值、重放和模型实测。
完整 kernel 移植仍需 FP8 的权重与scale管理、量化及其融合、QKV/RoPE/KV、attention、patch、
action更新和全组合验收。所有报告的 `full_port_complete` 继续保持 false。

源码参考：

- [FlashRT elementwise](https://github.com/flashrt-project/FlashRT/blob/e881e1424cf4fadee7dff121d01cc4e159e550e7/csrc/kernels/elementwise.cu)
- [SigLIP graph](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge/blob/1271fa933c3f81e1cb3a71a9f09f13bdefa4461b/tools/mtmd/models/siglip.cpp)
- [Attention/FFN construction](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge/blob/1271fa933c3f81e1cb3a71a9f09f13bdefa4461b/tools/mtmd/clip.cpp)
