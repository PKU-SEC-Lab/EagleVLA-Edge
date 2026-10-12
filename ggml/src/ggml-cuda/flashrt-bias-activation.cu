// SPDX-License-Identifier: Apache-2.0
// Adapted from FlashRT csrc/kernels/activation.cu, bias_gelu_kernel<T>,
// bias_gelu_strict_kernel<T> and fused_add_silu_kernel<T>, pinned commit
// e881e1424cf4fadee7dff121d01cc4e159e550e7:
// https://github.com/flashrt-project/FlashRT/blob/e881e1424cf4fadee7dff121d01cc4e159e550e7/csrc/kernels/activation.cu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Changes: F32/float2 data and a separate destination; channel-broadcast SiLU
// bias; explicit rounded ADD followed by the original GGML unary arithmetic;
// optional elision of a terminal contiguous copy, guarded by graph use, layout,
// alias, CUDA buffer and architecture contracts. This is a dtype/math/graph
// adaptation: FlashRT exports the bias-GELU fusion only for BF16, and its Pi0
// FP16 ViT path still calls separate bias and GELU kernels. Its Pi0 FP16
// action-time MLP does use the fused-add-SiLU kernel adapted here.

#ifdef GGML_CUDA_FLASHRT

#include "flashrt-bias-activation.cuh"
#include "flashrt-bias-activation-guard.h"
#include "common.cuh"
#include "unary.cuh"

#include <cstdlib>
#include <cstring>

namespace {

static bool flashrt_activation_env_one(const char * name) {
    const char * value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

static bool flashrt_activation_device_tensor(const ggml_tensor * t, int device) {
    return t && t->buffer &&
        ggml_backend_buffer_get_type(t->buffer) == ggml_backend_cuda_buffer_type(device);
}

// Original FlashRT packed-pair work assignment, specialized to the supported
// column counts. Exact input/output aliases are permitted, so do not use
// restrict: each pair of inputs is read before writing its output pair.
template <bool gelu, int width>
static __global__ void flashrt_bias_activation_f32_kernel(
        const float2 * x, const float2 * bias, float2 * output,
        int total_pairs, bool bias_first) {
    const int index = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (index >= total_pairs) {
        return;
    }
    const float2 xv = x[index];
    const float2 bv = bias[index % (width / 2)];
    const float2 sum = bias_first
        ? make_float2(__fadd_rn(bv.x, xv.x), __fadd_rn(bv.y, xv.y))
        : make_float2(__fadd_rn(xv.x, bv.x), __fadd_rn(xv.y, bv.y));
    output[index] = gelu
        ? make_float2(ggml_cuda_op_gelu_single(sum.x), ggml_cuda_op_gelu_single(sum.y))
        : make_float2(ggml_cuda_op_silu_single(sum.x), ggml_cuda_op_silu_single(sum.y));
}

} // namespace

int ggml_cuda_flashrt_bias_activation(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int index) {
    static const bool enabled = flashrt_activation_env_one("GGML_CUDA_FLASHRT") &&
        flashrt_activation_env_one("GGML_CUDA_FLASHRT_BIAS_ACTIVATION");
    static const bool trace = flashrt_activation_env_one("GGML_CUDA_FLASHRT_TRACE");
    if (!enabled || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_ADA_LOVELACE) {
        return 0;
    }
    ggml_flashrt_bias_activation_plan plan{};
    const char * rejection = nullptr;
    const int skip = ggml_flashrt_plan_bias_activation(graph, index, plan, trace ? &rejection : nullptr);
    if (!skip) {
        if (trace && rejection) {
            GGML_LOG_INFO("flashrt_bias_activation_skip: node=%s reason=%s\n",
                          graph->nodes[index]->name, rejection);
        }
        return 0;
    }
    if (!flashrt_activation_device_tensor(plan.x, ctx.device) ||
        !flashrt_activation_device_tensor(plan.bias, ctx.device) ||
        !flashrt_activation_device_tensor(plan.add, ctx.device) ||
        !flashrt_activation_device_tensor(plan.unary, ctx.device) ||
        (plan.cont && !flashrt_activation_device_tensor(plan.cont, ctx.device)) ||
        !flashrt_activation_device_tensor(plan.output, ctx.device)) {
        if (trace) {
            GGML_LOG_INFO("flashrt_bias_activation_skip: node=%s reason=device_buffer\n", plan.output->name);
        }
        return 0;
    }
    const int pairs = plan.width * plan.rows / 2;
    const int blocks = (pairs + 255) / 256;
    if (plan.kind == GGML_UNARY_OP_GELU) {
        flashrt_bias_activation_f32_kernel<true, 4304><<<blocks, 256, 0, ctx.stream()>>>(
            static_cast<const float2 *>(plan.x->data), static_cast<const float2 *>(plan.bias->data),
            static_cast<float2 *>(plan.output->data), pairs, plan.bias_first);
    } else {
        flashrt_bias_activation_f32_kernel<false, 1024><<<blocks, 256, 0, ctx.stream()>>>(
            static_cast<const float2 *>(plan.x->data), static_cast<const float2 *>(plan.bias->data),
            static_cast<float2 *>(plan.output->data), pairs, plan.bias_first);
    }
    CUDA_CHECK(cudaGetLastError());
    if (trace) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture));
        GGML_LOG_INFO("flashrt_bias_activation: kind=%s cont=%d node=%s width=%d rows=%d capture=%d\n",
            plan.kind == GGML_UNARY_OP_GELU ? "gelu" : "silu", plan.cont ? 1 : 0,
            plan.output->name, plan.width, plan.rows, capture == cudaStreamCaptureStatusActive ? 1 : 0);
    }
    return skip;
}

#endif // GGML_CUDA_FLASHRT
