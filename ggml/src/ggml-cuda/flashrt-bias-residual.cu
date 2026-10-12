// SPDX-License-Identifier: Apache-2.0
// Adapted from FlashRT csrc/kernels/elementwise.cu, bias_res_out_kernel<T>,
// commit e881e1424cf4fadee7dff121d01cc4e159e550e7:
// https://github.com/flashrt-project/FlashRT/blob/e881e1424cf4fadee7dff121d01cc4e159e550e7/csrc/kernels/elementwise.cu
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
// Changes: F32 input/output instead of BF16/FP16, float2 row access, separate
// destination, original GGML operand order and two explicit FP32 rounding
// points, and a graph/buffer/alias guard. This is an interface/math adaptation
// of FlashRT's native fusion, not its unchanged low-precision implementation.

#ifdef GGML_CUDA_FLASHRT

#include "flashrt-bias-residual.cuh"
#include "flashrt-bias-residual-guard.h"
#include "common.cuh"

#include <cstdlib>
#include <cstring>

namespace {

static bool flashrt_bias_env_one(const char * name) {
    const char * value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

static bool flashrt_bias_device_tensor(const ggml_tensor * t, int device) {
    return t && t->buffer &&
        ggml_backend_buffer_get_type(t->buffer) == ggml_backend_cuda_buffer_type(device);
}

// Exact same-element destination alias with X or residual is permitted by the
// guard. Deliberately no restrict qualifiers: all three pairs are loaded before
// writing their output pair, and partial/broadcast-write aliases are rejected.
static __global__ void flashrt_bias_residual_f32_kernel(
        const float2 * x, const float2 * bias, const float2 * residual,
        float2 * output, int width_pairs, bool bias_first, bool residual_first) {
    const int offset = int(blockIdx.x) * width_pairs;
    for (int column = int(threadIdx.x); column < width_pairs; column += int(blockDim.x)) {
        const float2 xv = x[offset + column];
        const float2 bv = bias[column];
        const float2 rv = residual[offset + column];
        const float2 sum = bias_first
            ? make_float2(__fadd_rn(bv.x, xv.x), __fadd_rn(bv.y, xv.y))
            : make_float2(__fadd_rn(xv.x, bv.x), __fadd_rn(xv.y, bv.y));
        output[offset + column] = residual_first
            ? make_float2(__fadd_rn(rv.x, sum.x), __fadd_rn(rv.y, sum.y))
            : make_float2(__fadd_rn(sum.x, rv.x), __fadd_rn(sum.y, rv.y));
    }
}

} // namespace

int ggml_cuda_flashrt_bias_residual(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int index) {
    static const bool enabled = flashrt_bias_env_one("GGML_CUDA_FLASHRT") &&
        flashrt_bias_env_one("GGML_CUDA_FLASHRT_BIAS_RESIDUAL");
    static const bool trace = flashrt_bias_env_one("GGML_CUDA_FLASHRT_TRACE");
    if (!enabled || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_ADA_LOVELACE) {
        return 0;
    }
    ggml_flashrt_bias_residual_plan plan{};
    const char * rejection = nullptr;
    const int skip = ggml_flashrt_plan_bias_residual(graph, index, plan, trace ? &rejection : nullptr);
    if (!skip) {
        if (trace && rejection) {
            GGML_LOG_INFO("flashrt_bias_residual_skip: node=%s reason=%s\n",
                          graph->nodes[index]->name, rejection);
        }
        return 0;
    }
    if (!flashrt_bias_device_tensor(plan.x, ctx.device) ||
        !flashrt_bias_device_tensor(plan.bias, ctx.device) ||
        !flashrt_bias_device_tensor(plan.residual, ctx.device) ||
        !flashrt_bias_device_tensor(plan.add, ctx.device) ||
        (plan.cont && !flashrt_bias_device_tensor(plan.cont, ctx.device)) ||
        !flashrt_bias_device_tensor(plan.output, ctx.device)) {
        if (trace) {
            GGML_LOG_INFO("flashrt_bias_residual_skip: node=%s reason=device_buffer\n", plan.output->name);
        }
        return 0;
    }
    flashrt_bias_residual_f32_kernel<<<plan.rows, 256, 0, ctx.stream()>>>(
        static_cast<const float2 *>(plan.x->data),
        static_cast<const float2 *>(plan.bias->data),
        static_cast<const float2 *>(plan.residual->data),
        static_cast<float2 *>(plan.output->data), plan.width / 2,
        plan.bias_first, plan.residual_first);
    CUDA_CHECK(cudaGetLastError());
    if (trace) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture));
        GGML_LOG_INFO("flashrt_bias_residual: pattern=%s node=%s width=%d rows=%d capture=%d\n",
            plan.cont ? "cont" : "direct", plan.output->name, plan.width, plan.rows,
            capture == cudaStreamCaptureStatusActive ? 1 : 0);
    }
    return skip;
}

#endif // GGML_CUDA_FLASHRT
