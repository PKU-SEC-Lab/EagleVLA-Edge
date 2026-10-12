// SPDX-License-Identifier: MIT
// SM89 FlashRT patch IM2COL fixtures. Both implementations must match an
// independent full-tensor CPU reference byte for byte, including FP16 rounding,
// signed zeros, padding, channel/batch order, and nonzero input view offsets.
#include "ggml.h"
#include "ggml-cuda/flashrt-patch-im2col-guard.h"
#ifndef FLASHRT_PATCH_HOST_ONLY
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
static void need(bool condition, const std::string & message) {
    if (!condition) { throw std::runtime_error(message); }
}
struct options {
    bool host_only = false;
    bool graphs = false;
    int mode = -1;
    std::string prefix;
};
static options parse(int argc, char ** argv) {
    options out;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--host-only") { out.host_only = true; continue; }
        if (key == "--help") {
            std::cout << "Usage: test-flashrt-patch-im2col --host-only\n"
                         "   or: --mode 0|1 --graphs off|on --output-prefix PATH\n";
            std::exit(0);
        }
        need(i + 1 < argc, "missing value after " + key);
        const std::string value = argv[++i];
        if (key == "--mode") {
            need(value == "0" || value == "1", "mode must be 0 or 1");
            out.mode = value == "1" ? 1 : 0;
        } else if (key == "--graphs") {
            need(value == "off" || value == "on", "graphs must be off or on");
            out.graphs = value == "on";
        } else if (key == "--output-prefix") { out.prefix = value; }
        else { throw std::runtime_error("unknown option: " + key); }
    }
    need(out.host_only || (out.mode >= 0 && !out.prefix.empty()), "mode and output-prefix are required");
    return out;
}
struct case_spec {
    std::string name;
    int image_width = 224;
    int image_height = 224;
    int patch_width = 14;
    int patch_height = 14;
    int stride_x = 14;
    int stride_y = 14;
    int pad_x = 0;
    int pad_y = 0;
    int batch = 1;
    ggml_type output_type = GGML_TYPE_F32;
    size_t view_offset_bytes = 0;
    bool eligible = true;
};
static std::vector<case_spec> gpu_cases() {
    std::vector<case_spec> cases;
    for (int batch : {1, 2}) {
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            case_spec s;
            s.name = "patch_b" + std::to_string(batch) + (type == GGML_TYPE_F32 ? "_f32" : "_f16");
            s.batch = batch;
            s.output_type = type;
            cases.push_back(s);
        }
    }
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        case_spec s;
        s.name = type == GGML_TYPE_F32 ? "patch_view_b1_f32" : "patch_view_b1_f16";
        s.output_type = type;
        // Deliberately element-aligned but not 16-byte-aligned.
        s.view_offset_bytes = 5 * sizeof(float);
        cases.push_back(s);
    }
    { case_spec s; s.name = "patch_fallback_image238_f32"; s.image_width = 238;
      s.eligible = false; cases.push_back(s); }
    { case_spec s; s.name = "patch_fallback_patch16_f32";
      s.patch_width = s.patch_height = s.stride_x = s.stride_y = 16;
      s.eligible = false; cases.push_back(s); }
    { case_spec s; s.name = "patch_fallback_pad1_f32"; s.pad_x = s.pad_y = 1;
      s.eligible = false; cases.push_back(s); }
    { case_spec s; s.name = "patch_fallback_stride7_f32"; s.stride_x = s.stride_y = 7;
      s.eligible = false; cases.push_back(s); }
    return cases;
}
struct fixture {
    ggml_context * ctx = nullptr;
    ggml_tensor * kernel = nullptr;
    ggml_tensor * storage = nullptr;
    ggml_tensor * image = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    case_spec spec;
#ifndef FLASHRT_PATCH_HOST_ONLY
    ggml_backend_buffer_t buffer = nullptr;
#endif
    explicit fixture(case_spec s, bool host = true) : spec(std::move(s)) {
        ctx = ggml_init({24 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true});
        need(ctx != nullptr, "metadata allocation failed");
        kernel = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, spec.patch_width, spec.patch_height, 3, 1152);
        if (spec.view_offset_bytes) {
            const size_t elements = size_t(spec.image_width) * spec.image_height * 3 * spec.batch;
            storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                int64_t(elements + spec.view_offset_bytes / sizeof(float) + 7));
            image = ggml_view_4d(ctx, storage, spec.image_width, spec.image_height, 3, spec.batch,
                size_t(spec.image_width) * sizeof(float),
                size_t(spec.image_width) * spec.image_height * sizeof(float),
                size_t(spec.image_width) * spec.image_height * 3 * sizeof(float), spec.view_offset_bytes);
        } else {
            image = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, spec.image_width, spec.image_height, 3, spec.batch);
            storage = image;
        }
        output = ggml_im2col(ctx, kernel, image, spec.stride_x, spec.stride_y, spec.pad_x, spec.pad_y,
                            1, 1, true, spec.output_type);
        ggml_set_name(kernel, (spec.name + "_kernel_metadata").c_str());
        ggml_set_name(storage, (spec.name + "_storage").c_str());
        ggml_set_name(image, (spec.name + "_image").c_str());
        ggml_set_name(output, spec.name.c_str());
        ggml_set_input(storage);
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, output);
        if (host) {
            // Only the guard examines these addresses; host tests never read them.
            kernel->data = reinterpret_cast<void *>(uintptr_t(0x10000000));
            storage->data = reinterpret_cast<void *>(uintptr_t(0x40000000));
            image->data = reinterpret_cast<void *>(uintptr_t(0x40000000) + spec.view_offset_bytes);
            output->data = reinterpret_cast<void *>(uintptr_t(0x70000000));
        }
    }
    fixture(const fixture &) = delete;
    fixture & operator=(const fixture &) = delete;
    ~fixture() {
#ifndef FLASHRT_PATCH_HOST_ONLY
        if (buffer) { ggml_backend_buffer_free(buffer); }
#endif
        if (ctx) { ggml_free(ctx); }
    }
    bool accepts() const {
        ggml_flashrt_patch_im2col_plan plan{};
        return ggml_flashrt_plan_patch_im2col(output, plan);
    }
#ifndef FLASHRT_PATCH_HOST_ONLY
    void allocate(ggml_backend_t backend) {
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        need(buffer != nullptr, "CUDA tensor allocation failed");
        need(ggml_nbytes(image) == size_t(spec.image_width) * spec.image_height * 3 * spec.batch * sizeof(float),
             "unexpected image extent");
        need(ggml_nbytes(output) == size_t(ggml_nelements(output)) * ggml_type_size(spec.output_type),
             "unexpected output extent");
    }
#endif
};
static int host_checks() {
    int count = 0;
    auto check = [&](bool actual, bool expected, const std::string & label) {
        ++count;
        need(actual == expected, "host contract: " + label);
    };
    for (const auto & spec : gpu_cases()) {
        fixture f(spec);
        check(f.accepts(), spec.eligible, spec.name);
        if (spec.eligible) {
            ggml_flashrt_patch_im2col_plan plan{};
            const char * reason = "unchanged";
            check(ggml_flashrt_plan_patch_im2col(f.output, plan, &reason) && reason == nullptr &&
                  plan.image == f.image && plan.kernel == f.kernel && plan.output == f.output &&
                  plan.batch == spec.batch, true, "accepted plan roles, batch and reason");
        }
    }
    const case_spec base{};
    { fixture f(base); f.output->op = GGML_OP_ADD; check(f.accepts(), false, "wrong operator"); }
    { fixture f(base); f.output->src[0] = nullptr; check(f.accepts(), false, "missing kernel metadata"); }
    { fixture f(base); f.output->src[1] = nullptr; check(f.accepts(), false, "missing image"); }
    { fixture f(base); f.output->src[2] = f.image; check(f.accepts(), false, "unexpected extra source"); }
    { fixture f(base); f.kernel->data = nullptr; check(f.accepts(), true, "kernel values are not read"); }
    { fixture f(base); f.kernel->type = GGML_TYPE_F32;
      for (int i = 0; i < 4; ++i) { f.kernel->nb[i] *= 2; }
      check(f.accepts(), true, "F32 kernel metadata"); }
    { fixture f(base); f.kernel->type = GGML_TYPE_BF16; check(f.accepts(), false, "unsupported kernel metadata dtype"); }
    { fixture f(base); f.image->type = GGML_TYPE_F16; check(f.accepts(), false, "unsupported image dtype"); }
    { fixture f(base); f.output->type = GGML_TYPE_BF16; check(f.accepts(), false, "unsupported output dtype"); }
    for (int param = 0; param < 7; ++param) {
        fixture f(base);
        int32_t value = 0;
        std::memcpy(&value, reinterpret_cast<const char *>(f.output->op_params) + param * sizeof(value), sizeof(value));
        ++value;
        std::memcpy(reinterpret_cast<char *>(f.output->op_params) + param * sizeof(value), &value, sizeof(value));
        check(f.accepts(), false, "unsupported convolution parameter");
    }
    for (int which : {0, 1, 2}) {
        for (int dimension = 0; dimension < 4; ++dimension) {
            { fixture f(base); ggml_tensor * t = which == 0 ? f.kernel : which == 1 ? f.image : f.output;
              t->nb[dimension] += 4; check(f.accepts(), false, "non-native stride"); }
            { fixture f(base); ggml_tensor * t = which == 0 ? f.kernel : which == 1 ? f.image : f.output;
              ++t->ne[dimension]; check(f.accepts(), false, "shape mismatch"); }
        }
    }
    for (int which : {0, 1}) {
        { fixture f(base); ggml_tensor * t = which == 0 ? f.image : f.output;
          t->data = nullptr; check(f.accepts(), false, "unallocated data tensor"); }
        { fixture f(base); ggml_tensor * t = which == 0 ? f.image : f.output;
          t->data = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(t->data) + 1);
          check(f.accepts(), false, "misaligned data tensor"); }
        { fixture f(base); ggml_tensor * t = which == 0 ? f.image : f.output;
          t->data = reinterpret_cast<void *>(std::numeric_limits<uintptr_t>::max() - 3);
          check(f.accepts(), false, "address interval overflow"); }
    }
    { fixture f(base); f.image->ne[0] = std::numeric_limits<int64_t>::max();
      check(f.accepts(), false, "huge malformed image extent"); }
    { fixture f(base); f.output->ne[0] = std::numeric_limits<int64_t>::max();
      check(f.accepts(), false, "huge malformed output extent"); }
    { fixture f(base); f.output->data = f.image->data; check(f.accepts(), false, "exact input/output alias"); }
    { fixture f(base); f.output->data = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(f.image->data) + 4);
      check(f.accepts(), false, "partial input/output alias"); }
    { fixture f(base); f.output->data = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(f.image->data) + ggml_nbytes(f.image));
      check(f.accepts(), true, "adjacent non-overlapping tensors"); }
    { fixture f(base); f.image->view_offs = sizeof(float);
      check(f.accepts(), false, "offset without a view source"); }
    { case_spec s = base; s.view_offset_bytes = 20; fixture f(s); f.storage->data = nullptr;
      check(f.accepts(), false, "view source without data"); }
    { case_spec s = base; s.view_offset_bytes = 20; fixture f(s);
      f.image->data = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(f.image->data) + sizeof(float));
      check(f.accepts(), false, "view address disagrees with source plus offset"); }
    { case_spec s = base; s.view_offset_bytes = 20; fixture f(s);
      f.image->view_offs = std::numeric_limits<size_t>::max() - 3;
      check(f.accepts(), false, "view source plus offset overflow"); }
    { ggml_flashrt_patch_im2col_plan plan{}; const char * reason = nullptr;
      check(ggml_flashrt_plan_patch_im2col(nullptr, plan, &reason), false, "null destination");
      check(reason != nullptr, true, "rejection reason supplied"); }
    std::cout << count << " patch IM2COL metadata/address contract checks passed; aliases are host-only\n";
    return count;
}

#ifndef FLASHRT_PATCH_HOST_ONLY
static void set_environment(const char * key, const char * value) {
#if defined(_WIN32)
    need(_putenv_s(key, value ? value : "") == 0, key);
#else
    need((value ? setenv(key, value, 1) : unsetenv(key)) == 0, key);
#endif
}
static std::string quoted(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') { out << '\\' << char(c); }
        else if (c < 32) { out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec; }
        else { out << char(c); }
    }
    out << '"';
    return out.str();
}
static std::vector<float> make_image(const case_spec & s, bool changed) {
    std::vector<float> image(size_t(s.image_width) * s.image_height * 3 * s.batch);
    for (int b = 0; b < s.batch; ++b) {
        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < s.image_height; ++y) {
                for (int x = 0; x < s.image_width; ++x) {
                    const size_t i = ((size_t(b) * 3 + c) * s.image_height + y) * s.image_width + x;
                    uint32_t hash = uint32_t(x + 1) * 0x9e3779b1u ^ uint32_t(y + 3) * 0x85ebca6bu ^
                        uint32_t(c + 7) * 0xc2b2ae35u ^ uint32_t(b + 11) * 0x27d4eb2fu ^
                        (changed ? 0xa142d37bu : 0x4392c8e1u);
                    hash ^= hash >> 16;
                    image[i] = float(int32_t(hash & 0xffffffu) - 8388608) / 2097152.f;
                    // Every channel and batch includes negative zero, positive
                    // zero, ties-to-even FP16 values, and first/last-pixel cases.
                    const size_t within_channel = size_t(y) * s.image_width + x;
                    if (within_channel % 997 == 0 || (x == s.image_width - 1 && y == s.image_height - 1)) {
                        static const float special[] = {
                            -0.0f, 0.0f, 1.00048828125f, 1.00146484375f,
                            -1.00048828125f, -1.00146484375f, 3.998046875f, -3.998046875f
                        };
                        const size_t index = (within_channel / 997 + size_t(c) + size_t(b) * 3 + (changed ? 3 : 0)) % 8;
                        image[i] = special[index];
                    }
                }
            }
        }
    }
    return image;
}
static std::vector<unsigned char> reference(const case_spec & s, const std::vector<float> & image) {
    // General 2D convolution coordinates, independent of the optimized patch
    // permutation. Every output coordinate is covered; padding is positive zero.
    const int ow = (s.image_width + 2 * s.pad_x - s.patch_width) / s.stride_x + 1;
    const int oh = (s.image_height + 2 * s.pad_y - s.patch_height) / s.stride_y + 1;
    const size_t patch_elements = size_t(3) * s.patch_width * s.patch_height;
    const size_t element_bytes = ggml_type_size(s.output_type);
    std::vector<unsigned char> result(size_t(s.batch) * oh * ow * patch_elements * element_bytes);
    for (int b = 0; b < s.batch; ++b) {
        for (int py = 0; py < oh; ++py) {
            for (int px = 0; px < ow; ++px) {
                for (int c = 0; c < 3; ++c) {
                    for (int ky = 0; ky < s.patch_height; ++ky) {
                        for (int kx = 0; kx < s.patch_width; ++kx) {
                            const int x = px * s.stride_x + kx - s.pad_x;
                            const int y = py * s.stride_y + ky - s.pad_y;
                            float value = 0.0f;
                            if (x >= 0 && x < s.image_width && y >= 0 && y < s.image_height) {
                                const size_t source = ((size_t(b) * 3 + c) * s.image_height + y) * s.image_width + x;
                                value = image.at(source);
                            }
                            const size_t element = ((size_t(b) * oh + py) * ow + px) * patch_elements +
                                (size_t(c) * s.patch_height + ky) * s.patch_width + kx;
                            if (s.output_type == GGML_TYPE_F16) {
                                const ggml_fp16_t rounded = ggml_fp32_to_fp16(value);
                                std::memcpy(result.data() + element * element_bytes, &rounded, sizeof(rounded));
                            } else { std::memcpy(result.data() + element * element_bytes, &value, sizeof(value)); }
                        }
                    }
                }
            }
        }
    }
    return result;
}
static void check_output(const std::vector<unsigned char> & actual, const std::vector<unsigned char> & expected,
                         ggml_type type, const std::string & label) {
    need(actual.size() == expected.size() && !actual.empty(), label + ": output extent mismatch");
    const size_t bytes = ggml_type_size(type);
    for (size_t element = 0; element < actual.size() / bytes; ++element) {
        float value;
        if (type == GGML_TYPE_F16) {
            ggml_fp16_t half;
            std::memcpy(&half, actual.data() + element * bytes, sizeof(half));
            value = ggml_fp16_to_fp32(half);
        } else { std::memcpy(&value, actual.data() + element * bytes, sizeof(value)); }
        if (!std::isfinite(value)) {
            throw std::runtime_error(label + ": nonfinite output at " + std::to_string(element));
        }
        if (std::memcmp(actual.data() + element * bytes, expected.data() + element * bytes, bytes) != 0) {
            uint32_t actual_bits = 0, expected_bits = 0;
            std::memcpy(&actual_bits, actual.data() + element * bytes, bytes);
            std::memcpy(&expected_bits, expected.data() + element * bytes, bytes);
            std::ostringstream message;
            message << label << ": full CPU reference bit mismatch at element=" << element << " actual=0x"
                    << std::hex << actual_bits << " expected=0x" << expected_bits;
            throw std::runtime_error(message.str());
        }
    }
}
struct backend_owner {
    ggml_backend_t backend = nullptr;
    ~backend_owner() { if (backend) { ggml_backend_synchronize(backend); ggml_backend_free(backend); } }
};
static void gpu_tests(const options & opts, int checks) {
    set_environment("GGML_CUDA_FLASHRT", "1");
    set_environment("GGML_CUDA_FLASHRT_PATCH_IM2COL", opts.mode ? "1" : "0");
    set_environment("GGML_CUDA_FLASHRT_TRACE", "1");
    set_environment("GGML_CUDA_DISABLE_GRAPHS", opts.graphs ? nullptr : "1");
    backend_owner device;
    device.backend = ggml_backend_cuda_init(0);
    need(device.backend != nullptr, "CUDA0 backend initialization failed");
    const uint16_t endian = 1;
    need(*reinterpret_cast<const unsigned char *>(&endian) == 1, "raw export requires little-endian host");
    std::ofstream raw(opts.prefix + ".bin", std::ios::binary | std::ios::trunc);
    need(bool(raw), "cannot open raw output file");
    std::ostringstream records;
    size_t offset = 0;
    bool first_record = true;
    for (const auto & spec : gpu_cases()) {
        fixture f(spec, false);
        f.allocate(device.backend);
        need(f.accepts() == spec.eligible, "allocated plan disagrees for " + spec.name);
        // Kernel values are deliberately irrelevant: this operation only reads
        // its metadata. Keep the allocated bytes deterministic nevertheless.
        ggml_backend_buffer_clear(f.buffer, 0);
        std::vector<unsigned char> initial;
        for (int phase_index = 0; phase_index < 2; ++phase_index) {
            const bool changed = phase_index != 0;
            const std::string phase = changed ? "changed" : "initial";
            const auto image = make_image(spec, changed);
            const auto expected = reference(spec, image);
            std::vector<float> storage;
            if (spec.view_offset_bytes) {
                storage.assign(size_t(ggml_nelements(f.storage)), -777.25f);
                std::copy(image.begin(), image.end(), storage.begin() + spec.view_offset_bytes / sizeof(float));
                ggml_backend_tensor_set(f.storage, storage.data(), 0, storage.size() * sizeof(float));
            } else { ggml_backend_tensor_set(f.image, image.data(), 0, image.size() * sizeof(float)); }
            const int calls = changed ? 3 : 8;
            std::vector<unsigned char> first;
            std::vector<unsigned char> actual(ggml_nbytes(f.output));
            for (int call = 0; call < calls; ++call) {
                need(ggml_backend_graph_compute(device.backend, f.graph) == GGML_STATUS_SUCCESS,
                     spec.name + ": graph compute failed");
                ggml_backend_synchronize(device.backend);
                ggml_backend_tensor_get(f.output, actual.data(), 0, actual.size());
                check_output(actual, expected, spec.output_type, spec.name + "/" + phase);
                if (call == 0) { first = actual; }
                else { need(first == actual, spec.name + "/" + phase + ": repeated output bit mismatch"); }
            }
            if (!changed) { initial = actual; }
            else { need(initial != actual, spec.name + ": input update did not change output"); }
            if (spec.view_offset_bytes) {
                std::vector<float> after(storage.size());
                ggml_backend_tensor_get(f.storage, after.data(), 0, after.size() * sizeof(float));
                need(std::memcmp(after.data(), storage.data(), storage.size() * sizeof(float)) == 0,
                     spec.name + ": source view storage or guards modified");
            }
            raw.write(reinterpret_cast<const char *>(actual.data()), std::streamsize(actual.size()));
            need(bool(raw), "raw output write failed");
            if (!first_record) { records << ",\n"; }
            first_record = false;
            records << "{\"name\":" << quoted(spec.name) << ",\"node\":" << quoted(ggml_get_name(f.output))
                    << ",\"image_width\":" << spec.image_width << ",\"image_height\":" << spec.image_height
                    << ",\"patch_width\":" << spec.patch_width << ",\"patch_height\":" << spec.patch_height
                    << ",\"stride_x\":" << spec.stride_x << ",\"stride_y\":" << spec.stride_y
                    << ",\"pad_x\":" << spec.pad_x << ",\"pad_y\":" << spec.pad_y << ",\"batch\":" << spec.batch
                    << ",\"output_type\":" << quoted(spec.output_type == GGML_TYPE_F32 ? "f32" : "f16")
                    << ",\"view_offset_bytes\":" << spec.view_offset_bytes
                    << ",\"eligible\":" << (spec.eligible ? "true" : "false")
                    << ",\"expected_candidate\":" << (spec.eligible && opts.mode ? "true" : "false")
                    << ",\"phase\":" << quoted(phase) << ",\"raw_dtype\":"
                    << quoted(spec.output_type == GGML_TYPE_F32 ? "little-endian-f32" : "little-endian-f16")
                    << ",\"raw_offset_bytes\":" << offset << ",\"raw_bytes\":" << actual.size()
                    << ",\"elements\":" << ggml_nelements(f.output) << ",\"compute_calls\":" << calls
                    << ",\"finite\":true,\"reference_elements\":" << ggml_nelements(f.output)
                    << ",\"reference_bit_mismatches\":0,\"repeat_bit_mismatches\":0}";
            offset += actual.size();
            std::cout << "PATCH_IM2COL_CASE name=" << spec.name << " phase=" << phase
                      << " expected_candidate=" << (spec.eligible && opts.mode ? 1 : 0)
                      << " elements=" << ggml_nelements(f.output) << " reference_exact=1 repeat_exact=1\n";
        }
    }
    raw.close();
    need(bool(raw), "raw output close failed");
    std::ofstream json(opts.prefix + ".json", std::ios::trunc);
    need(bool(json), "cannot open manifest output file");
    json << "{\"schema_version\":1,\"status\":\"tests_complete\",\"backend\":\"CUDA0\",\"mode\":" << opts.mode
         << ",\"graphs_requested\":" << (opts.graphs ? "true" : "false") << ",\"host_checks\":" << checks
         << ",\"case_count\":" << gpu_cases().size() << ",\"record_count\":" << 2 * gpu_cases().size()
         << ",\"raw_file\":" << quoted(opts.prefix + ".bin") << ",\"raw_bytes\":" << offset
         << ",\"input_update_checked\":true,\"gpu_alias_cases\":0"
         << ",\"reference\":\"Full independent CPU IM2COL coordinates; F32 copy or FP16 round-to-nearest-even\""
         << ",\"reference_scope\":\"all_output_elements_both_phases_every_compute\""
         << ",\"comparison_policy\":\"all four mode and Graph combinations must match raw output bytes exactly\""
         << ",\"records\":[\n" << records.str() << "\n]}\n";
    json.close();
    need(bool(json), "manifest output close failed");
    std::cout << "CUDA patch IM2COL tests passed: " << opts.prefix << ".json\n";
}
#endif
} // namespace
int main(int argc, char ** argv) {
    try {
        const options opts = parse(argc, argv);
        const int checks = host_checks();
#ifdef FLASHRT_PATCH_HOST_ONLY
        (void) checks;
#endif
        if (!opts.host_only) {
#ifdef FLASHRT_PATCH_HOST_ONLY
            throw std::runtime_error("this binary was compiled host-only; supply --host-only");
#else
            gpu_tests(opts, checks);
#endif
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Patch IM2COL test failed: " << error.what() << '\n';
        return 1;
    }
}
