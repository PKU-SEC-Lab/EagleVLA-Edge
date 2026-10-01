// Generated from Jetson-PImerge_beifen/tools/mtmd/mtmd-helper.cpp; public symbols renamed for PI05 adapter dispatch.
// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama.h"
#include "clip-impl.h"
#include "pi-model.h"
#include "pi05-debug-dump.h"
#include "clip.h"

#include <algorithm>
#include <cinttypes>
#include <vector>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cstring>

//#define MTMD_AUDIO_DEBUG

#ifndef MTMD_AUDIO_DEBUG
#   define MA_NO_ENCODING
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

#include "stb/stb_image.h"

//
// internal logging functions
//

struct mtmd_helper_logger {
    ggml_log_callback default_callback = [](ggml_log_level level, const char * text, void * user_data) {
        (void) level;
        (void) user_data;
        fputs(text, stderr);
        fflush(stderr);
    };

    ggml_log_callback log_callback = default_callback;
    void * log_callback_user_data;

    void log_v(enum ggml_log_level level, const char * format, va_list args) {
        if (format == NULL) {
            return;
        }
        va_list args_copy;
        va_copy(args_copy, args);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            log_callback(level, buffer, log_callback_user_data);
        } else {
            char * buffer2 = (char *) calloc(len + 1, sizeof(char));
            vsnprintf(buffer2, len + 1, format, args_copy);
            buffer2[len] = 0;
            log_callback(level, buffer2, log_callback_user_data);
            free(buffer2);
        }
        va_end(args_copy);
    }

    void log(enum ggml_log_level level, const char * format, ...) {
        va_list args;
        va_start(args, format);
        log_v(level, format, args);
        va_end(args);
    }
} g_logger_pi05_adapter;

#define LOG_INF(...) g_logger_pi05_adapter.log(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) g_logger_pi05_adapter.log(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) g_logger_pi05_adapter.log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

static bool mtmd_is_pi0_model(mtmd_context *) {
    return true;
}

// Keep PI05's original "all camera views in one ViT graph" behavior while
// using the current mtmd batch API. The caller owns the flattened copy so the
// temporary mtmd_batch can be released immediately after encoding.
static int32_t mtmd_encode_chunks_batched_pi05(
        mtmd_context * ctx,
        const mtmd_input_chunks * chunks,
        const size_t * image_indices,
        size_t n_image_indices,
        int n_mmproj_embd,
        std::vector<float> & output) {
    output.clear();
    if (n_image_indices == 0) {
        return 0;
    }

    mtmd::batch_ptr batch(mtmd_batch_init(ctx));
    if (!batch) {
        LOG_ERR("failed to create PI05 ViT batch\n");
        return 1;
    }

    size_t total_floats = 0;
    for (size_t ii = 0; ii < n_image_indices; ++ii) {
        const size_t chunk_index = image_indices[ii];
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, chunk_index);
        if (chunk == nullptr || mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            LOG_ERR("PI05 batch entry %zu (chunk %zu) is not an image\n", ii, chunk_index);
            return 1;
        }

        const int32_t add_result = mtmd_batch_add_chunk(batch.get(), chunk);
        if (add_result != 0) {
            LOG_ERR("failed to add PI05 image chunk %zu to ViT batch: error %d\n",
                    chunk_index, add_result);
            return add_result;
        }
        total_floats += mtmd_input_chunk_get_n_tokens(chunk) * (size_t) n_mmproj_embd;
    }

    const int32_t encode_result = mtmd_batch_encode(batch.get());
    if (encode_result != 0) {
        return encode_result;
    }

    output.reserve(total_floats);
    for (size_t ii = 0; ii < n_image_indices; ++ii) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, image_indices[ii]);
        const size_t n_floats = mtmd_input_chunk_get_n_tokens(chunk) * (size_t) n_mmproj_embd;
        const float * embd = mtmd_batch_get_output_embd(batch.get(), chunk);
        if (embd == nullptr) {
            LOG_ERR("PI05 ViT batch produced no output for chunk %zu\n", image_indices[ii]);
            output.clear();
            return 1;
        }
        output.insert(output.end(), embd, embd + n_floats);
    }
    return 0;
}

static std::ofstream & pi05_debug_host_log_stream() {
    static bool initialized = false;
    static std::ofstream ofs;
    if (!initialized) {
        initialized = true;
        const char * dump_path = pi_model_debug_dump_file();
        if (dump_path != nullptr && dump_path[0] != '\0') {
            ofs.open(dump_path, std::ios::out | std::ios::app);
        }
    }
    return ofs;
}

static void pi05_debug_host_log_line(const std::string & line) {
    if (!pi_model_debug_enabled()) {
        return;
    }
    auto & ofs = pi05_debug_host_log_stream();
    if (!ofs.is_open()) {
        return;
    }
    ofs << "[mtmd] " << line << '\n';
    ofs.flush();
}

static void pi_model_debug_dump_tokens(const char * name, const llama_token * tokens, size_t n_tokens) {
    if (!pi_model_debug_enabled() || name == nullptr || tokens == nullptr) {
        return;
    }
    std::ostringstream oss;
    const size_t preview = std::min<size_t>((size_t) pi_model_debug_dump_values(32), n_tokens);
    oss << "name=" << name << " type=tokens n=" << n_tokens << " preview_count=" << preview << " values=";
    for (size_t i = 0; i < preview; ++i) {
        if (i > 0) {
            oss << ' ';
        }
        oss << tokens[i];
    }
    pi05_debug_host_log_line(oss.str());
}

static bool pi0_log_prefix_tokens_enabled() {
    return std::getenv("PI0_LOG_PREFIX_TOKENS") != nullptr;
}

static void pi0_log_prefix_token_ids(const char * label, const llama_token * tokens, size_t n_tokens) {
    if (!pi0_log_prefix_tokens_enabled()) {
        return;
    }
    fprintf(stderr, "[PI0_PREFIX] %s n=%zu ids:", label, n_tokens);
    for (size_t i = 0; i < n_tokens; ++i) {
        fprintf(stderr, "%s%d", i == 0 ? " " : ",", (int) tokens[i]);
    }
    fprintf(stderr, "\n");
}
static std::vector<llama_token> pi0_read_i32_file(const char * path) {
    std::vector<llama_token> out;
    if (path == nullptr || path[0] == '\0') {
        return out;
    }

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        fprintf(stderr, "[PI0_PREFIX] failed to open token file: %s\n", path);
        return out;
    }

    std::string data(
            (std::istreambuf_iterator<char>(ifs)),
            std::istreambuf_iterator<char>());
    for (char & ch : data) {
        if (ch == ',' || ch == '[' || ch == ']') {
            ch = ' ';
        }
    }

    std::stringstream ss(data);
    long long v = 0;
    while (ss >> v) {
        out.push_back((llama_token) v);
    }
    return out;
}

static std::vector<llama_token> pi0_forced_prefix_text_tokens_from_env() {
    const char * tokens_path = std::getenv("PI05_LANG_TOKENS_FILE");
    const char * masks_path  = std::getenv("PI05_LANG_MASKS_FILE");
    std::vector<llama_token> tokens = pi0_read_i32_file(tokens_path);
    std::vector<llama_token> masks  = pi0_read_i32_file(masks_path);

    if (tokens.empty()) {
        return {};
    }

    if (masks.empty()) {
        fprintf(stderr,
                "[PI0_PREFIX] using env text tokens n=%zu from %s\n",
                tokens.size(), tokens_path ? tokens_path : "");
        return tokens;
    }

    std::vector<llama_token> filtered;
    const size_t n = std::min(tokens.size(), masks.size());
    filtered.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (masks[i] != 0) {
            filtered.push_back(tokens[i]);
        }
    }

    fprintf(stderr,
            "[PI0_PREFIX] using env text tokens n=%zu from %s mask=%s\n",
            filtered.size(), tokens_path ? tokens_path : "", masks_path ? masks_path : "");
    return filtered;
}

static bool pi0_env_forced_prefix_text_enabled() {
    const char * tokens_path = std::getenv("PI05_LANG_TOKENS_FILE");
    return tokens_path != nullptr && tokens_path[0] != '\0';
}

static std::vector<llama_token> pi0_build_legacy_text_tokens(
        const llama_token * tokens,
        size_t n_tokens) {
    std::vector<llama_token> out;
    out.reserve(n_tokens + 1);
    out.push_back(2);

    const size_t n_payload = n_tokens > 3 ? n_tokens - 3 : n_tokens;
    if (tokens != nullptr && n_payload > 0) {
        out.insert(out.end(), tokens, tokens + n_payload);
    }

    return out;
}

static void pi0_append_text_tokens(
        llama_batch & combined_batch,
        llama_pos & n_past,
        llama_seq_id seq_id,
        const llama_token * tokens,
        size_t n_tokens,
        bool chunk_logits_last) {
    for (size_t i = 0; i < n_tokens; ++i) {
        const int32_t combined_j = combined_batch.n_tokens;
        combined_batch.token   [combined_j] = tokens[i];
        combined_batch.pos     [combined_j] = n_past++;
        combined_batch.n_seq_id[combined_j] = 1;
        combined_batch.seq_id  [combined_j][0] = seq_id;
        combined_batch.logits  [combined_j] = chunk_logits_last && (i + 1 == n_tokens);
        combined_batch.n_tokens++;
    }
}

void mtmd_helper_log_set_pi05_adapter(ggml_log_callback log_callback, void * user_data) {
    if (log_callback == nullptr) {
        log_callback = g_logger_pi05_adapter.default_callback;
    }
    g_logger_pi05_adapter.log_callback = log_callback;
    g_logger_pi05_adapter.log_callback_user_data = user_data;
    mtmd_log_set(log_callback, user_data);
}

//
// helper functions
//


size_t mtmd_helper_get_n_tokens_pi05_adapter(const mtmd_input_chunks * chunks) {
    size_t n_tokens = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_tokens += mtmd_input_chunk_get_n_tokens(chunk);
    }
    return n_tokens;
}

llama_pos mtmd_helper_get_n_pos_pi05_adapter(const mtmd_input_chunks * chunks) {
    llama_pos n_pos = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_pos += mtmd_input_chunk_get_n_pos(chunk);
    }
    return n_pos;
}

// helper struct to make working with embd batch easier
// note: this will be removed after llama_batch_ext refactoring
struct decode_embd_batch {
    int n_pos_per_embd;
    int n_mmproj_embd;
    std::vector<llama_pos>      pos;
    std::vector<llama_pos>      pos_view; // used by mrope
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, int n_pos_per_embd, int n_mmproj_embd) : n_pos_per_embd(n_pos_per_embd), n_mmproj_embd(n_mmproj_embd) {
        pos     .resize(n_tokens * n_pos_per_embd);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_ids [n_tokens] = nullptr;
        batch.n_tokens             = n_tokens;
        batch.token                = nullptr;
        batch.embd                 = embd;
        batch.embd2                = nullptr;
        batch.embd3                = nullptr;
        batch.img_token_num        = 0;
        batch.single_img_token_num = 0;
        batch.pos                  = pos.data();
        batch.n_seq_id             = n_seq_id.data();
        batch.seq_id               = seq_ids.data();
        batch.logits               = logits.data();
    }

    void set_position_normal(llama_pos pos_0, llama_seq_id seq_id) {
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    // M-RoPE for image
    void set_position_mrope_2d(llama_pos pos_0, int nx, int ny, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        seq_id_0[0] = seq_id;
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                int i = y * nx + x;
                pos[i                     ] = pos_0;
                pos[i + batch.n_tokens    ] = pos_0 + y;
                pos[i + batch.n_tokens * 2] = pos_0 + x;
                pos[i + batch.n_tokens * 3] = 0; // last pos dim is unused
            }
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    // M-RoPE for audio
    void set_position_mrope_1d(llama_pos pos_0, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; i++) {
            pos[i                     ] = pos_0 + i;
            pos[i + batch.n_tokens    ] = pos_0 + i;
            pos[i + batch.n_tokens * 2] = pos_0 + i;
            pos[i + batch.n_tokens * 3] = 0; // last pos dim is unused
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    llama_batch get_view(int offset, int n_tokens) {
        llama_pos * pos_ptr;
        pos_view.clear();
        pos_view.reserve(n_tokens * n_pos_per_embd);
        if (n_pos_per_embd > 1) {
            // mrope
            // for example, with layout of src: 1234...1234...1234...1234...
            //       offset 2 will give us dst: 34...34...34...34...
            for (int i = 0; i < n_pos_per_embd; i++) {
                // assume n_tokens is less than or equal to batch.n_tokens
                // batch.n_tokens is number of **total** tokens
                // n_tokens is number of viewed token
                size_t src_idx = i * batch.n_tokens + offset;
                pos_view.insert(pos_view.end(),
                    pos.data() + src_idx,
                    pos.data() + src_idx + n_tokens);
            }
            pos_ptr = pos_view.data();
        } else {
            // normal
            pos_ptr = pos.data() + offset;
        }
        llama_batch view = {};
        view.n_tokens             = n_tokens;
        view.token                = nullptr;
        view.embd                 = batch.embd + offset * n_mmproj_embd;
        view.embd2                = nullptr;
        view.embd3                = nullptr;
        view.img_token_num        = 0;
        view.single_img_token_num = 0;
        view.pos                  = pos_ptr;
        view.n_seq_id             = batch.n_seq_id + offset;
        view.seq_id               = batch.seq_id   + offset;
        view.logits               = batch.logits   + offset;
        return view;
    }
};

struct pi0_combined_chunk_info {
    size_t    src_idx;
    int       type;
    int32_t   combined_start;
    int32_t   combined_count;
    bool      skipped;
    llama_pos pos_min;
    llama_pos pos_max;
    bool      has_mrope;
    llama_pos mrope_t_min;
    llama_pos mrope_t_max;
    llama_pos mrope_y_min;
    llama_pos mrope_y_max;
    llama_pos mrope_x_min;
    llama_pos mrope_x_max;
};

static const char * pi0_chunk_type_name(int ty) {
    switch (ty) {
        case MTMD_INPUT_CHUNK_TYPE_TEXT:  return "text";
        case MTMD_INPUT_CHUNK_TYPE_IMAGE: return "image";
        case MTMD_INPUT_CHUNK_TYPE_AUDIO: return "audio";
        default: return "other";
    }
}

static void pi0_chunk_info_set_pos_range_from_combined(
        pi0_combined_chunk_info & info,
        const llama_batch & combined_batch) {
    if (info.combined_count <= 0) {
        info.pos_min = 0;
        info.pos_max = 0;
        return;
    }
    info.pos_min = combined_batch.pos[info.combined_start];
    info.pos_max = combined_batch.pos[info.combined_start];
    for (int32_t ti = info.combined_start + 1;
         ti < info.combined_start + info.combined_count; ++ti) {
        info.pos_min = std::min(info.pos_min, combined_batch.pos[ti]);
        info.pos_max = std::max(info.pos_max, combined_batch.pos[ti]);
    }
}

static void pi0_chunk_info_set_mrope_range(
        pi0_combined_chunk_info & info,
        const decode_embd_batch & batch_embd) {
    const int n = batch_embd.batch.n_tokens;
    if (n <= 0 || batch_embd.n_pos_per_embd <= 1) {
        info.has_mrope = false;
        return;
    }
    info.has_mrope = true;
    const auto & pos = batch_embd.pos;
    info.mrope_t_min = info.mrope_t_max = pos[0];
    info.mrope_y_min = info.mrope_y_max = pos[n];
    info.mrope_x_min = info.mrope_x_max = pos[2 * n];
    for (int i = 1; i < n; ++i) {
        info.mrope_t_min = std::min(info.mrope_t_min, pos[i]);
        info.mrope_t_max = std::max(info.mrope_t_max, pos[i]);
        info.mrope_y_min = std::min(info.mrope_y_min, pos[n + i]);
        info.mrope_y_max = std::max(info.mrope_y_max, pos[n + i]);
        info.mrope_x_min = std::min(info.mrope_x_min, pos[2 * n + i]);
        info.mrope_x_max = std::max(info.mrope_x_max, pos[2 * n + i]);
    }
}

static void pi0_log_combined_chunk_summary(
        const std::vector<pi0_combined_chunk_info> & infos,
        const llama_batch & combined_batch) {
    GGML_UNUSED(infos);
    GGML_UNUSED(combined_batch);
}

// Helper function for decoding an image whose embeddings have already been calculated
int32_t mtmd_helper_decode_image_chunk_pi05_adapter(
        mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        float * encoded_embd,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        llama_pos * new_n_past) {

    auto chunk_type = mtmd_input_chunk_get_type(chunk);
    const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_ERR("failed to decode chunk: input chunk not of image/audio type\n");
        return -1;
    }

    const llama_model * model = llama_get_model(lctx);
    int n_mmproj_embd = llama_model_n_embd_inp(model);
    int n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;

    int32_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    int32_t i_batch = 0;
    int32_t n_img_batches = GGML_PAD(n_tokens, n_batch) / n_batch;
    // printf("Debug: n_tokens = %d, n_batch = %d, n_mmproj_embd = %d, n_pos_per_embd = %d\n",
        // n_tokens, n_batch, n_mmproj_embd, n_pos_per_embd);
    // printf("Debug: embd pointer = %p\n", (void*)encoded_embd);
    decode_embd_batch batch_embd(encoded_embd, n_tokens, n_pos_per_embd, n_mmproj_embd);

    if (mtmd_decode_use_mrope(ctx)) {
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            if (!image_tokens) {
                LOG_ERR("failed to decode chunk: image tokens are null\n");
                return -1;
            }
            const int nx = mtmd_image_tokens_get_nx(image_tokens);
            const int ny = mtmd_image_tokens_get_ny(image_tokens);
            batch_embd.set_position_mrope_2d(n_past, nx, ny, seq_id);
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            batch_embd.set_position_mrope_1d(n_past, seq_id);
        } else {
            GGML_ABORT("invalid chunk type for M-RoPE");
        }
    } else {
        batch_embd.set_position_normal(n_past, seq_id);
    }

    if (mtmd_decode_use_non_causal(ctx, chunk)) {
        llama_set_causal_attn(lctx, false);
        // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the image
    }
    // batch_embd.batch.logits[n_tokens - 1] = true;
    while (i_batch < n_img_batches) { // split into batches
        int pos_offset = i_batch*n_batch;
        int n_tokens_batch = std::min(n_batch, n_tokens - pos_offset);
        llama_batch batch_embd_view = batch_embd.get_view(pos_offset, n_tokens_batch);

        LOG_INF("decoding %s batch %d/%d, n_tokens_batch = %d\n", name, i_batch+1, n_img_batches, n_tokens_batch);

        int64_t t1 = ggml_time_ms();

        if (mtmd_is_pi0_model(ctx)) {
            if (llama_encode(lctx, batch_embd_view)) {
                LOG_ERR("%s : failed to eval\n", __func__);
                return 1;
            }

            int32_t ret = llama_decode(lctx, batch_embd_view);

            if (ret != 0) {
                LOG_ERR("failed to decode %s\n", name);
                llama_set_causal_attn(lctx, true); // restore causal attn
                return ret;
            }
            // llama_batch batch = llama_batch_init(params.max_length, 0, 1);
            // batch.n_tokens    = params.max_length;
            // for (int32_t step = 0; step < steps_per_block; step++) {
            //     int32_t global_step = block_num * steps_per_block + step;
            //     if (params.step_callback) {
            //         if (!params.step_callback(
            //                 global_step, params.steps, output_tokens, params.max_length, params.step_callback_user_data)) {
            //             break;
            //         }
            //     }
            //     // Setup batch
            //     for (int32_t i = 0; i < params.max_length; i++) {
            //         batch.token[i]     = output_tokens[i];
            //         batch.pos[i]       = i;
            //         batch.n_seq_id[i]  = 1;
            //         batch.seq_id[i][0] = 0;
            //         batch.logits[i]    = 1;
            //     }
            //     int ret = llama_decode(ctx, batch);
            // }
        }
        else{
            int32_t ret = llama_decode(lctx, batch_embd_view);
            if (ret != 0) {
                LOG_ERR("failed to decode %s\n", name);
                llama_set_causal_attn(lctx, true); // restore causal attn
                return ret;
            }
        }

        LOG_INF("%s decoded (batch %d/%d) in %" PRId64 " ms\n", name, i_batch+1, n_img_batches, ggml_time_ms() - t1);


        i_batch++;
    }

    n_past += mtmd_input_chunk_get_n_pos(chunk);
    *new_n_past = n_past;

    if (mtmd_decode_use_non_causal(ctx, chunk)) {
        llama_set_causal_attn(lctx, true);
    }
    return 0;
}

int32_t mtmd_helper_eval_chunk_single_pi05_adapter(mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        bool logits_last,
        llama_pos * new_n_past) {
    int32_t ret;
    llama_batch text_batch = llama_batch_init(n_batch, 0, 1);
    auto chunk_type = mtmd_input_chunk_get_type(chunk);

    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        
        size_t n_tokens;
        const auto tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        // LOG_INF("decoding text chunk, n_tokens = %zu\n", n_tokens);
        size_t i = 0;
        while (i < n_tokens) { // split into batches
            text_batch.n_tokens = 0; // clear the batch
            for (; i < n_tokens && text_batch.n_tokens < n_batch; i++) {
                int32_t j = text_batch.n_tokens;
                text_batch.token   [j]    = tokens[i];
                text_batch.pos     [j]    = n_past++;
                text_batch.n_seq_id[j]    = 1;
                text_batch.seq_id  [j][0] = seq_id;
                text_batch.logits  [j]    = false;

                text_batch.n_tokens++;
            }
            
            bool is_last_token = (i == n_tokens);
            if (logits_last && is_last_token) {
                text_batch.logits[text_batch.n_tokens - 1] = true;
            }
            ret = llama_decode(lctx, text_batch);
            if (ret != 0) {
                LOG_ERR("failed to decode text\n");
                llama_batch_free(text_batch);
                return ret;
            }
            *new_n_past += text_batch.n_tokens;
        }
        

    } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
        int64_t t0 = ggml_time_ms();

        LOG_INF("encoding %s slice...\n", name);

        ret = mtmd_encode_chunk(ctx, chunk);

        if (ret != 0) {
            LOG_ERR("failed to encode %s slice\n", name);
            llama_batch_free(text_batch);
            return ret;
        }


        float * embd = mtmd_get_output_embd(ctx);

        ret = mtmd_helper_decode_image_chunk_pi05_adapter(ctx, lctx, chunk, embd, n_past, seq_id, n_batch, new_n_past);

        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            llama_batch_free(text_batch);
            return ret;
        }

    } else {
        GGML_ABORT("chunk type not supported");
    }

    llama_batch_free(text_batch);
    return 0;
}

void print_vector(const std::vector<float>& vec, const std::string& name = "Vector") {
    GGML_UNUSED(vec);
    GGML_UNUSED(name);
}

void mtmd_pi0_result_free_pi05_adapter(mtmd_pi0_result * result) {
    if (result == nullptr) {
        return;
    }

    if (result->state_data != nullptr) {
        free(result->state_data);
        result->state_data = nullptr;
    }

    if (result->action_data != nullptr) {
        free(result->action_data);
        result->action_data = nullptr;
    }

    if (result->action_final_data != nullptr) {
        free(result->action_final_data);
        result->action_final_data = nullptr;
    }

    result->has_state = false;
    result->state_dim = 0;
    result->has_action = false;
    result->action_dim = 0;
    result->action_steps = 0;
    result->has_action_final = false;
}

extern "C" int32_t mtmd_helper_eval_chunks_pi0_legacy(mtmd_context * ctx,
    struct llama_context * lctx,
    const mtmd_input_chunks * chunks,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * old_n_past,
    mtmd_pi0_result * pi0_result);

static int32_t mtmd_helper_eval_chunks_pi0_pi05(mtmd_context * ctx,
    struct llama_context * lctx,
    const mtmd_input_chunks * chunks,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * old_n_past,
    mtmd_pi0_result * pi0_result) {

    n_past = 0;
    if (pi0_result) {
        mtmd_pi0_result_free_pi05_adapter(pi0_result);
        *pi0_result = mtmd_pi0_result {
            /* vit_ms             = */ 0.0,
            /* encode_ms          = */ 0.0,
            /* decode_ms          = */ 0.0,
            /* total_ms           = */ 0.0,
            /* batch_build_ms     = */ 0.0,
            /* output_extract_ms  = */ 0.0,
            /* batch_free_ms      = */ 0.0,
            /* has_vit_ms         = */ false,
            /* has_encode_ms      = */ false,
            /* has_decode_ms      = */ false,
            /* has_total_ms       = */ false,
            /* has_batch_build_ms = */ false,
            /* has_output_extract_ms = */ false,
            /* has_batch_free_ms  = */ false,
            /* state_dim          = */ 0,
            /* has_state          = */ false,
            /* state_data         = */ nullptr,
            /* action_dim         = */ 0,
            /* action_steps       = */ 0,
            /* has_action         = */ false,
            /* action_data        = */ nullptr,
            /* has_action_final   = */ false,
            /* action_final_data  = */ nullptr,
            /* vit_preprocess_ms       = */ 0.0,
            /* vit_graph_build_alloc_ms = */ 0.0,
            /* vit_set_inputs_ms       = */ 0.0,
            /* vit_graph_compute_ms    = */ 0.0,
            /* vit_output_get_ms       = */ 0.0,
            /* vit_graph_reused        = */ false,
            /* has_vit_breakdown       = */ false,
        };
    }

    size_t n_chunks = mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        LOG_WRN("no chunks to eval\n");
        return 0;
    }
    const llama_model * pi0_text_model = llama_get_model(lctx);
    const pi_model_kind pi_model = pi_model_kind_from_env();
    const bool use_pi05_adapter = pi_model_use_pi05_adapters_by_default(pi_model) != 0;
    const bool use_pi0_legacy_adapter = !use_pi05_adapter;
    GGML_UNUSED(pi0_text_model);
    const int pi0_n_embd = llama_model_n_embd_inp(pi0_text_model);

    llama_batch combined_batch = llama_batch_init_pi0(n_batch, pi0_n_embd, 1);
    combined_batch.n_tokens = 0;
    bool pi0_prefix_text_appended = false;

    const int64_t t_batch_build_start = ggml_time_us();

    // P0 optimization: batch-encode all images before the main loop
    std::vector<size_t> batched_image_indices;
    for (size_t ij = 0; ij < n_chunks; ij++) {
        auto chunk = mtmd_input_chunks_get(chunks, ij);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            batched_image_indices.push_back(ij);
        }
    }

    std::vector<float> batched_embds;
    int batched_image_idx = 0;
    int n_tokens_per_image_cached = 0;
    int n_mmproj_embd_cached = pi0_n_embd;
    double batched_vit_ms = 0.0;
    double total_vit_ms = 0.0;
    // Match the original PI05 implementation: batch all camera views by
    // default. Retain the switch only as a diagnostic fallback.
    const bool disable_vit_batch = std::getenv("PI0_DISABLE_VIT_BATCH") != nullptr;

    int32_t pi0_prefix_image_tokens = 0;
    int32_t pi0_prefix_text_tokens  = 0;
    std::vector<pi0_combined_chunk_info> combined_chunk_infos;
    combined_chunk_infos.reserve(n_chunks);

    if (pi0_log_prefix_tokens_enabled()) {
        fprintf(stderr, "[PI0_PREFIX] n_chunks=%zu (build combined prefix batch)\n", n_chunks);
        for (size_t ij = 0; ij < n_chunks; ++ij) {
            auto c = mtmd_input_chunks_get(chunks, ij);
            const auto ty = mtmd_input_chunk_get_type(c);
            const char * ty_name = ty == MTMD_INPUT_CHUNK_TYPE_TEXT ? "text"
                                 : ty == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image"
                                 : ty == MTMD_INPUT_CHUNK_TYPE_AUDIO ? "audio"
                                 : "other";
            fprintf(stderr,
                    "[PI0_PREFIX]   chunk[%zu] type=%s n_tokens=%zu n_pos=%d\n",
                    ij, ty_name,
                    mtmd_input_chunk_get_n_tokens(c),
                    (int) mtmd_input_chunk_get_n_pos(c));
        }
    }

    {
        std::ostringstream oss;
        oss << "pi_model=" << pi_model_kind_name(pi_model)
            << " use_pi05_adapter=" << (use_pi05_adapter ? 1 : 0)
            << " n_chunks=" << n_chunks
            << " n_image_chunks=" << batched_image_indices.size();
        pi05_debug_host_log_line(oss.str());
    }
    if (use_pi05_adapter && !disable_vit_batch && batched_image_indices.size() > 1) {
        auto t_vit_start = std::chrono::high_resolution_clock::now();
        int32_t ret_batched = mtmd_encode_chunks_batched_pi05(
            ctx,
            chunks,
            batched_image_indices.data(),
            batched_image_indices.size(),
            n_mmproj_embd_cached,
            batched_embds);
        auto t_vit_end = std::chrono::high_resolution_clock::now();
        batched_vit_ms = std::chrono::duration<double, std::milli>(t_vit_end - t_vit_start).count();

        if (ret_batched == 0) {
            auto first_chunk = mtmd_input_chunks_get(chunks, batched_image_indices[0]);
            n_tokens_per_image_cached = mtmd_input_chunk_get_n_tokens(first_chunk);
            if (pi_model_env_truthy(std::getenv("LLAMA_PI0_PERF"))) {
                LOG_INF("batched ViT encoded %zu images in %.2f ms\n", batched_image_indices.size(), batched_vit_ms);
            }
            {
                std::ostringstream oss;
                oss << "name=pi_model_mtmd_batched_vit"
                    << " type=f32 shape=[" << n_mmproj_embd_cached << ", "
                    << n_tokens_per_image_cached << ", "
                    << batched_image_indices.size() << ", 1]"
                    << " n=" << batched_embds.size()
                    << " ms=" << batched_vit_ms;
                pi05_debug_host_log_line(oss.str());
            }
        } else {
            LOG_WRN("batched ViT failed, falling back to sequential encode\n");
        }
    }
#if 0
    {
        std::ostringstream oss;
        oss << "pi05_prefix_debug: n_chunks=" << n_chunks;
        pi05_debug_host_log_line(oss.str());
    }
#endif

    for (size_t ij = 0; ij < n_chunks; ij++) {
        
        bool chunk_logits_last = (ij == n_chunks - 1) && logits_last;
        auto chunk = mtmd_input_chunks_get(chunks, ij);










        // int32_t res = mtmd_helper_eval_chunk_single_pi05_adapter(ctx, lctx, chunk, n_past, seq_id, n_batch, chunk_logits_last, &n_past);
        llama_pos * new_n_past = &n_past;
        int32_t ret = 0;
        auto chunk_type = mtmd_input_chunk_get_type(chunk);
        if (use_pi0_legacy_adapter && n_chunks > 1 && ij == 0) {
            if (pi0_log_prefix_tokens_enabled()) {
                fprintf(stderr,
                        "[PI0_PREFIX] chunk[0]: skipped by PI_MODEL=pi0 legacy adapter\n");
            }
            pi0_combined_chunk_info skipped_info {};
            skipped_info.src_idx        = ij;
            skipped_info.type           = chunk_type;
            skipped_info.combined_start = combined_batch.n_tokens;
            skipped_info.combined_count = 0;
            skipped_info.skipped        = true;
            combined_chunk_infos.push_back(skipped_info);
            continue;
        }

#if 0
        {
            std::ostringstream oss;
            oss << "pi05_prefix_debug: chunk=" << ij
                << " type=" << (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT ? "text" : (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio"))
                << " n_tokens=" << mtmd_input_chunk_get_n_tokens(chunk)
                << " n_past=" << n_past;
            pi05_debug_host_log_line(oss.str());
        }
#endif
    
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            if (use_pi05_adapter && pi0_prefix_text_appended) {
                if (pi0_log_prefix_tokens_enabled()) {
                    fprintf(stderr,
                            "[PI0_PREFIX] text_chunk[%zu]: ignored (prefix text already appended)\n",
                            ij);
                }
                pi0_combined_chunk_info skipped_info {};
                skipped_info.src_idx        = ij;
                skipped_info.type           = chunk_type;
                skipped_info.combined_start = combined_batch.n_tokens;
                skipped_info.combined_count = 0;
                skipped_info.skipped        = true;
                combined_chunk_infos.push_back(skipped_info);
                continue;
            }

            size_t n_tokens_raw = 0;
            const auto tokens_raw = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens_raw);
            pi0_log_prefix_token_ids("text_chunk_raw_from_prompt", tokens_raw, n_tokens_raw);

            const int32_t combined_n_before_text = combined_batch.n_tokens;
            std::vector<llama_token> env_forced_tokens;
            std::vector<llama_token> text_tokens_with_bos;
            std::vector<llama_token> legacy_text_tokens;
            const llama_token * text_tokens = tokens_raw;
            size_t n_text_tokens = n_tokens_raw;
            pi_model_debug_dump_tokens("pi_model_mtmd_text_raw", tokens_raw, n_tokens_raw);
            if (use_pi0_legacy_adapter) {
                legacy_text_tokens = pi0_build_legacy_text_tokens(tokens_raw, n_tokens_raw);
                text_tokens = legacy_text_tokens.data();
                n_text_tokens = legacy_text_tokens.size();
            } else if (n_text_tokens > 0 && text_tokens[0] != 2) {
                text_tokens_with_bos.reserve(n_text_tokens + 1);
                text_tokens_with_bos.push_back(2);
                text_tokens_with_bos.insert(
                    text_tokens_with_bos.end(),
                    text_tokens,
                    text_tokens + n_text_tokens);
                text_tokens = text_tokens_with_bos.data();
                n_text_tokens = text_tokens_with_bos.size();
            }
            const bool use_env_forced_text = use_pi05_adapter && pi0_env_forced_prefix_text_enabled();
            if (use_env_forced_text) {
                env_forced_tokens = pi0_forced_prefix_text_tokens_from_env();
                text_tokens = env_forced_tokens.data();
                n_text_tokens = env_forced_tokens.size();
            }
            pi0_append_text_tokens(
                    combined_batch, n_past, seq_id, text_tokens, n_text_tokens, chunk_logits_last);
            pi0_prefix_text_appended = true;
            pi_model_debug_dump_tokens("pi_model_mtmd_text_appended", text_tokens, n_text_tokens);

            pi0_log_prefix_token_ids(
                    use_env_forced_text ? "text_in_combined_batch_env_forced" : "text_in_combined_batch_from_prompt",
                    text_tokens,
                    n_text_tokens);

            const int32_t n_text_added = combined_batch.n_tokens - combined_n_before_text;
            pi0_prefix_text_tokens += n_text_added;

            pi0_combined_chunk_info text_info {};
            text_info.src_idx        = ij;
            text_info.type           = chunk_type;
            text_info.combined_start = combined_n_before_text;
            text_info.combined_count = n_text_added;
            text_info.skipped        = false;
            text_info.has_mrope      = false;
            pi0_chunk_info_set_pos_range_from_combined(text_info, combined_batch);
            combined_chunk_infos.push_back(text_info);
            if (pi0_log_prefix_tokens_enabled()) {
                fprintf(stderr,
                        "[PI0_PREFIX] after text (chunk[%zu]): combined_n=%d (+%d text, image_so_far=%d)\n",
                        ij,
                        combined_batch.n_tokens,
                        n_text_added,
                        pi0_prefix_image_tokens);
            }
            // }
    
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
            int64_t t0 = ggml_time_ms();

            float * embd = nullptr;
            double vit_ms = 0.0;

            // Check if this image was batch-encoded
            if (!batched_embds.empty() && chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                int offset = batched_image_idx * n_tokens_per_image_cached * n_mmproj_embd_cached;
                embd = batched_embds.data() + offset;
                batched_image_idx++;
                vit_ms = (batched_image_idx == 1) ? batched_vit_ms : 0.0;
                total_vit_ms += vit_ms;
                if (pi_model_env_truthy(std::getenv("LLAMA_PI0_PERF"))) {
                    LOG_INF("using batched ViT result for image %d\n", batched_image_idx);
                }
            } else {
                // Sequential fallback
                LOG_INF("encoding %s slice...\n", name);
                auto t_vit_start = std::chrono::high_resolution_clock::now();
                ret = mtmd_encode_chunk(ctx, chunk);
                auto t_vit_end = std::chrono::high_resolution_clock::now();
                vit_ms = std::chrono::duration<double, std::milli>(t_vit_end - t_vit_start).count();
                if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                    total_vit_ms += vit_ms;
                }
                if (pi_model_env_truthy(std::getenv("LLAMA_PI0_PERF"))) {
                    fprintf(stderr, "Vit took: %.2f ms\n", vit_ms);
                }

                if (ret != 0) {
                    LOG_ERR("failed to encode %s slice\n", name);
                    return ret;
                }

                LOG_INF("%s slice encoded in %" PRId64 " ms\n", name, ggml_time_ms() - t0);
                embd = mtmd_get_output_embd(ctx);
            }

            if (pi0_result && chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                pi0_result->vit_ms = total_vit_ms;
                pi0_result->has_vit_ms = true;
            }








            // ret = mtmd_helper_decode_image_chunk_pi05_adapter(ctx, lctx, chunk, embd, n_past, seq_id, n_batch, new_n_past);


            // int32_t mtmd_helper_decode_image_chunk_pi05_adapter(
            //     mtmd_context * ctx,
            //     struct llama_context * lctx,
            //     const mtmd_input_chunk * chunk,
            //     float * encoded_embd,
            //     llama_pos n_past,
            //     llama_seq_id seq_id,
            //     int32_t n_batch,
            //     llama_pos * new_n_past) {
            float * encoded_embd = embd;
            auto chunk_type = mtmd_input_chunk_get_type(chunk);
            if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                LOG_ERR("failed to decode chunk: input chunk not of image/audio type\n");
                return -1;
            }

            const llama_model * model = llama_get_model(lctx);
            int n_mmproj_embd = llama_model_n_embd_inp(model);
            int n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;
        
            int32_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
            int32_t i_batch = 0;
            int32_t n_img_batches = GGML_PAD(n_tokens, n_batch) / n_batch;

            const int32_t combined_n_before_image = combined_batch.n_tokens;

            decode_embd_batch batch_embd(encoded_embd, n_tokens, n_pos_per_embd, n_mmproj_embd);

            if (mtmd_decode_use_mrope(ctx)) {
                if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                    const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
                    if (!image_tokens) {
                        LOG_ERR("failed to decode chunk: image tokens are null\n");
                        return -1;
                    }
                    const int nx = mtmd_image_tokens_get_nx(image_tokens);
                    const int ny = mtmd_image_tokens_get_ny(image_tokens);
                    batch_embd.set_position_mrope_2d(n_past, nx, ny, seq_id);
                } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                    batch_embd.set_position_mrope_1d(n_past, seq_id);
                } else {
                    GGML_ABORT("invalid chunk type for M-RoPE");
                }
            } else {
                batch_embd.set_position_normal(n_past, seq_id);
            }
        
            if (mtmd_decode_use_non_causal(ctx, chunk)) {
                llama_set_causal_attn(lctx, false);
                // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the image
            }

            // batch_embd.batch.logits[n_tokens - 1] = true;
            while (i_batch < n_img_batches) { // split into batches
                int pos_offset = i_batch*n_batch;
                int n_tokens_batch = std::min(n_batch, n_tokens - pos_offset);
                llama_batch batch_embd_view = batch_embd.get_view(pos_offset, n_tokens_batch);
                
                for (int i_image = 0; i_image < n_tokens_batch; i_image++) {
                    int32_t combined_j = combined_batch.n_tokens + i_image;
                    combined_batch.token   [combined_j]    = 0; // placeholder, will be filled during text chunk processing
                    combined_batch.pos     [combined_j]    = batch_embd_view.pos[i_image];
                    combined_batch.n_seq_id[combined_j]    = 1;
                    combined_batch.seq_id  [combined_j][0] = seq_id;
                    combined_batch.logits  [combined_j]    = false;
                    // printf("Debug: combined_batch.pos[%d] = %d\n", combined_j, combined_batch.pos[combined_j]);
                }
                
                combined_batch.n_tokens = batch_embd_view.n_tokens+combined_batch.n_tokens;
                combined_batch.img_token_num = combined_batch.img_token_num+ batch_embd_view.n_tokens;
                combined_batch.single_img_token_num = batch_embd_view.n_tokens;
                int img_num = 0;
                if (combined_batch.img_token_num>0){
                    img_num = combined_batch.img_token_num/ combined_batch.single_img_token_num;
                }
                if (img_num==1){
                    std::memcpy(combined_batch.embd, batch_embd_view.embd, (size_t) n_tokens_batch * n_mmproj_embd * sizeof(float));
                }
                else{
                    if (img_num==2){
                        std::memcpy(combined_batch.embd2, batch_embd_view.embd, (size_t) n_tokens_batch * n_mmproj_embd * sizeof(float));
                    }
                    else{
                        if (img_num==3){
                            std::memcpy(combined_batch.embd3, batch_embd_view.embd, (size_t) n_tokens_batch * n_mmproj_embd * sizeof(float));
                        }
                        else{
                            GGML_ABORT("Not supported more than 3 embd buffers");
                        }
                    }
                }
                
                
                if (pi_model_env_truthy(std::getenv("LLAMA_PI0_PERF"))) {
                    LOG_INF("decoding %s batch %d/%d, n_tokens_batch = %d\n", name, i_batch+1, n_img_batches, n_tokens_batch);
                }

                int64_t t1 = ggml_time_ms();

        

                

        
                
        
                if (pi_model_env_truthy(std::getenv("LLAMA_PI0_PERF"))) {
                    LOG_INF("%s decoded (batch %d/%d) in %" PRId64 " ms\n", name, i_batch+1, n_img_batches, ggml_time_ms() - t1);
                }
        
                i_batch++;
            }

            n_past += mtmd_input_chunk_get_n_pos(chunk);
            *new_n_past = n_past;
        
            if (mtmd_decode_use_non_causal(ctx, chunk)) {
                llama_set_causal_attn(lctx, true);
            }

            if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                const int32_t n_image_added = combined_batch.n_tokens - combined_n_before_image;
                pi0_prefix_image_tokens += n_image_added;
                if (pi0_log_prefix_tokens_enabled()) {
                    fprintf(stderr,
                            "[PI0_PREFIX] after image_chunk[%zu]: +%d image tokens, combined_n=%d "
                            "(image_total=%d text_total=%d)\n",
                            ij,
                            n_image_added,
                            combined_batch.n_tokens,
                            pi0_prefix_image_tokens,
                            pi0_prefix_text_tokens);
                }
            }

            {
                pi0_combined_chunk_info embd_info {};
                embd_info.src_idx        = ij;
                embd_info.type           = chunk_type;
                embd_info.combined_start = combined_n_before_image;
                embd_info.combined_count = combined_batch.n_tokens - combined_n_before_image;
                embd_info.skipped        = false;
                pi0_chunk_info_set_pos_range_from_combined(embd_info, combined_batch);
                pi0_chunk_info_set_mrope_range(embd_info, batch_embd);
                combined_chunk_infos.push_back(embd_info);
            }
        








            if (ret != 0) {
                LOG_ERR("failed to decode %s\n", name);
                return ret;
            }

        } else {
            GGML_ABORT("chunk type not supported");
        }
    
        if (ret != 0) {
            LOG_ERR("failed to eval chunk %zu\n", ij);
            return ret;
        }
        *old_n_past = n_past;
    }
// print_vector(std::vector<float>(combined_batch.embd, combined_batch.embd + 10), "Combined Embeddings image 0 token 1");
    // print_vector(std::vector<float>(combined_batch.embd + 255 * 2048, combined_batch.embd + 10 + 255 * 2048), "Combined Embeddings");
    

    // print_vector(std::vector<float>(combined_batch.pos, combined_batch.pos + combined_batch.n_tokens), "Combined Positions");
    // //print combined_batch.token
    // print_vector(std::vector<float>(combined_batch.token, combined_batch.token + combined_batch.n_tokens), "Combined Tokens");

    const int64_t t_batch_build_done = ggml_time_us();

    pi0_log_combined_chunk_summary(combined_chunk_infos, combined_batch);

    {
        std::ostringstream oss;
        oss << "name=pi_model_mtmd_combined_batch"
            << " n_tokens=" << combined_batch.n_tokens
            << " image_tokens=" << combined_batch.img_token_num
            << " single_img_token_num=" << combined_batch.single_img_token_num
            << " text_tokens=" << (combined_batch.n_tokens - combined_batch.img_token_num);
        pi05_debug_host_log_line(oss.str());
    }

    if (pi05_debug_binary_enabled()) {
        const int32_t n_tokens_total = combined_batch.n_tokens;
        const int32_t n_image_tokens = combined_batch.img_token_num;
        const int32_t n_text_tokens  = n_tokens_total - n_image_tokens;
        const int32_t tokens_per_image = combined_batch.single_img_token_num;
        const int32_t n_images = tokens_per_image > 0 ? n_image_tokens / tokens_per_image : 0;

        pi05_debug_dump_i32(
                "prefix_token_ids_all",
                reinterpret_cast<const int32_t *>(combined_batch.token),
                (size_t) n_tokens_total,
                { n_tokens_total });
        pi05_debug_dump_i32(
                "prefix_positions",
                reinterpret_cast<const int32_t *>(combined_batch.pos),
                (size_t) n_tokens_total,
                { n_tokens_total });
        if (n_text_tokens > 0) {
            pi05_debug_dump_i32(
                    "prefix_text_token_ids",
                    reinterpret_cast<const int32_t *>(combined_batch.token + n_image_tokens),
                    (size_t) n_text_tokens,
                    { n_text_tokens });
        }

        const size_t image_values = (size_t) tokens_per_image * (size_t) pi0_n_embd;
        if (n_images > 0 && combined_batch.embd != nullptr) {
            pi05_debug_dump_f32(
                    "image_0_mmproj",
                    combined_batch.embd,
                    image_values,
                    { pi0_n_embd, tokens_per_image });
        }
        if (n_images > 1 && combined_batch.embd2 != nullptr) {
            pi05_debug_dump_f32(
                    "image_1_mmproj",
                    combined_batch.embd2,
                    image_values,
                    { pi0_n_embd, tokens_per_image });
        }
        if (n_images > 2 && combined_batch.embd3 != nullptr) {
            pi05_debug_dump_f32(
                    "image_2_mmproj",
                    combined_batch.embd3,
                    image_values,
                    { pi0_n_embd, tokens_per_image });
        }
    }

    if (pi0_log_prefix_tokens_enabled()) {
        const int32_t text_slots = combined_batch.n_tokens - combined_batch.img_token_num;
        fprintf(stderr,
                "[PI0_PREFIX] prefix batch final: combined_n=%d image_tokens=%d "
                "single_img_per_view=%d text_slots=%d tracked_text=%d\n",
                combined_batch.n_tokens,
                combined_batch.img_token_num,
                combined_batch.single_img_token_num,
                text_slots,
                pi0_prefix_text_tokens);
        if (text_slots > 0) {
            pi0_log_prefix_token_ids(
                    "text_ids_in_combined_batch",
                    combined_batch.token + combined_batch.img_token_num,
                    (size_t) text_slots);
        }
        fprintf(stderr, "[PI0_PREFIX] combined positions (first 32):");
        const int32_t n_pos_print = std::min(combined_batch.n_tokens, 32);
        for (int32_t ti = 0; ti < n_pos_print; ++ti) {
            fprintf(stderr, "%s%d", ti == 0 ? " " : ",", (int) combined_batch.pos[ti]);
        }
        if (combined_batch.n_tokens > n_pos_print) {
            fprintf(stderr, ",...");
        }
        fprintf(stderr, "\n");
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto t_encode_start = std::chrono::high_resolution_clock::now();
    if (llama_encode(lctx, combined_batch)) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return 1;
    }
    
    auto t_encode_end = std::chrono::high_resolution_clock::now();
    double encode_ms = std::chrono::duration<double, std::milli>(t_encode_end - t_encode_start).count();

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    int32_t ret = llama_decode(lctx, combined_batch);

    
    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    double total_ms = std::chrono::duration<double, std::milli>(t_decode_end - start_time).count();
    if (pi0_result) {
        pi0_result->encode_ms = encode_ms;
        pi0_result->decode_ms = decode_ms;
        pi0_result->total_ms = total_ms;
        pi0_result->batch_build_ms = (t_batch_build_done - t_batch_build_start) / 1000.0;
        pi0_result->has_encode_ms = true;
        pi0_result->has_decode_ms = true;
        pi0_result->has_total_ms = true;
        pi0_result->has_batch_build_ms = true;
    }

    if (ret != 0) {
        LOG_ERR("failed to decode final\n");
        llama_set_causal_attn(lctx, true); // restore causal attn
        return ret;
    }

    const int64_t t_output_extract_start = ggml_time_us();
    if (pi0_result) {
        const float * state = llama_get_pi0_state(lctx);
        const int32_t state_dim = llama_get_pi0_state_size(lctx);
        if (state != nullptr && state_dim > 0) {
            pi0_result->state_data = (float *) malloc(sizeof(float) * state_dim);
            if (pi0_result->state_data == nullptr) {
                LOG_ERR("failed to allocate pi0 state buffer\n");
                llama_batch_free_pi0(combined_batch);
                return 1;
            }

            std::copy(state, state + state_dim, pi0_result->state_data);
            pi0_result->state_dim = state_dim;
            pi0_result->has_state = true;
        }

        const float * action = llama_get_pi0_action_input(lctx);
        const float * action_final = llama_get_pi0_action(lctx);
        const int32_t action_dim = llama_get_pi0_action_dim(lctx);
        const int32_t action_steps = llama_get_pi0_action_steps(lctx);
        const int64_t action_count = (int64_t) action_dim * action_steps;

        if (action != nullptr && action_dim > 0 && action_steps > 0 && action_count > 0) {
            pi0_result->action_data = (float *) malloc(sizeof(float) * action_count);
            if (pi0_result->action_data == nullptr) {
                LOG_ERR("failed to allocate pi0 action buffer\n");
                llama_batch_free_pi0(combined_batch);
                return 1;
            }

            std::copy(action, action + action_count, pi0_result->action_data);
            pi0_result->action_dim = action_dim;
            pi0_result->action_steps = action_steps;
            pi0_result->has_action = true;
        }

        if (action_final != nullptr && action_dim > 0 && action_steps > 0 && action_count > 0) {
            pi0_result->action_final_data = (float *) malloc(sizeof(float) * action_count);
            if (pi0_result->action_final_data == nullptr) {
                LOG_ERR("failed to allocate pi0 final action buffer\n");
                llama_batch_free_pi0(combined_batch);
                return 1;
            }

            std::copy(action_final, action_final + action_count, pi0_result->action_final_data);
            pi0_result->has_action_final = true;
        }
    }
    const int64_t t_output_extract_done = ggml_time_us();
    const int64_t t_batch_free_start = ggml_time_us();
    llama_batch_free_pi0(combined_batch);
    const int64_t t_batch_free_done = ggml_time_us();

    if (pi0_result) {
        pi0_result->output_extract_ms = (t_output_extract_done - t_output_extract_start) / 1000.0;
        pi0_result->batch_free_ms = (t_batch_free_done - t_batch_free_start) / 1000.0;
        pi0_result->has_output_extract_ms = true;
        pi0_result->has_batch_free_ms = true;
    }

    return 0;
}




extern "C" int32_t mtmd_helper_eval_chunks_pi0_pi05_adapter(mtmd_context * ctx,
    struct llama_context * lctx,
    const mtmd_input_chunks * chunks,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * old_n_past,
    mtmd_pi0_result * pi0_result) {

    const pi_model_kind pi_model = pi_model_kind_from_env();
    if (pi_model_kind_is_pi0(pi_model)) {
        pi05_debug_host_log_line("[mtmd] pi_model=pi0 dispatch=legacy_jetson_pi");
        return mtmd_helper_eval_chunks_pi0_legacy(
            ctx, lctx, chunks, n_past, seq_id, n_batch, logits_last, old_n_past, pi0_result);
    }

    {
        std::ostringstream oss;
        oss << "[mtmd] pi_model=" << pi_model_kind_name(pi_model)
            << " dispatch=pi05_current";
        pi05_debug_host_log_line(oss.str());
    }

    return mtmd_helper_eval_chunks_pi0_pi05(
        ctx, lctx, chunks, n_past, seq_id, n_batch, logits_last, old_n_past, pi0_result);
}

int32_t mtmd_helper_eval_chunks_pi05_adapter_eval_chunks(mtmd_context * ctx,
                                struct llama_context * lctx,
                                const mtmd_input_chunks * chunks,
                                llama_pos n_past,
                                llama_seq_id seq_id,
                                int32_t n_batch,
                                bool logits_last,
                                llama_pos * new_n_past) {
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        LOG_WRN("no chunks to eval\n");
        return 0;
    }

    for (size_t i = 0; i < n_chunks; i++) {
        
        bool chunk_logits_last = (i == n_chunks - 1) && logits_last;
        auto chunk = mtmd_input_chunks_get(chunks, i);

        int32_t res = mtmd_helper_eval_chunk_single_pi05_adapter(ctx, lctx, chunk, n_past, seq_id, n_batch, chunk_logits_last, &n_past);
        if (res != 0) {
            LOG_ERR("failed to eval chunk %zu\n", i);
            return res;
        }
        *new_n_past = n_past;
    }

    return 0;
}

namespace audio_helpers {

static bool is_audio_file(const char * buf, size_t len) {
    GGML_UNUSED(buf);
    GGML_UNUSED(len);
    return false;
#if 0
    if (len < 12) {
        return false;
    }

    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    bool is_wav = memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WAVE", 4) == 0;
    bool is_mp3 = len >= 3 && (
        memcmp(buf, "ID3", 3) == 0 ||
        // Check for MPEG sync word (simplified check)
        ((unsigned char)buf[0] == 0xFF && ((unsigned char)buf[1] & 0xE0) == 0xE0)
    );
    bool is_flac = memcmp(buf, "fLaC", 4) == 0;

    return is_wav || is_mp3 || is_flac;
#endif
}

// returns true if the buffer is a valid audio file
static bool decode_audio_from_buf(const unsigned char * buf_in, size_t len, int target_sampler_rate, std::vector<float> & pcmf32_mono) {
    GGML_UNUSED(buf_in);
    GGML_UNUSED(len);
    GGML_UNUSED(target_sampler_rate);
    GGML_UNUSED(pcmf32_mono);
    return false;
#if 0
    ma_result result;
    const int channels = 1;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, channels, target_sampler_rate);
    ma_decoder decoder;

    result = ma_decoder_init_memory(buf_in, len, &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        return false;
    }

    ma_uint64 frame_count;
    ma_uint64 frames_read;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    pcmf32_mono.resize(frame_count);
    result = ma_decoder_read_pcm_frames(&decoder, pcmf32_mono.data(), frame_count, &frames_read);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

#ifdef MTMD_AUDIO_DEBUG
    // save audio to wav file
    ma_encoder_config config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, target_sampler_rate);
    ma_encoder encoder;
    ma_encoder_init_file("output.wav", &config, &encoder);
    ma_encoder_write_pcm_frames(&encoder, pcmf32_mono.data(), pcmf32_mono.size(), &frames_read);
    ma_encoder_uninit(&encoder);
#endif

    ma_decoder_uninit(&decoder);
    return true;
#endif
}

} // namespace audio_helpers

mtmd_bitmap * mtmd_helper_bitmap_init_from_buf_pi05_adapter(mtmd_context * ctx, const unsigned char * buf, size_t len) {
    if (audio_helpers::is_audio_file((const char *)buf, len)) {
        std::vector<float> pcmf32;
        int bitrate = mtmd_get_audio_sample_rate(ctx);
        if (bitrate < 0) {
            LOG_ERR("This model does not support audio input\n");
            return nullptr;
        }
        if (!audio_helpers::decode_audio_from_buf(buf, len, bitrate, pcmf32)) {
            LOG_ERR("Unable to read WAV audio file from buffer\n");
            return nullptr;
        }
        return mtmd_bitmap_init_from_audio(pcmf32.size(), pcmf32.data());
    }

    // otherwise, we assume it's an image
    mtmd_bitmap * result = nullptr;
    {
        int nx, ny, nc;
        auto * data = stbi_load_from_memory(buf, len, &nx, &ny, &nc, 3);
        if (!data) {
            LOG_ERR("%s: failed to decode image bytes\n", __func__);
            return nullptr;
        }
        result = mtmd_bitmap_init(nx, ny, data);
        stbi_image_free(data);
    }
    return result;
}

mtmd_bitmap * mtmd_helper_bitmap_init_from_file_pi05_adapter(mtmd_context * ctx, const char * fname) {
    std::vector<unsigned char> buf;
    FILE * f = fopen(fname, "rb");
    if (!f) {
        LOG_ERR("Unable to open file %s: %s\n", fname, strerror(errno));
        return nullptr;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf.resize(file_size);

    size_t n_read = fread(buf.data(), 1, file_size, f);
    fclose(f);
    if (n_read != (size_t)file_size) {
        LOG_ERR("Failed to read entire file %s", fname);
        return nullptr;
    }

    return mtmd_helper_bitmap_init_from_buf_pi05_adapter(ctx, buf.data(), buf.size());
}
