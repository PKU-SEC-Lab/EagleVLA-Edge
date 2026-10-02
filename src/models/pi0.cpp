#include "models.h"
#include "pi-model.h"

#include <cstdlib>
#include <cstring>


void llama_model_pi0::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH_AE, hparams.n_ff_arr_ae, hparams.n_layer(), false);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_AE, hparams.n_embd_ae);
    ml.get_key(LLM_KV_INFERENCE_STEPS, hparams.inference_steps);
    ml.get_key(LLM_KV_ACTION_STEPS, hparams.action_steps);
    ml.get_key(LLM_KV_ACTION_DIM, hparams.action_dim);

    bool pi05 = false;
    ml.get_key("pi0.pi05", pi05, false);
    const pi_model_kind pi_model = pi_model_kind_from_env();
    if (pi_model_kind_is_pi0(pi_model)) {
        hparams.use_par_res = false;
    } else if (pi_model_kind_is_pi05(pi_model)) {
        hparams.use_par_res = true;
    } else {
        hparams.use_par_res = pi05;
    }

    hparams.f_norm_eps = 1e-6f;
    hparams.f_norm_rms_eps = 1e-6f;

    switch (hparams.n_layer()) {
        case 18: type = LLM_TYPE_2B; break;
        case 28: type = LLM_TYPE_7B; break;
        case 36: type = LLM_TYPE_2_8B; break;
        default: type = LLM_TYPE_UNKNOWN;
   }
}

void llama_model_pi0::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const bool load_pi05_tensors = hparams.use_par_res;
    const int pi0_tensor_flags = load_pi05_tensors ? TENSOR_NOT_REQUIRED : 0;
    const int pi05_tensor_flags = load_pi05_tensors ? 0 : TENSOR_NOT_REQUIRED;

    action_in = create_tensor(tn(LLM_TENSOR_ACTION_IN, "weight"), {n_action_dim, n_embd_ae}, 0);
    action_in_b = create_tensor(tn(LLM_TENSOR_ACTION_IN, "bias"), {n_embd_ae}, 0);
    action_out = create_tensor(tn(LLM_TENSOR_ACTION_OUT, "weight"), {n_embd_ae, n_action_dim}, 0);
    action_out_b = create_tensor(tn(LLM_TENSOR_ACTION_OUT, "bias"), {n_action_dim}, 0);

    action_time_in = create_tensor(tn(LLM_TENSOR_ACTION_TIME_IN, "weight"), {2 * n_embd_ae, n_embd_ae}, pi0_tensor_flags);
    action_time_in_b = create_tensor(tn(LLM_TENSOR_ACTION_TIME_IN, "bias"), {n_embd_ae}, pi0_tensor_flags);
    action_time_out = create_tensor(tn(LLM_TENSOR_ACTION_TIME_OUT, "weight"), {n_embd_ae, n_embd_ae}, pi0_tensor_flags);
    action_time_out_b = create_tensor(tn(LLM_TENSOR_ACTION_TIME_OUT, "bias"), {n_embd_ae}, pi0_tensor_flags);
    state_proj = create_tensor(tn(LLM_TENSOR_STATE_PROJ, "weight"), {n_action_dim, n_embd_ae}, pi0_tensor_flags);
    state_proj_b = create_tensor(tn(LLM_TENSOR_STATE_PROJ, "bias"), {n_embd_ae}, pi0_tensor_flags);

    time_mlp_in = create_tensor(tn(LLM_TENSOR_TIME_MLP_IN, "weight"), {n_embd_ae, n_embd_ae}, pi05_tensor_flags);
    time_mlp_in_b = create_tensor(tn(LLM_TENSOR_TIME_MLP_IN, "bias"), {n_embd_ae}, pi05_tensor_flags);
    time_mlp_out = create_tensor(tn(LLM_TENSOR_TIME_MLP_OUT, "weight"), {n_embd_ae, n_embd_ae}, pi05_tensor_flags);
    time_mlp_out_b = create_tensor(tn(LLM_TENSOR_TIME_MLP_OUT, "bias"), {n_embd_ae}, pi05_tensor_flags);

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    if (load_pi05_tensors) {
        ae_output_norm_dense = create_tensor(tn(LLM_TENSOR_AE_OUTPUT_NORM_DENSE, "weight"), {n_embd_ae, 3 * n_embd_ae}, pi05_tensor_flags);
        ae_output_norm_dense_b = create_tensor(tn(LLM_TENSOR_AE_OUTPUT_NORM_DENSE, "bias"), {3 * n_embd_ae}, pi05_tensor_flags);
    } else {
        ae_output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd_ae}, pi0_tensor_flags);
        ae_output_norm_dense = create_tensor(tn(LLM_TENSOR_AE_OUTPUT_NORM_DENSE, "weight"), {n_embd_ae, 3 * n_embd_ae}, TENSOR_NOT_REQUIRED);
        ae_output_norm_dense_b = create_tensor(tn(LLM_TENSOR_AE_OUTPUT_NORM_DENSE, "bias"), {3 * n_embd_ae}, TENSOR_NOT_REQUIRED);
    }
    output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

    for (int i = 0; i < n_layer / 2; ++i) {
        auto & layer = layers[i];
        const int64_t n_ff_i = hparams.n_ff(i);
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff_i}, 0);
        layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff_i}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_i, n_embd}, 0);
    }

    for (int i = n_layer / 2; i < n_layer; ++i) {
        auto & layer = layers[i];
        const int64_t n_ff_i = hparams.n_ff_ae(i);
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd_ae}, pi0_tensor_flags);
        layer.attn_norm_dense = create_tensor(tn(LLM_TENSOR_ATTN_NORM_DENSE, "weight", i), {n_embd_ae, 3 * n_embd_ae}, pi05_tensor_flags);
        layer.attn_norm_dense_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_DENSE, "bias", i), {3 * n_embd_ae}, pi05_tensor_flags);
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd_ae, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd_ae, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd_ae, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd_ae}, 0);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd_ae}, pi0_tensor_flags);
        layer.ffn_norm_dense = create_tensor(tn(LLM_TENSOR_FFN_NORM_DENSE, "weight", i), {n_embd_ae, 3 * n_embd_ae}, pi05_tensor_flags);
        layer.ffn_norm_dense_b = create_tensor(tn(LLM_TENSOR_FFN_NORM_DENSE, "bias", i), {3 * n_embd_ae}, pi05_tensor_flags);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd_ae, n_ff_i}, 0);
        layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd_ae, n_ff_i}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_i, n_embd_ae}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_pi0::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER || (params.ubatch.embd && params.ubatch.img_token_num == 0)) {
        return std::make_unique<llm_build_pi0_ae>(*this, params);
    }
    return std::make_unique<llm_build_pi0>(*this, params);
}



llm_build_pi0::llm_build_pi0(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    if (pi_model_kind_is_pi0(pi_model_kind_from_env())) {
    

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd_pi0(model.tok_embd);
    res->inpL   = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, inpL->ne[0], inpL->ne[1], inpL->ne[2]);
    ggml_tensor* inpL_copy_op = ggml_cpy(ctx0, inpL, res->inpL);
    ggml_build_forward_expand(gf, inpL_copy_op);

    cb(inpL, "inp", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_no_cache_pi0();


    ggml_tensor * inp_out_ids = build_inp_out_ids();
    

    for (int il = 0; il < n_layer/2; ++il) {

        cur = build_norm_pi0(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);

        cb(cur, "attn_norm", il);
        
        // self-attention
        {
            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            
            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);


            ggml_set_output(Kcur); // Force GGML to keep this tensor and its buffer alive.
            ggml_set_output(Vcur);

            
            res->encoded_kv[2*il]   = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, Kcur->ne[0],Kcur->ne[1],Kcur->ne[2]);
            res->encoded_kv[2*il+1] = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, Kcur->ne[0],Kcur->ne[1],Kcur->ne[2]);
            ggml_tensor* k_copy_op = ggml_cpy(ctx0, Kcur, res->encoded_kv[2*il]);
            ggml_tensor* v_copy_op = ggml_cpy(ctx0, Vcur, res->encoded_kv[2*il+1]);
            ggml_build_forward_expand(gf, k_copy_op);
            ggml_build_forward_expand(gf, v_copy_op);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);


            Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));
            cb(Qcur, "Qcur_scaled", il);
            
            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);
        }

        if (il == n_layer/2 - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        cur = build_norm_pi0(sa_out,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);

        cb(cur, "ffn_norm", il);

        // feed-forward network
        {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, sa_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    res->t_embd = cur;
    if (res->t_embd) {
        ggml_build_forward_expand(gf, res->t_embd);
    }

    ggml_build_forward_expand(gf, cur);

        return;
    }
    const bool is_pi05 = model.time_mlp_in != nullptr &&
                         model.time_mlp_out != nullptr &&
                         model.ae_output_norm_dense != nullptr;
    const char * pi05_debug_env = std::getenv("PI05_DEBUG_PREFIX");
    const bool keep_debug = is_pi05 && pi05_debug_env != nullptr && pi05_debug_env[0] != '\0' && std::strcmp(pi05_debug_env, "0") != 0;

    auto mark_debug = [&](ggml_tensor * t, const char * name, int il) {
        if (keep_debug && t != nullptr) {
            ggml_set_name(t, name);
            ggml_set_output(t);
        }
        cb(t, name, il);
    };

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd_pi0(model.tok_embd);
    if (keep_debug) {
        mark_debug(inpL, "pi05_dbg_prefix_input", -1);
    }

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * rope_ff = build_inp_rope_freq_factors_pi0();

    auto * inp_attn = build_attn_inp_no_cache_pi0();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int i = 0; i < n_layer/2; i++) {
        if (res->encoded_kv[2*i] == nullptr ||
                res->encoded_kv[2*i]->ne[0] != n_embd_head ||
                res->encoded_kv[2*i]->ne[1] != n_head_kv ||
                res->encoded_kv[2*i]->ne[2] != n_tokens) {
            res->encoded_kv[2*i]   = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd_head, n_head_kv, n_tokens);
        }
        if (res->encoded_kv[2*i+1] == nullptr ||
                res->encoded_kv[2*i+1]->ne[0] != n_embd_head ||
                res->encoded_kv[2*i+1]->ne[1] != n_head_kv ||
                res->encoded_kv[2*i+1]->ne[2] != n_tokens) {
            res->encoded_kv[2*i+1] = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd_head, n_head_kv, n_tokens);
        }
    }

    float q_scale = 1.0f / sqrtf(float(n_embd_head));

    std::vector<ggml_tensor*> kv_copy_ops;
    kv_copy_ops.reserve(n_layer);

    for (int il = 0; il < n_layer/2; ++il) {

        cur = build_norm_pi0(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);

        if (keep_debug && il == 0) {
            mark_debug(cur, "pi05_dbg_prefix_l0_attn_norm", -1);
        }

        // self-attention
        {
            // cuBLAS otherwise converts the same F32 normalized activation
            // independently for Q, K and V. Share one BF16 conversion when
            // all three projections use BF16 weights.
            ggml_tensor * qkv_in = cur;
            if (cur->type == GGML_TYPE_F32 &&
                    model.layers[il].wq->type == GGML_TYPE_BF16 &&
                    model.layers[il].wk->type == GGML_TYPE_BF16 &&
                    model.layers[il].wv->type == GGML_TYPE_BF16) {
                qkv_in = ggml_cast(ctx0, cur, GGML_TYPE_BF16);
            }
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, qkv_in);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, qkv_in);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, qkv_in);

            if (keep_debug && il == 0) {
                mark_debug(Qcur, "pi05_dbg_prefix_l0_q_mm", -1);
                mark_debug(Kcur, "pi05_dbg_prefix_l0_k_mm", -1);
                mark_debug(Vcur, "pi05_dbg_prefix_l0_v_mm", -1);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_ff,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_ff,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            if (keep_debug && il == 0) {
                mark_debug(Qcur, "pi05_dbg_prefix_l0_q_rope", -1);
                mark_debug(Kcur, "pi05_dbg_prefix_l0_k_rope", -1);
            }

            ggml_tensor* k_copy_op = ggml_cpy(ctx0, Kcur, res->encoded_kv[2*il]);
            ggml_tensor* v_copy_op = ggml_cpy(ctx0, Vcur, res->encoded_kv[2*il+1]);
            kv_copy_ops.push_back(k_copy_op);
            kv_copy_ops.push_back(v_copy_op);

            Qcur = ggml_scale(ctx0, Qcur, q_scale);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);
            if (keep_debug && il == 0) {
                cur = ggml_cont(ctx0, cur);
                mark_debug(cur, "pi05_dbg_prefix_l0_attn_out", -1);
            }
        }

        if (il == n_layer/2 - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        else{
            ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
            if (keep_debug && il == 0) {
                mark_debug(sa_out, "pi05_dbg_prefix_l0_attn_residual", -1);
            }

            cur = build_norm_pi0(sa_out,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);

            if (keep_debug && il == 0) {
                mark_debug(cur, "pi05_dbg_prefix_l0_ffn_norm", -1);
            }

            {
                // Parallel gate/up projections consume the same activation.
                // Convert it once instead of once per GEMM.
                ggml_tensor * ffn_in = cur;
                if (cur->type == GGML_TYPE_F32 &&
                        model.layers[il].ffn_up->type == GGML_TYPE_BF16 &&
                        model.layers[il].ffn_gate->type == GGML_TYPE_BF16) {
                    ffn_in = ggml_cast(ctx0, cur, GGML_TYPE_BF16);
                }
                cur = build_ffn(ffn_in,
                        model.layers[il].ffn_up,   NULL, NULL,
                        model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_PAR, il);
                if (keep_debug && il == 0) {
                    mark_debug(cur, "pi05_dbg_prefix_l0_ffn_out", -1);
                }
            }

            cur = ggml_add(ctx0, cur, sa_out);

            cur = build_cvec(cur, il);
            if (keep_debug && il == 0) {
                mark_debug(cur, "pi05_dbg_prefix_l0_ffn_residual", -1);
            }

            inpL = cur;
        }
    }

    for (auto* op : kv_copy_ops) {
        ggml_build_forward_expand(gf, op);
    }

    cur = inpL;

    res->t_embd = cur;
    if (keep_debug) {
        mark_debug(cur, "pi05_dbg_prefix_final_embd", -1);
    }

    if (res->t_embd) {
        ggml_build_forward_expand(gf, res->t_embd);
    }

    ggml_build_forward_expand(gf, cur);
}
