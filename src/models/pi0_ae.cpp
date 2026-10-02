#include "models.h"
#include "pi-model.h"
#include "pi0-perf.h"

#include <cstdlib>
#include <cstring>


llm_build_pi0_ae::llm_build_pi0_ae(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t n_inference_steps = hparams.inference_steps;
    const int64_t action_steps = hparams.action_steps;
    const int64_t ae_hidden_dim = hparams.n_embd_ae;
    GGML_UNUSED(ae_hidden_dim);

    // Legacy Jetson-PI AE path for PI_MODEL=pi0
    if (pi_model_kind_is_pi0(pi_model_kind_from_env())) {
        const int32_t n_unroll = cross && cross->pi0_decode_unroll >= 2
            ? cross->pi0_decode_unroll
            : 1;
        ggml_tensor * cur;
        ggml_tensor * inpL;

        auto cross_kv_tensors = build_inp_cross_kv_pi0();
        ggml_tensor * inp_pos_ae = build_inp_pos_ae(action_steps);
        auto * inp_attn_ae = build_attn_inp_no_cache_ae(action_steps+1,cross_kv_tensors[0]->ne[2]+action_steps+1);

        ggml_tensor * state = build_inp_state();
        ggml_tensor * actions = build_inp_action();
        ggml_cont(ctx0, actions);
        cb(state, "state_in", -1);
        cb(actions, "action_in", -1);
        
        
        
        state = build_lora_mm(model.state_proj, state);
        state = ggml_add(ctx0, state, model.state_proj_b);
        ggml_tensor * last_state = nullptr;
        for (int32_t step = 0; step < n_unroll; ++step) {
            ggml_tensor * time_expanded = build_inp_sinusoidal_embedding(step);
            cur = build_lora_mm(model.action_in, actions);
            cur = ggml_add(ctx0, cur, model.action_in_b);
            cur = ggml_concat(ctx0, cur, time_expanded, 0);
            cur = build_lora_mm(model.action_time_in, cur);
            cur = ggml_add(ctx0, cur, model.action_time_in_b);
            cur = ggml_silu(ctx0, cur);
            cur = build_lora_mm(model.action_time_out, cur);
            cur = ggml_add(ctx0, cur, model.action_time_out_b);
            cur = ggml_concat(ctx0, state, cur, 1);

            
            inpL=cur;
            {
                for (int il = n_layer/2; il < n_layer; ++il) {
                    cur = build_norm_pi0(inpL,
                        model.layers[il].attn_norm, NULL,
                        LLM_NORM_RMS, il);
                    
                    // self-attention
                    {
                        // compute Q and K and RoPE them
                        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    action_steps+1);
                        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, action_steps+1);
                        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, action_steps+1);

                        Qcur = ggml_rope_ext(
                                ctx0, Qcur, inp_pos_ae, nullptr,
                                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                ext_factor, attn_factor, beta_fast, beta_slow);
                        Kcur = ggml_rope_ext(
                                ctx0, Kcur, inp_pos_ae, nullptr,
                                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                ext_factor, attn_factor, beta_fast, beta_slow);
                        
                        ggml_tensor * Kcur_old = cross_kv_tensors[2*(il-n_layer/2)];
                        ggml_tensor * Vcur_old = cross_kv_tensors[2*(il-n_layer/2)+1];

                        Kcur = ggml_concat(ctx0, Kcur_old, Kcur, 2);
                        Vcur = ggml_concat(ctx0, Vcur_old, Vcur, 2);

                        Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));

                        
                        cur = build_attn(inp_attn_ae,
                                model.layers[il].wo, NULL, nullptr,
                                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);

                    }
                    ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);

                    cur = build_norm_pi0(sa_out,
                            model.layers[il].ffn_norm, NULL,
                            LLM_NORM_RMS, il);
                    {
                        cur = build_ffn(cur,
                                model.layers[il].ffn_up,   NULL, NULL,
                                model.layers[il].ffn_gate, NULL, NULL,
                                model.layers[il].ffn_down, NULL, NULL,
                                NULL,
                                LLM_FFN_GELU, LLM_FFN_PAR, il);

                    }
                    cur = ggml_add(ctx0, cur, sa_out);
                    cur = build_cvec(cur, il);
                    inpL = cur;

                }




                cur = build_norm_pi0(cur,
                    model.ae_output_norm, NULL,
                    LLM_NORM_RMS, -1);
                ggml_tensor * action_out = (model.action_out->type == GGML_TYPE_F16)
                    ? ggml_cast(ctx0, model.action_out, GGML_TYPE_F32)
                    : model.action_out;
                cur = build_lora_mm(action_out, cur);
                if (model.action_out_b) {
                    ggml_tensor * action_out_b = (model.action_out_b->type == GGML_TYPE_F16)
                    ? ggml_cast(ctx0, model.action_out_b, GGML_TYPE_F32)
                    : model.action_out_b;
                    cur = ggml_add(ctx0, cur, action_out_b);
                }
                cur = ggml_scale(ctx0, cur, 1.0f / n_inference_steps);
            }

            if (n_unroll >= 2) {
                last_state = ggml_view_2d(ctx0, cur, hparams.action_dim, 1, cur->nb[1], 0);
                ggml_tensor * action_velocity = ggml_view_2d(
                    ctx0, cur, hparams.action_dim, action_steps, cur->nb[1], cur->nb[1]);
                actions = ggml_sub(ctx0, actions, action_velocity);
            }
        }

        res->action = n_unroll >= 2
            ? ggml_cont(ctx0, ggml_concat(ctx0, last_state, actions, 1))
            : cur;
        ggml_build_forward_expand(gf, res->action);
        
        
        return;
    }

    ggml_tensor * cur;
    ggml_tensor * inpL;

    auto cross_kv_tensors = build_inp_cross_kv_pi0();
    const bool weights_are_pi05 = model.time_mlp_in != nullptr &&
                                  model.time_mlp_out != nullptr &&
                                  model.ae_output_norm_dense != nullptr;
    const pi_model_kind pi_model = pi_model_kind_from_env();
    const bool is_pi05 = weights_are_pi05 && !pi_model_kind_is_pi0(pi_model);

    const char * pi05_debug_env = std::getenv("PI05_DEBUG_PREFIX");
    const bool keep_debug = is_pi05 && pi05_debug_env != nullptr && pi05_debug_env[0] != '\0' && std::strcmp(pi05_debug_env, "0") != 0;

    auto mark_debug = [&](ggml_tensor * t, const char * name, int il) {
        if (keep_debug && t != nullptr) {
            ggml_set_name(t, name);
            ggml_set_output(t);
        }
        cb(t, name, il);
    };

    auto mark_debug_window = [&](ggml_tensor * t, const char * name, int64_t offset, int64_t len, int il) {
        if (!keep_debug || t == nullptr || t->type != GGML_TYPE_F32 || offset < 0 || len <= 0 || offset + len > t->ne[0]) {
            return;
        }
        ggml_tensor * w = ggml_view_1d(ctx0, t, len, offset * ggml_element_size(t));
        mark_debug(w, name, il);
        ggml_build_forward_expand(gf, w);
    };

    auto pi05_dense = [&](ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) -> ggml_tensor * {
        ggml_tensor * out = build_lora_mm(w, x);
        if (b != nullptr) {
            out = ggml_add(ctx0, out, b);
        }
        return out;
    };

    ggml_tensor * state = is_pi05 ? nullptr : build_inp_state();
    ggml_tensor * actions = build_inp_action();
    std::vector<ggml_tensor *> persistent_action_nodes;
    if (is_pi05) {
        if (keep_debug) {
            mark_debug(actions, "action_in", -1);
        }
    } else {
        ggml_cont(ctx0, actions);
        cb(state, "state_in", -1);
        cb(actions, "action_in", -1);
    }

    const int64_t suffix_len = is_pi05 ? action_steps : (action_steps + 1);

    ggml_tensor * inp_pos_ae = build_inp_pos_ae(suffix_len - 1);
    ggml_tensor * rope_ff = is_pi05 ? build_inp_rope_freq_factors_pi0() : nullptr;
    auto * inp_attn_ae = build_attn_inp_no_cache_ae(suffix_len, cross_kv_tensors[0]->ne[2] + suffix_len);

    auto build_pi05_modulation = [&](ggml_tensor * cond, ggml_tensor * dense_w, ggml_tensor * dense_b, int il, const char * tag) -> std::array<ggml_tensor *, 3> {
        GGML_ASSERT(cond != nullptr);
        GGML_ASSERT(dense_w != nullptr);

        ggml_tensor * modulation = pi05_dense(dense_w, dense_b, cond);

        const int64_t dim = modulation->ne[0] / 3;
        GGML_ASSERT(modulation->ne[0] == 3 * dim);

        ggml_tensor * scale = ggml_view_2d(ctx0, modulation, dim, modulation->ne[1], modulation->nb[1], 0 * dim * modulation->nb[0]);
        ggml_tensor * shift = ggml_view_2d(ctx0, modulation, dim, modulation->ne[1], modulation->nb[1], 1 * dim * modulation->nb[0]);
        ggml_tensor * gate  = ggml_view_2d(ctx0, modulation, dim, modulation->ne[1], modulation->nb[1], 2 * dim * modulation->nb[0]);

        GGML_UNUSED(tag);
        GGML_UNUSED(il);
        return { scale, shift, gate };
    };

    auto build_pi05_norm = [&](ggml_tensor * x, ggml_tensor * norm_w, ggml_tensor * dense_w, ggml_tensor * dense_b, ggml_tensor * cond, int il, const char * tag) -> std::pair<ggml_tensor *, ggml_tensor *> {
        auto mod = build_pi05_modulation(cond, dense_w, dense_b, il, tag);
        ggml_tensor * normed = is_pi05
            ? build_norm_pi0(x, nullptr, nullptr, LLM_NORM_RMS, il)
            : build_norm_pi0(x, norm_w, nullptr, LLM_NORM_RMS, il);

        ggml_tensor * scale = mod[0];
        ggml_tensor * shift = mod[1];
        ggml_tensor * gate  = mod[2];
        if (scale->ne[1] != normed->ne[1]) {
            scale = ggml_repeat(ctx0, scale, normed);
            shift = ggml_repeat(ctx0, shift, normed);
            gate  = ggml_repeat(ctx0, gate,  normed);
        }

        ggml_tensor * scaled_normed = ggml_mul(ctx0, normed, scale);
        normed = ggml_add(ctx0, normed, scaled_normed);
        normed = ggml_add(ctx0, normed, shift);
        return { normed, gate };
    };

    auto build_pi05_gated_residual = [&](ggml_tensor * residual, ggml_tensor * branch, ggml_tensor * gate, int il, const char * tag) -> ggml_tensor * {
        ggml_tensor * gated = ggml_mul(ctx0, branch, gate);
        ggml_tensor * out = ggml_add(ctx0, residual, gated);
        GGML_UNUSED(tag);
        GGML_UNUSED(il);
        return out;
    };

    auto pi05_adarms_from_sinusoid = [&](ggml_tensor * time_expanded) -> ggml_tensor * {
        ggml_tensor * time_cond = ggml_view_2d(ctx0, time_expanded, hparams.n_embd_ae, 1, time_expanded->nb[1], 0);
        ggml_tensor * adarms_cond_raw = build_lora_mm(model.time_mlp_in, time_cond);
        adarms_cond_raw = ggml_add(ctx0, adarms_cond_raw, model.time_mlp_in_b);
        adarms_cond_raw = ggml_silu(ctx0, adarms_cond_raw);
        adarms_cond_raw = build_lora_mm(model.time_mlp_out, adarms_cond_raw);
        adarms_cond_raw = ggml_add(ctx0, adarms_cond_raw, model.time_mlp_out_b);
        adarms_cond_raw = ggml_silu(ctx0, adarms_cond_raw);
        return adarms_cond_raw;
    };

    int ae_call_count = 0;

    auto run_ae_stack = [&](ggml_tensor * actions_in, ggml_tensor * adarms_cond_in) -> ggml_tensor * {
        bool dbg_this = keep_debug && (ae_call_count == 0);
        ae_call_count++;

        if (dbg_this) {
            mark_debug(actions_in, "pi05_dbg_ae_input", -1);
            mark_debug(adarms_cond_in, "pi05_dbg_ae_adarms", -1);
        }

        ggml_tensor * cur_local = build_lora_mm(model.action_in, actions_in);
        cur_local = ggml_add(ctx0, cur_local, model.action_in_b);

        if (dbg_this) {
            mark_debug(cur_local, "pi05_dbg_ae_input_proj", -1);
        }

        ggml_tensor * inpL_local = cur_local;
        for (int il = n_layer / 2; il < n_layer; ++il) {
            bool dbg_layer_detail = dbg_this && (il == n_layer / 2);  // Q/K/V/RoPE for layer 0 only
            bool dbg_layer_summary = dbg_this;  // attn_out/ffn_out/out for ALL layers
            int dbg_li = il - n_layer / 2;
            ggml_tensor * attn_gate = nullptr;
            if (is_pi05) {
                auto norm_and_gate = build_pi05_norm(
                    inpL_local,
                    model.layers[il].attn_norm,
                    model.layers[il].attn_norm_dense,
                    model.layers[il].attn_norm_dense_b,
                    adarms_cond_in,
                    il,
                    "pi05_attn_mod");
                cur_local = norm_and_gate.first;
                attn_gate = norm_and_gate.second;
                if (dbg_layer_detail) {
                    mark_debug(ggml_cont(ctx0, cur_local), "pi05_dbg_ae_l0_attn_norm", il);
                    mark_debug(ggml_cont(ctx0, attn_gate), "pi05_dbg_ae_l0_attn_gate", il);
                }
            } else {
                cur_local = build_norm_pi0(inpL_local,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
            }

            {
                ggml_tensor * Qcur = is_pi05 ? pi05_dense(model.layers[il].wq, nullptr, cur_local) : build_lora_mm(model.layers[il].wq, cur_local);
                ggml_tensor * Kcur = is_pi05 ? pi05_dense(model.layers[il].wk, nullptr, cur_local) : build_lora_mm(model.layers[il].wk, cur_local);
                ggml_tensor * Vcur = is_pi05 ? pi05_dense(model.layers[il].wv, nullptr, cur_local) : build_lora_mm(model.layers[il].wv, cur_local);

                if (dbg_layer_detail) {
                    mark_debug(Qcur, "pi05_dbg_ae_l0_q_pre", il);
                    mark_debug(Kcur, "pi05_dbg_ae_l0_k_pre", il);
                    mark_debug(Vcur, "pi05_dbg_ae_l0_v", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    suffix_len);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, suffix_len);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, suffix_len);

                if (dbg_layer_detail) {
                    mark_debug_window(Qcur, "pi05_dbg_ae_l0_q_pre_rope_d128_135", 128, 8, il);
                    mark_debug_window(Kcur, "pi05_dbg_ae_l0_k_pre_rope_d128_135", 128, 8, il);
                }
            


                Qcur = ggml_rope_ext(
                        ctx0, Qcur, inp_pos_ae, rope_ff,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);
                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos_ae, rope_ff,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow);

                if (dbg_layer_detail) {
                    mark_debug(Qcur, "pi05_dbg_ae_l0_q_post", il);
                    mark_debug(Kcur, "pi05_dbg_ae_l0_k_post", il);
                    mark_debug_window(Qcur, "pi05_dbg_ae_l0_q_post_d128", 128, 8, il);
                    mark_debug_window(Kcur, "pi05_dbg_ae_l0_k_post_d128", 128, 8, il);
                }

                ggml_tensor * Kcur_old = cross_kv_tensors[2 * (il - n_layer / 2)];
                ggml_tensor * Vcur_old = cross_kv_tensors[2 * (il - n_layer / 2) + 1];

                Kcur = ggml_concat(ctx0, Kcur_old, Kcur, 2);
                Vcur = ggml_concat(ctx0, Vcur_old, Vcur, 2);

                Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));

                cur_local = build_attn(inp_attn_ae,
                        is_pi05 ? nullptr : model.layers[il].wo, NULL, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);
                if (is_pi05) {
                    cur_local = pi05_dense(model.layers[il].wo, nullptr, cur_local);
                }
                if (dbg_layer_summary) {
                    cur_local = ggml_cont(ctx0, cur_local);
                    { char _dn[64]; snprintf(_dn, sizeof(_dn), "pi05_dbg_ae_l%d_attn_out", dbg_li); mark_debug(cur_local, _dn, il); }
                }
            }

            ggml_tensor * sa_out = is_pi05
                ? build_pi05_gated_residual(inpL_local, cur_local, attn_gate, il, "pi05_attn_gated")
                : ggml_add(ctx0, cur_local, inpL_local);

            if (dbg_layer_summary) {
                { char _dn[64]; snprintf(_dn, sizeof(_dn), "pi05_dbg_ae_l%d_attn_res", dbg_li); mark_debug(sa_out, _dn, il); }
            }

            ggml_tensor * ffn_gate = nullptr;
            if (is_pi05) {
                auto norm_and_gate = build_pi05_norm(
                    sa_out,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_dense,
                    model.layers[il].ffn_norm_dense_b,
                    adarms_cond_in,
                    il,
                    "pi05_ffn_mod");
                cur_local = norm_and_gate.first;
                ffn_gate = norm_and_gate.second;
                if (dbg_layer_detail) {
                    mark_debug(ggml_cont(ctx0, cur_local), "pi05_dbg_ae_l0_ffn_norm", il);
                }
            } else {
                cur_local = build_norm_pi0(sa_out,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, il);
            }

            {
                if (is_pi05) {
                    ggml_tensor * ffn_up = pi05_dense(model.layers[il].ffn_up, nullptr, cur_local);
                    ggml_tensor * ffn_gate_proj = pi05_dense(model.layers[il].ffn_gate, nullptr, cur_local);
                    cur_local = ggml_geglu_split(ctx0, ffn_gate_proj, ffn_up);
                    ggml_build_forward_expand(gf, cur_local);
                    cur_local = pi05_dense(model.layers[il].ffn_down, nullptr, cur_local);
                } else {
                    cur_local = build_ffn(cur_local,
                            model.layers[il].ffn_up,   NULL, NULL,
                            model.layers[il].ffn_gate, NULL, NULL,
                            model.layers[il].ffn_down, NULL, NULL,
                            NULL,
                            LLM_FFN_GELU, LLM_FFN_PAR, il);
                }
                if (dbg_layer_summary) {
                    cur_local = ggml_cont(ctx0, cur_local);
                    { char _dn[64]; snprintf(_dn, sizeof(_dn), "pi05_dbg_ae_l%d_ffn_out", dbg_li); mark_debug(cur_local, _dn, il); }
                }
            }

            cur_local = is_pi05
                ? build_pi05_gated_residual(sa_out, cur_local, ffn_gate, il, "pi05_ffn_gated")
                : ggml_add(ctx0, cur_local, sa_out);
            cur_local = build_cvec(cur_local, il);

            if (dbg_layer_summary) {
                { char _dn[64]; snprintf(_dn, sizeof(_dn), "pi05_dbg_ae_l%d_out", dbg_li); mark_debug(cur_local, _dn, il); }
            }

            inpL_local = cur_local;
        }

        if (is_pi05) {
            auto norm_and_gate = build_pi05_norm(
                cur_local,
                model.ae_output_norm,
                model.ae_output_norm_dense,
                model.ae_output_norm_dense_b,
                adarms_cond_in,
                -1,
                "pi05_out_mod");
            cur_local = norm_and_gate.first;
        } else {
            cur_local = build_norm_pi0(cur_local,
                model.ae_output_norm, NULL,
                LLM_NORM_RMS, -1);
        }
        if (dbg_this) {
            mark_debug(cur_local, "pi05_dbg_ae_norm_out", -1);
        }
        cur_local = build_lora_mm(model.action_out, cur_local);
        if (model.action_out_b) {
            cur_local = ggml_add(ctx0, cur_local, model.action_out_b);
        }
        if (dbg_this) {
            mark_debug(cur_local, "pi05_dbg_action_out_proj", -1);
        }
        return ggml_cont(ctx0, cur_local);
    };

    const int n_unroll = (is_pi05 && cross && cross->pi0_decode_unroll >= 2)
        ? (int) cross->pi0_decode_unroll
        : 1;
    const float dt = -1.0f / (float) hparams.inference_steps;

    if (is_pi05) {
        if (n_unroll == 1) {
            ggml_tensor * time0 = build_inp_sinusoidal_embedding(0);
            if (keep_debug) {
                mark_debug(time0, "pi05_dbg_ae_time_sinusoid", -1);
            }
            ggml_tensor * adarms = pi05_adarms_from_sinusoid(time0);
            if (keep_debug) {
                mark_debug(adarms, "pi05_dbg_ae_adarms_from_sinusoid", -1);
            }
            ggml_tensor * velocity = run_ae_stack(actions, adarms);
            if (llama_pi0_persistent_decode_enabled()) {
                ggml_tensor * updated = ggml_add(ctx0, actions, ggml_scale(ctx0, velocity, dt));
                res->action = ggml_cpy(ctx0, updated, actions);
            } else {
                res->action = velocity;
            }
        } else {
            ggml_tensor * cur_actions = actions;
            for (int u = 0; u < n_unroll; ++u) {
                ggml_tensor * time_u = build_inp_sinusoidal_embedding(u);
                ggml_tensor * v_u    = run_ae_stack(cur_actions, pi05_adarms_from_sinusoid(time_u));
                ggml_tensor * updated = ggml_add(ctx0, cur_actions, ggml_scale(ctx0, v_u, dt));
                // Keep each Euler result as an internal CUDA graph value. An
                // in-place CPY to the external action tensor at every step
                // makes the generic scheduler alternate CPU/CUDA splits on
                // Thor, defeating whole-trajectory capture.
                if (llama_pi0_persistent_decode_enabled()) {
                    persistent_action_nodes.push_back(updated);
                }
                cur_actions = updated;
            }
            res->action = cur_actions;
        }
    } else {
        ggml_tensor * time_expanded = build_inp_sinusoidal_embedding(0);
        state = build_lora_mm(model.state_proj, state);
        state = ggml_add(ctx0, state, model.state_proj_b);

        cur = build_lora_mm(model.action_in, actions);
        cur = ggml_add(ctx0, cur, model.action_in_b);
        cur = ggml_concat(ctx0, cur, time_expanded, 0);
        cur = build_lora_mm(model.action_time_in, cur);
        cur = ggml_add(ctx0, cur, model.action_time_in_b);
        cur = ggml_silu(ctx0, cur);
        cur = build_lora_mm(model.action_time_out, cur);
        cur = ggml_add(ctx0, cur, model.action_time_out_b);
        cur = ggml_concat(ctx0, state, cur, 1);

        inpL = cur;
        for (int il = n_layer / 2; il < n_layer; ++il) {
            cur = build_norm_pi0(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);

            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    suffix_len);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, suffix_len);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, suffix_len);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos_ae, rope_ff,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos_ae, rope_ff,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            ggml_tensor * Kcur_old = cross_kv_tensors[2 * (il - n_layer / 2)];
            ggml_tensor * Vcur_old = cross_kv_tensors[2 * (il - n_layer / 2) + 1];
            Kcur = ggml_concat(ctx0, Kcur_old, Kcur, 2);
            Vcur = ggml_concat(ctx0, Vcur_old, Vcur, 2);
            Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));

            cur = build_attn(inp_attn_ae, model.layers[il].wo, NULL, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);

            ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);

            cur = build_norm_pi0(sa_out, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cur = ggml_add(ctx0, cur, sa_out);
            cur = build_cvec(cur, il);
            inpL = cur;
        }

        cur = build_norm_pi0(cur, model.ae_output_norm, NULL, LLM_NORM_RMS, -1);
        ggml_tensor * action_out = (model.action_out->type == GGML_TYPE_F16)
            ? ggml_cast(ctx0, model.action_out, GGML_TYPE_F32)
            : model.action_out;
        cur = build_lora_mm(action_out, cur);
        if (model.action_out_b) {
            ggml_tensor * action_out_b = (model.action_out_b->type == GGML_TYPE_F16)
                ? ggml_cast(ctx0, model.action_out_b, GGML_TYPE_F32)
                : model.action_out_b;
            cur = ggml_add(ctx0, cur, action_out_b);
        }
        cur = ggml_scale(ctx0, cur, 1.0f / n_inference_steps);
        res->action = cur;
    }

    ggml_build_forward_expand(gf, res->action);
    if (is_pi05 && llama_pi0_persistent_decode_enabled()) {
        set_tensor_backend_from_buffer(actions, actions);
        for (ggml_tensor * node : persistent_action_nodes) {
            set_tensor_backend_from_buffer(node, actions);
        }
        pin_persistent_pi_time_views();
    }
}
