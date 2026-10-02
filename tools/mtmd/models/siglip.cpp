#include "models.h"
#include "pi-model.h"

ggml_tensor * clip_graph_siglip::build_vit_pi0_legacy(ggml_tensor * inp) {
    // Jetson-PI evaluates one camera at a time and keeps the ViT activations
    // three-dimensional: [head_dim, n_head, n_patches].  Do not reuse the
    // PI05 four-dimensional batched graph here even when n_batch == 1: the
    // singleton batch dimension changes tensor strides and backend kernels.
    GGML_ASSERT(n_batch == 1);

    ggml_tensor * inpL = ggml_reshape_2d(ctx0, inp, n_embd, n_patches);

    for (int il = 0; il < n_layer; ++il) {
        const clip_layer & layer = model.layers[il];
        ggml_tensor * cur = build_norm(
            inpL, layer.ln_1_w, layer.ln_1_b,
            NORM_TYPE_NORMAL, eps, il);
        cb(cur, "layer_inp_normed", il);

        ggml_tensor * Qcur = build_mm(layer.q_w, cur);
        ggml_tensor * Kcur = build_mm(layer.k_w, cur);
        ggml_tensor * Vcur = build_mm(layer.v_w, cur);
        if (layer.q_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.q_b);
        }
        if (layer.k_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.k_b);
        }
        if (layer.v_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.v_b);
        }

        Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
        Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
        Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // Match Jetson-PI: form K^T Q first and pass 1/sqrt(head_dim) to
        // attention/softmax.  PI05 intentionally uses the other, numerically
        // non-identical ordering (scale Q before K^T Q).
        cur = build_attn(
            layer.o_w, layer.o_b,
            Qcur, Kcur, Vcur,
            nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;
        cb(cur, "ffn_inp", il);

        cur = build_norm(
            cur, layer.ln_2_w, layer.ln_2_b,
            NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ffn_inp_normed", il);
        cur = build_ffn(
            cur,
            layer.ff_up_w, layer.ff_up_b,
            nullptr, nullptr,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);
        cb(cur, "ffn_out", il);

        inpL = ggml_add(ctx0, inpL, cur);
        cb(inpL, "layer_out", il);
    }

    inpL = build_norm(
        inpL, model.post_ln_w, model.post_ln_b,
        NORM_TYPE_NORMAL, eps, -1);
    return inpL;
}

ggml_tensor * clip_graph_siglip::build_vit_pi05_legacy(ggml_tensor * inp) {
    ggml_tensor * inpL = inp;

    for (int il = 0; il < n_layer; ++il) {
        const clip_layer & layer = model.layers[il];
        ggml_tensor * cur = build_norm(
            inpL, layer.ln_1_w, layer.ln_1_b,
            NORM_TYPE_NORMAL, eps, il);
        cb(cur, "layer_inp_normed", il);

        // Q/K/V share the same normalized activation. For a BF16 ViT,
        // materialize that conversion once instead of inside all three GEMMs.
        ggml_tensor * qkv_in = cur;
        if (cur->type == GGML_TYPE_F32 &&
                layer.q_w->type == GGML_TYPE_BF16 &&
                layer.k_w->type == GGML_TYPE_BF16 &&
                layer.v_w->type == GGML_TYPE_BF16) {
            qkv_in = ggml_cast(ctx0, cur, GGML_TYPE_BF16);
        }
        ggml_tensor * Qcur = build_mm(layer.q_w, qkv_in);
        ggml_tensor * Kcur = build_mm(layer.k_w, qkv_in);
        ggml_tensor * Vcur = build_mm(layer.v_w, qkv_in);
        if (layer.q_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.q_b);
        }
        if (layer.k_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.k_b);
        }
        if (layer.v_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.v_b);
        }

        Qcur = ggml_reshape_4d(ctx0, Qcur, d_head, n_head, n_patches, n_batch);
        Kcur = ggml_reshape_4d(ctx0, Kcur, d_head, n_head, n_patches, n_batch);
        Vcur = ggml_reshape_4d(ctx0, Vcur, d_head, n_head, n_patches, n_batch);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // Match Jetson-PImerge's PI05 graph exactly: scale Q before QK and
        // leave softmax's scale at one.
        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);
        ggml_tensor * k = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
        ggml_tensor * v = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 1, 2, 0, 3));
        ggml_tensor * q_scaled = ggml_scale(ctx0, Qcur, kq_scale);
        q_scaled = ggml_permute(ctx0, q_scaled, 0, 2, 1, 3);
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q_scaled);
        kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        ggml_tensor * kqv_bthd = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        ggml_tensor * attn_out = ggml_cont_3d(
            ctx0, kqv_bthd, d_head * n_head, n_patches, n_batch);

        cur = build_mm(layer.o_w, attn_out);
        if (layer.o_b) {
            cur = ggml_add(ctx0, cur, layer.o_b);
        }
        cb(cur, "attn_out", il);

        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;
        cb(cur, "ffn_inp", il);

        cur = build_norm(
            cur, layer.ln_2_w, layer.ln_2_b,
            NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ffn_inp_normed", il);
        cur = build_ffn(
            cur,
            layer.ff_up_w, layer.ff_up_b,
            nullptr, nullptr,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);
        cb(cur, "ffn_out", il);

        inpL = ggml_add(ctx0, inpL, cur);
        cb(inpL, "layer_out", il);
    }

    inpL = build_norm(
        inpL, model.post_ln_w, model.post_ln_b,
        NORM_TYPE_NORMAL, eps, -1);
    cb(inpL, "pi05_dbg_siglip_encoder_norm", -1);
    return inpL;
}

ggml_cgraph * clip_graph_siglip::build() {
    const bool is_pi0_legacy = proj_type == PROJECTOR_TYPE_PI0 &&
        pi_model_kind_is_pi0(pi_model_kind_from_env());
    ggml_tensor * inp = nullptr;
    if (proj_type == PROJECTOR_TYPE_PI0) {
        // Preserve the original PI05 patch-embedding graph:
        //   * promote an F16 convolution kernel to F32;
        //   * convolve each camera view independently;
        //   * concatenate the resulting token tensors on the batch axis.
        // A single batched convolution is mathematically equivalent, but it
        // selects a different backend kernel and was measurably different.
        ggml_tensor * inp_raw = build_inp_raw();
        ggml_tensor * patch_kernel = model.patch_embeddings_0->type == GGML_TYPE_F16
            ? ggml_cast(ctx0, model.patch_embeddings_0, GGML_TYPE_F32)
            : model.patch_embeddings_0;

        std::vector<ggml_tensor *> patch_list;
        patch_list.reserve(n_batch);
        for (int ib = 0; ib < n_batch; ++ib) {
            ggml_tensor * image = ggml_view_3d(
                ctx0, inp_raw,
                img.nx(), img.ny(), 3,
                inp_raw->nb[1], inp_raw->nb[2],
                (int64_t) ib * inp_raw->nb[3]);
            ggml_tensor * patches = ggml_conv_2d(
                ctx0, patch_kernel, image,
                patch_size, patch_size, 0, 0, 1, 1);
            patches = ggml_reshape_2d(ctx0, patches, n_patches, n_embd);
            patches = ggml_cont(ctx0, ggml_transpose(ctx0, patches));
            if (model.patch_bias) {
                patches = ggml_add(ctx0, patches, model.patch_bias);
            }
            patches = ggml_reshape_3d(ctx0, patches, n_embd, n_patches, 1);
            patch_list.push_back(patches);
        }

        inp = patch_list[0];
        for (int ib = 1; ib < n_batch; ++ib) {
            inp = ggml_concat(ctx0, inp, patch_list[ib], 2);
        }
        if (is_pi0_legacy) {
            // Drop the singleton camera-batch axis before adding positions,
            // as in Jetson-PI's original single-image SigLIP graph.
            GGML_ASSERT(n_batch == 1);
            inp = ggml_reshape_2d(ctx0, inp, n_embd, n_patches);
        }
        cb(inp, "patch_bias", -1);
    } else {
        inp = build_inp();
    }

    ggml_tensor * learned_pos_embd = model.position_embeddings;
    if (proj_type == PROJECTOR_TYPE_LFM2 || proj_type == PROJECTOR_TYPE_PHI4) {
        learned_pos_embd = resize_position_embeddings();
    }
    if (proj_type == PROJECTOR_TYPE_PI0 && learned_pos_embd && learned_pos_embd->type != GGML_TYPE_F32) {
        learned_pos_embd = ggml_cast(ctx0, learned_pos_embd, GGML_TYPE_F32);
    }

    ggml_tensor * cur = nullptr;
    if (proj_type == PROJECTOR_TYPE_PI0) {
        if (learned_pos_embd) {
            inp = ggml_add(ctx0, inp, learned_pos_embd);
            cb(inp, "pos_embed", -1);
        }
        cur = is_pi0_legacy
            ? build_vit_pi0_legacy(inp)
            : build_vit_pi05_legacy(inp);
    } else {
        cur = build_vit(
            inp, n_patches,
            NORM_TYPE_NORMAL,
            hparams.ffn_op,
            learned_pos_embd,
            nullptr);
    }

    if (proj_type == PROJECTOR_TYPE_GEMMA3) {
        const int batch_size = 1;
        GGML_ASSERT(n_patches_x == n_patches_y);
        const int patches_per_image = n_patches_x;
        const int kernel_size = hparams.n_merge;

        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont_4d(ctx0, cur, patches_per_image, patches_per_image, n_embd, batch_size);

        // doing a pool2d to reduce the number of output tokens
        cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0] * cur->ne[0], n_embd, batch_size);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        // apply norm before projection
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);

        // apply projection
        cur = ggml_mul_mat(ctx0,
            ggml_cont(ctx0, ggml_transpose(ctx0, model.mm_input_proj_w)),
            cur);

    } else if (proj_type == PROJECTOR_TYPE_PI0) {
        cur = build_mm(model.mm_fc_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_fc_b);

    } else if (proj_type == PROJECTOR_TYPE_IDEFICS3) {
        // pixel_shuffle
        // https://github.com/huggingface/transformers/blob/0a950e0bbe1ed58d5401a6b547af19f15f0c195e/src/transformers/models/idefics3/modeling_idefics3.py#L578
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);
        cur = build_mm(model.mm_fc_w, cur);

    } else if (proj_type == PROJECTOR_TYPE_LFM2) {
        // pixel unshuffle block
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);

        // projection, in LFM2-VL input norm is optional
        if (model.mm_input_norm_w) {
            cur = ggml_norm(ctx0, cur, 1e-5); // default nn.LayerNorm
            cur = ggml_mul(ctx0, cur, model.mm_input_norm_w);
        }

        if (model.mm_input_norm_b) {
            cur = ggml_add(ctx0, cur, model.mm_input_norm_b);
        }

        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_JANUS_PRO) {
        cur = build_ffn(cur,
            model.mm_0_w, model.mm_0_b,
            nullptr, nullptr,
            model.mm_1_w, model.mm_1_b,
            hparams.ffn_op,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_PHI4) {
        cur = build_ffn(cur,
            model.mm_0_w, model.mm_0_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);

    } else {
        GGML_ABORT("SigLIP: Unsupported projector type");
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
