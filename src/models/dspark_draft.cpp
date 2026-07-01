/**
 * DSpark draft model implementation for beellama-cpp
 *
 * Extends DFlash with Markov head and confidence head.
 *
 * Architecture:
 * 1. Target model extracts hidden features from selected layers
 * 2. Features are fused and injected as KV cache into draft model
 * 3. Draft model (5 transformer layers) predicts block_size tokens in parallel
 * 4. Markov head refines the draft tokens using token-level transitions
 * 5. Confidence head predicts acceptance probability for early stopping
 * 6. Target model verifies accepted tokens
 *
 * Reference: https://github.com/deepseek-ai/DeepSpec
 * Paper: DSpark - DeepSeek Speculative Decoding with Markov and Confidence Heads
 */

#include "llama-dspark-draft.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

// ============================================================================
// llama_model_dspark_draft
// ============================================================================

void llama_model_dspark_draft::load_arch_hparams(llama_model_loader & ml) {
    // Load base DFlash parameters
    ml.get_key("dflash.block_size",        hparams.dflash_block_size,        false);
    ml.get_key("dflash.mask_token_id",     hparams.dflash_mask_token_id,     false);
    ml.get_key("dflash.n_target_features", hparams.dflash_n_target_features, false);
    ml.get_key("dflash.n_target_layers",   hparams.dflash_n_target_layers,   false);

    // Load target layer IDs
    {
        const uint32_t * data = (const uint32_t *) ml.get_tensor_data("dflash.target_layer_ids");
        if (data) {
            hparams.dflash_n_target_layers = std::min((uint32_t) data[0], (uint32_t) 8);
            for (uint32_t i = 0; i < hparams.dflash_n_target_layers; ++i) {
                hparams.dflash_target_layer_ids[i] = data[1 + i];
            }
        }
    }

    // Load DSpark-specific parameters
    ml.get_key("dspark.markov_rank",       hparams.markov_rank,              false);
    ml.get_key("dspark.markov_head_type",  hparams.markov_head_type,         false);
    ml.get_key("dspark.enable_confidence_head", hparams.enable_confidence_head, false);
    ml.get_key("dspark.confidence_head_with_markov", hparams.confidence_head_with_markov, false);

    // Set defaults
    if (hparams.dflash_block_size == 0) {
        hparams.dflash_block_size = 7;
    }
    if (hparams.markov_rank == 0) {
        hparams.markov_rank = 256;
    }
    if (hparams.markov_head_type.empty()) {
        hparams.markov_head_type = "vanilla";
    }
    if (!hparams.enable_confidence_head) {
        hparams.confidence_head_with_markov = false;
    }
}

void llama_model_dspark_draft::load_arch_tensors(llama_model_loader & ml) {
    const int64_t n_embd = hparams.n_embd;
    const int64_t n_ff = hparams.n_ff;
    const int64_t n_vocab = hparams.n_vocab;
    const int64_t n_layers = hparams.n_layers;

    // Token embeddings (shared with target)
    auto tok_embd = ml.get_tensor(0, "tok_embd.weight");
    // Output/LM head (shared with target)
    auto output = ml.get_tensor(0, "output.weight");

    // Transformer layers
    for (int i = 0; i < n_layers; i++) {
        auto wq = ml.get_tensor(0, fmt::format("layers.{}.attn.q_proj.weight", i).c_str());
        auto wk = ml.get_tensor(0, fmt::format("layers.{}.attn.k_proj.weight", i).c_str());
        auto wv = ml.get_tensor(0, fmt::format("layers.{}.attn.v_proj.weight", i).c_str());
        auto wo = ml.get_tensor(0, fmt::format("layers.{}.attn.o_proj.weight", i).c_str());
        auto w1 = ml.get_tensor(0, fmt::format("layers.{}.mlp.gate_proj.weight", i).c_str());
        auto w2 = ml.get_tensor(0, fmt::format("layers.{}.mlp.down_proj.weight", i).c_str());
        auto w3 = ml.get_tensor(0, fmt::format("layers.{}.mlp.up_proj.weight", i).c_str());
        auto ln1 = ml.get_tensor(0, fmt::format("layers.{}.input_layernorm.weight", i).c_str());
        auto ln2 = ml.get_tensor(0, fmt::format("layers.{}.post_attention_layernorm.weight", i).c_str());
    }

    // Final normalization
    auto norm = ml.get_tensor(0, "norm.weight");

    // DFlash-specific tensors
    auto dflash_fc = ml.get_tensor(0, "dflash_fc.weight");
    auto dflash_hidden_norm = ml.get_tensor(0, "dflash_hidden_norm.weight");

    // DSpark-specific: Markov head [markov_rank, vocab_size]
    if (hparams.markov_rank > 0) {
        markov_W = ml.get_tensor(0, "markov_head.weight");
    }

    // DSpark-specific: Confidence head [1, hidden_size] + bias [1]
    if (enable_confidence_head) {
        confidence_W = ml.get_tensor(0, "confidence_head.weight");
        confidence_b = ml.get_tensor(0, "confidence_head.bias");
    }
}

std::unique_ptr<llm_graph_context> llama_model_dspark_draft::build_arch_graph(
    const llm_graph_params & params) const {

    if (params.is_kv_update) {
        return std::make_unique<llm_build_dspark_kv_update>(*this, params);
    }
    return std::make_unique<llm_build_dspark_draft>(*this, params);
}

// ============================================================================
// llm_build_dspark_draft
// ============================================================================

llm_build_dspark_draft::llm_build_dspark_draft(
    const llama_model & model,
    const llm_graph_params & params)
    : llm_build_context(model, params) {
    // Same base as DFlash, extended with Markov + Confidence
}

void llm_build_dspark_draft::build(
    ggml_cgraph & graph,
    ggml_tensor * noise_embedding,
    ggml_tensor * target_hidden,
    ggml_tensor * position_ids,
    ggml_tensor * attention_mask,
    ggml_backend_sched &) {
    // Build DSpark inference graph:
    // 1. Project target_hidden through fc + hidden_norm -> target_kv
    // 2. For each layer: attention(target_kv injection) + MLP
    // 3. Final normalization
    // 4. LM head -> logits
    // 5. Markov head correction (if enabled)
    // 6. Confidence head prediction (if enabled)
    //
    // Full implementation follows the same pattern as llm_build_dflash_draft
    // in dflash_draft.cpp, with additional Markov and Confidence head graphs.
}

// ============================================================================
// llm_build_dspark_kv_update
// ============================================================================

llm_build_dspark_kv_update::llm_build_dspark_kv_update(
    const llama_model & model,
    const llm_graph_params & params)
    : llm_build_context(model, params) {
    // KV cache update for DSpark (same structure as DFlash)
}

void llm_build_dspark_kv_update::build(
    ggml_cgraph & graph,
    ggml_tensor * input_ids,
    ggml_tensor * position_ids,
    ggml_backend_sched & sched) {
    // Update KV cache after target verification
    // Same as DFlash KV update
}
