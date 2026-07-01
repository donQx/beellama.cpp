#pragma once

/**
 * DSpark draft model for beellama-cpp
 *
 * DSpark draft model extension:
 * - 5 transformer layers (same as DFlash)
 * - Reuses target model embeddings + LM head
 * - Conditions on target model hidden features (KV injection)
 * - Block diffusion: predicts block_size tokens in parallel
 * - Markov head refines draft tokens using token-level transitions
 * - Confidence head predicts acceptance probability for early stopping
 *
 * Reference: https://github.com/deepseek-ai/DeepSpec
 */

#include "llama.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"

#include <vector>
#include <memory>
#include <string>

// DSpark model architecture enum (must be after LLM_ARCH_DFLASH)
enum llama_arch {
    LLM_ARCH_DSPARK,
};

// DSpark hyperparameters (extends llama_hparams)
struct llama_hparams_dspark {
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t n_layers;
    uint32_t n_vocab;
    uint32_t block_size;
    uint32_t mask_token_id;

    // DSpark-specific
    uint32_t markov_rank;
    std::string markov_head_type;
    bool enable_confidence_head;
    bool confidence_head_with_markov;

    // DFlash base (inherited)
    uint32_t dflash_block_size;
    uint32_t dflash_mask_token_id;
    uint32_t dflash_n_target_features;
    uint32_t dflash_n_target_layers;
    uint32_t dflash_target_layer_ids[8];
};

// DSpark model class
class llama_model_dspark_draft : public llama_model {
public:
    llama_arch get_arch() const override { return LLM_ARCH_DSPARK; }

    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;
    std::unique_ptr<llm_graph_context> build_arch_graph(
        const llm_graph_params & params) const override;

    // DSpark-specific members
    uint32_t markov_rank;
    std::string markov_head_type;
    bool enable_confidence_head;
    bool confidence_head_with_markov;

    // Markov head tensor: [markov_rank, vocab_size]
    std::unique_ptr<ggml_tensor> markov_W;
    // Confidence head: [1, hidden_size] + bias [1]
    std::unique_ptr<ggml_tensor> confidence_W;
    std::unique_ptr<ggml_tensor> confidence_b;
};

// DSpark draft graph builder
class llm_build_dspark_draft : public llm_build_context {
public:
    llm_build_dspark_draft(
        const llama_model & model,
        const llm_graph_params & params);

    void build(
        ggml_cgraph & graph,
        ggml_tensor * noise_embedding,
        ggml_tensor * target_hidden,
        ggml_tensor * position_ids,
        ggml_tensor * attention_mask,
        ggml_backend_sched & sched);
};

// DSpark KV cache update graph builder
class llm_build_dspark_kv_update : public llm_build_context {
public:
    llm_build_dspark_kv_update(
        const llama_model & model,
        const llm_graph_params & params);

    void build(
        ggml_cgraph & graph,
        ggml_tensor * input_ids,
        ggml_tensor * position_ids,
        ggml_backend_sched & sched);
};
