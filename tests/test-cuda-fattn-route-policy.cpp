#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

static std::string read_file(const std::string & path) {
    std::ifstream file(path);
    if (!file.good()) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(1);
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static bool expect(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
    }
    return ok;
}

static std::string slice_between(const std::string & text, const std::string & begin, const std::string & end) {
    const size_t b = text.find(begin);
    if (b == std::string::npos) {
        return {};
    }
    const size_t e = text.find(end, b);
    if (e == std::string::npos) {
        return text.substr(b);
    }
    return text.substr(b, e - b);
}

int main(int argc, char ** argv) {
    bool ok = true;

    ok &= expect(argc == 2, "expected repo root argument");
    if (!ok) {
        return 1;
    }

    const std::string root = argv[1];
    const std::string fattn = read_file(root + "/ggml/src/ggml-cuda/fattn.cu");
    const std::string helper = slice_between(fattn,
            "static inline bool ggml_cuda_fattn_prefers_native_vec_for_turbo_k_classic_v",
            "struct ggml_cuda_fattn_route_plan");
    const std::string planner = slice_between(fattn,
            "static ggml_cuda_fattn_route_plan ggml_cuda_fattn_make_route_plan",
            "size_t ggml_cuda_flash_attn_ext_get_alloc_size");
    const std::string prefill = slice_between(fattn,
            "static void ggml_cuda_turbo_prefill_attend",
            "#define FATTN_VEC_CASE");
    const std::string prefill_policy = slice_between(fattn,
            "static inline bool ggml_cuda_fattn_prefill_mma_can_materialize_turbo_k_classic_v",
            "// Shape guard for the effective K/V pair after Turbo V decode-dequant.");
    const std::string exec = slice_between(fattn,
            "void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst)",
            "bool ggml_cuda_flash_attn_ext_support");

    ok &= expect(!helper.empty(),
        "CUDA FA routing must have an explicit native vec policy for Turbo K + classic V pairs");
    ok &= expect(helper.find("ggml_cuda_fattn_is_turbo_kv_type(K->type)") != std::string::npos &&
                 helper.find("ggml_cuda_fattn_is_turbo_kv_type(V->type)") != std::string::npos,
        "native vec policy must distinguish Turbo K from Turbo V");
    ok &= expect(helper.find("ggml_cuda_fattn_is_classic_non_q8_type(V->type)") != std::string::npos,
        "native vec policy must be limited to classic non-q8 V types");
    ok &= expect(helper.find("ggml_cuda_fattn_pair_compiled(K->type, V->type)") != std::string::npos,
        "native vec policy must require a compiled raw K/V vec pair");
    ok &= expect(helper.find("Q->ne[0] <= 512") != std::string::npos &&
                 helper.find("Q->ne[0] % 64 == 0") != std::string::npos,
        "native vec policy must preserve the vec kernel head-dimension guard");

    ok &= expect(planner.find("const bool prefer_native_vec =") != std::string::npos &&
                 planner.find("ggml_cuda_fattn_prefers_native_vec_for_turbo_k_classic_v(Q, K, V)") != std::string::npos,
        "route planner must compute the Turbo K + classic V native vec preference");
    ok &= expect(planner.find("!prefer_native_vec &&") != std::string::npos,
        "decode-dequant policy must not override preferred native vec Turbo K + classic V pairs");

    ok &= expect(prefill.find("classic_non_q8_v") != std::string::npos &&
                 prefill.find("ggml_cuda_fattn_materialize_to_f16(V, v_fp16, stream, V_f16)") != std::string::npos,
        "Turbo K + classic non-q8 V prefill must materialize classic V to f16 for the MMA path");
    ok &= expect(exec.find("turbo_k_classic_v_prefill") != std::string::npos &&
                 exec.find("ggml_cuda_fattn_prefill_mma_can_materialize_turbo_k_classic_v(K, V)") != std::string::npos,
        "Turbo K + classic non-q8 V batch prefill must be eligible for the prefill MMA path");
    ok &= expect(exec.find("Q->ne[0] <= 512") != std::string::npos,
        "Turbo K + classic non-q8 V batch prefill must cover Gemma D512 layers");
    ok &= expect(prefill_policy.find("ggml_cuda_fattn_is_turbo_kv_type(K->type) &&") != std::string::npos &&
                 prefill_policy.find("!ggml_cuda_fattn_is_turbo_kv_type(V->type)") != std::string::npos &&
                 prefill_policy.find("ggml_cuda_fattn_is_classic_non_q8_type(V->type)") != std::string::npos,
        "Turbo K + classic V prefill eligibility must not broaden classic-K/Turbo-V routing");

    return ok ? 0 : 1;
}
