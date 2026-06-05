#include "../tools/server/server-context.h"

#undef NDEBUG
#include <cassert>

int main() {
    assert(server_context_n_ctx_for_internal_seqs(
            /*n_ctx =*/ 32768,
            /*n_parallel_user =*/ 1,
            /*n_seq_max_full =*/ 2,
            /*kv_unified_effective =*/ false) == 65536);
    assert(server_context_n_ctx_for_internal_seqs(
            /*n_ctx =*/ 32768,
            /*n_parallel_user =*/ 1,
            /*n_seq_max_full =*/ 2,
            /*kv_unified_effective =*/ true) == 32768);
    assert(server_context_n_ctx_for_internal_seqs(
            /*n_ctx =*/ 32768,
            /*n_parallel_user =*/ 2,
            /*n_seq_max_full =*/ 4,
            /*kv_unified_effective =*/ false) == 65536);
    assert(server_context_n_ctx_for_internal_seqs(
            /*n_ctx =*/ 32768,
            /*n_parallel_user =*/ 2,
            /*n_seq_max_full =*/ 2,
            /*kv_unified_effective =*/ false) == 32768);
    assert(server_context_n_ctx_for_internal_seqs(
            /*n_ctx =*/ 0,
            /*n_parallel_user =*/ 1,
            /*n_seq_max_full =*/ 2,
            /*kv_unified_effective =*/ false) == 0);

    return 0;
}
