#include <torch/extension.h>
#include <vector>
#include <tuple>
#include "utils.h"

namespace grouped_gemm {
// Computes per-chunk cumsums and per-chunk token totals from a 1-D
// `tokens_per_expert` CPU tensor, entirely in C++ with zero Python overhead.
//
// Returns:
//   chunk_cumsums  – vector of int32 tensors (one per chunk), each the local
//                    cumulative sum within that chunk, already on `device`.
//   chunk_totals   – int64 vector of the total token count in each chunk.
//
// This replaces:
//   tokens_per_expert_chunks      = torch.split(tokens_per_expert, chunk_size)
//   tokens_per_expert_chunks_psum = [cumsum(c).int().to(device) for c in ...]
//   total_token_num_per_chunk     = [c.sum().item() for c in ...]

std::tuple<std::vector<torch::Tensor>, std::vector<int64_t>>
TokensPerExpertChunkSum(
    const torch::Tensor& tokens_per_expert, // 1-D, CPU, any integer dtype
    int64_t chunk_size,
    torch::Device device)
{
    TORCH_CHECK(tokens_per_expert.dim() == 1, "tokens_per_expert must be 1-D");
    TORCH_CHECK(tokens_per_expert.is_cpu(), "tokens_per_expert must be a CPU tensor");

    const int64_t n = tokens_per_expert.size(0);
    TORCH_CHECK(chunk_size > 0 && chunk_size <= n,
                "chunk_size must be in [1, tokens_per_expert.numel()]");

    // Work in int64 for accumulation safety, then narrow to int32 at the end.
    auto tpe = tokens_per_expert.to(torch::kInt64).contiguous();
    const int64_t* src = tpe.data_ptr<int64_t>();

    const int64_t num_chunks = (n + chunk_size - 1) / chunk_size;

    std::vector<torch::Tensor> chunk_cumsums;
    std::vector<int64_t> chunk_totals;
    chunk_cumsums.reserve(num_chunks);
    chunk_totals.reserve(num_chunks);

    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const int64_t start = chunk_idx * chunk_size;
        const int64_t end   = std::min(start + chunk_size, n);
        const int64_t len   = end - start;

        // Allocate the output chunk directly as int32.
        auto out = torch::empty({len}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
        int32_t* dst = out.data_ptr<int32_t>();

        int64_t running = 0;
        for (int64_t i = 0; i < len; ++i) {
            running += src[start + i];
            dst[i] = static_cast<int32_t>(running);
        }

        chunk_totals.push_back(running);

        // Async copy to target device (non-blocking for CUDA).
        chunk_cumsums.push_back(out.to(device, /*non_blocking=*/true));
    }

    return {std::move(chunk_cumsums), std::move(chunk_totals)};
}

}