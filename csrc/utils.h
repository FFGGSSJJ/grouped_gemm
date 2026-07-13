#include <torch/extension.h>
#include <vector>
#include <tuple>

namespace grouped_gemm {

std::tuple<std::vector<torch::Tensor>, std::vector<int64_t>>
TokensPerExpertChunkSum(
    const torch::Tensor& tokens_per_expert, // 1-D, CPU, any integer dtype
    int64_t chunk_size,
    torch::Device device);

}