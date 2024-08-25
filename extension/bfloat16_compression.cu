#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <iostream>

__device__ __forceinline__ unsigned short __bfloat162ushort(__nv_bfloat16 val) {
    return *reinterpret_cast<unsigned short*>(&val);
}

__device__ __forceinline__ __nv_bfloat16 ushort_to_bfloat16(unsigned short val) {
    return *reinterpret_cast<__nv_bfloat16*>(&val);
}

__global__ void bfloat16_compress_kernel(
    const __nv_bfloat16* __restrict__ input,
    unsigned short* __restrict__ output,
    const int input_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < input_size) {
        output[tid] = __bfloat162ushort(input[tid]);
    }
}

__global__ void bfloat16_decompress_kernel(
    const unsigned short* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int input_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < input_size) {
        output[tid] = ushort_to_bfloat16(input[tid]);
    }
}

std::tuple<torch::Tensor, torch::Tensor> compress_bfloat16(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "Input tensor must be BFloat16");

    const int input_size = input.numel();
    auto options = torch::TensorOptions().device(input.device()).dtype(torch::kInt16);
    auto output = torch::empty({input_size}, options);
    auto output_size = torch::tensor({input_size}, options.dtype(torch::kInt32));

    const int block_size = 256;
    const int grid_size = (input_size + block_size - 1) / block_size;

    bfloat16_compress_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<unsigned short*>(output.data_ptr()),
        input_size
    );

    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA error in compression: %s", cudaGetErrorString(cudaGetLastError()));

    return std::make_tuple(output, output_size);
}

torch::Tensor decompress_bfloat16(torch::Tensor input, int64_t output_size) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kInt16, "Input tensor must be Int16");

    auto options = torch::TensorOptions().device(input.device()).dtype(torch::kBFloat16);
    auto output = torch::empty({output_size}, options);

    const int block_size = 256;
    const int grid_size = (output_size + block_size - 1) / block_size;

    bfloat16_decompress_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const unsigned short*>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        static_cast<int>(output_size)
    );

    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA error in decompression: %s", cudaGetErrorString(cudaGetLastError()));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compress_bfloat16", &compress_bfloat16, "Compress BFloat16 tensor");
    m.def("decompress_bfloat16", &decompress_bfloat16, "Decompress BFloat16 tensor");
}
