#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <torch/extension.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned short __bfloat162ushort(__nv_bfloat16 val) {
    return *reinterpret_cast<unsigned short*>(&val);
}

__device__ __forceinline__ __nv_bfloat16 ushort_to_bfloat16(unsigned short val) {
    return *reinterpret_cast<__nv_bfloat16*>(&val);
}

// Compress small deltas into fewer bits
__device__ __forceinline__ unsigned int compress_delta(short delta) {
    unsigned int compressed;
    if (delta == 0) {
        compressed = 0;
    } else if (delta >= -8 && delta <= 7) {
        compressed = (1U << 3) | (delta & 0x0F);
    } else if (delta >= -128 && delta <= 127) {
        compressed = (1U << 4) | (delta & 0xFF);
    } else {
        compressed = (1U << 5) | (delta & 0xFFFF);
    }
    return compressed;
}

// Decompress the compressed delta
__device__ __forceinline__ short decompress_delta(unsigned int compressed) {
    if ((compressed & (1U << 3)) == 0) {
        return 0;
    } else if ((compressed & (1U << 4)) == 0) {
        return (compressed & 0x0F) | ((compressed & 0x08) ? 0xFFF0 : 0);
    } else if ((compressed & (1U << 5)) == 0) {
        return (compressed & 0xFF) | ((compressed & 0x80) ? 0xFF00 : 0);
    } else {
        return compressed & 0xFFFF;
    }
}

__global__ void bfloat16_compress_kernel(
    const __nv_bfloat16* __restrict__ input,
    unsigned int* __restrict__ output,
    int* __restrict__ output_size,
    const int input_size
) {
    cg::thread_block block = cg::this_thread_block();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_id = threadIdx.x;

    __shared__ unsigned short s_input[1024];
    __shared__ unsigned int s_compressed[1024];
    __shared__ unsigned int s_prefix_sum[1024];

    // Load input into shared memory
    if (tid < input_size) {
        s_input[local_id] = __bfloat162ushort(input[tid]);
    } else {
        s_input[local_id] = 0;
    }
    block.sync();

    // Compute deltas and compress
    unsigned short prev = (local_id > 0) ? s_input[local_id - 1] :
                          (blockIdx.x > 0 ? __bfloat162ushort(input[tid - 1]) : 0);
    short delta = s_input[local_id] - prev;
    unsigned int compressed = compress_delta(delta);
    s_compressed[local_id] = compressed;

    // Compute prefix sum of compressed sizes
    unsigned int size = (compressed == 0) ? 1 : ((compressed & (1U << 5)) ? 17 : ((compressed & (1U << 4)) ? 9 : 5));
    s_prefix_sum[local_id] = size;
    block.sync();

    // Parallel prefix sum
    for (int stride = 1; stride < 1024; stride *= 2) {
        unsigned int n = 0;
        if (local_id >= stride) {
            n = s_prefix_sum[local_id - stride];
        }
        block.sync();
        if (local_id >= stride) {
            s_prefix_sum[local_id] += n;
        }
        block.sync();
    }

    // Write compressed data
    if (tid < input_size) {
        unsigned int write_offset = (blockIdx.x > 0) ? atomicAdd(output_size, s_prefix_sum[1023]) : 0;
        write_offset += (local_id > 0) ? s_prefix_sum[local_id - 1] : 0;

        unsigned int bits_to_write = (compressed == 0) ? 1 : ((compressed & (1U << 5)) ? 17 : ((compressed & (1U << 4)) ? 9 : 5));
        unsigned int word_offset = write_offset / 32;
        unsigned int bit_offset = write_offset % 32;

        atomicOr(&output[word_offset], compressed << bit_offset);
        if (bit_offset + bits_to_write > 32) {
            atomicOr(&output[word_offset + 1], compressed >> (32 - bit_offset));
        }
    }
}

__global__ void bfloat16_decompress_kernel(
    const unsigned int* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int input_bit_size,
    const int output_size
) {
    cg::thread_block block = cg::this_thread_block();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_id = threadIdx.x;

    __shared__ unsigned short s_decompressed[1024];

    unsigned int bit_offset = blockIdx.x * 1024 * 5;  // Assuming minimum 5 bits per value on average
    unsigned short prev = (blockIdx.x > 0) ? __bfloat162ushort(output[blockIdx.x * 1024 - 1]) : 0;

    while (bit_offset < input_bit_size && local_id + blockIdx.x * 1024 < output_size) {
        unsigned int word_offset = bit_offset / 32;
        unsigned int word_bit_offset = bit_offset % 32;

        unsigned int compressed = (input[word_offset] >> word_bit_offset) |
                                  (input[word_offset + 1] << (32 - word_bit_offset));

        short delta = decompress_delta(compressed);
        unsigned short value = prev + delta;
        s_decompressed[local_id] = value;

        unsigned int bits_read = (compressed == 0) ? 1 : ((compressed & (1U << 5)) ? 17 : ((compressed & (1U << 4)) ? 9 : 5));
        bit_offset += bits_read;
        prev = value;

        block.sync();

        if (local_id + blockIdx.x * 1024 < output_size) {
            output[local_id + blockIdx.x * 1024] = ushort_to_bfloat16(s_decompressed[local_id]);
        }

        block.sync();
    }
}

std::tuple<torch::Tensor, torch::Tensor> compress_bfloat16(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "Input tensor must be BFloat16");

    const int input_size = input.numel();
    const int max_output_size = (input_size * 17 + 31) / 32;  // Worst case: 17 bits per value
    auto options = torch::TensorOptions().device(input.device());
    auto output = torch::empty({max_output_size}, options.dtype(torch::kInt32));
    auto output_bit_size = torch::zeros({1}, options.dtype(torch::kInt32));

    const int block_size = 1024;
    const int grid_size = (input_size + block_size - 1) / block_size;

    bfloat16_compress_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<unsigned int*>(output.data_ptr()),
        reinterpret_cast<int*>(output_bit_size.data_ptr()),
        input_size
    );

    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));

    int actual_bit_size = output_bit_size.item<int>();
    int actual_size = (actual_bit_size + 31) / 32;
    return std::make_tuple(output.slice(0, 0, actual_size), output_bit_size);
}

torch::Tensor decompress_bfloat16(torch::Tensor input, torch::Tensor bit_size, int64_t output_size) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kInt32, "Input tensor must be Int32");

    auto options = torch::TensorOptions().device(input.device()).dtype(torch::kBFloat16);
    auto output = torch::empty({output_size}, options);

    const int block_size = 1024;
    const int grid_size = (output_size + block_size - 1) / block_size;

    bfloat16_decompress_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const unsigned int*>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        bit_size.item<int>(),
        static_cast<int>(output_size)
    );

    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA error: %s", cudaGetErrorString(cudaGetLastError()));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compress_bfloat16", &compress_bfloat16, "Compress BFloat16 tensor");
    m.def("decompress_bfloat16", &decompress_bfloat16, "Decompress BFloat16 tensor");
}
