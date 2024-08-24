#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <iostream>

namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned short __bfloat162ushort(__nv_bfloat16 val) {
    return *reinterpret_cast<unsigned short*>(&val);
}

__device__ __forceinline__ __nv_bfloat16 ushort_to_bfloat16(unsigned short val) {
    return *reinterpret_cast<__nv_bfloat16*>(&val);
}

__device__ void write_bits(unsigned int* buffer, int& bit_pos, unsigned int value, int num_bits) {
    int word_idx = bit_pos / 32;
    int bit_offset = bit_pos % 32;
    unsigned int mask = (1U << num_bits) - 1;
    value &= mask;

    atomicOr(&buffer[word_idx], value << bit_offset);
    if (bit_offset + num_bits > 32) {
        atomicOr(&buffer[word_idx + 1], value >> (32 - bit_offset));
    }

    bit_pos += num_bits;
}

__device__ unsigned int read_bits(const unsigned int* buffer, int& bit_pos, int num_bits) {
    int word_idx = bit_pos / 32;
    int bit_offset = bit_pos % 32;
    unsigned int value = buffer[word_idx] >> bit_offset;
    if (bit_offset + num_bits > 32) {
        value |= buffer[word_idx + 1] << (32 - bit_offset);
    }
    value &= (1U << num_bits) - 1;
    bit_pos += num_bits;
    return value;
}

__global__ void bfloat16_compress_kernel(
    const __nv_bfloat16* __restrict__ input,
    unsigned int* __restrict__ output,
    int* __restrict__ output_bit_size,
    const int input_size
) {
    cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    __shared__ int shared_bit_pos;
    if (threadIdx.x == 0) {
        shared_bit_pos = 0;
    }
    __syncthreads();

    for (int i = tid; i < input_size; i += stride) {
        unsigned short current = __bfloat162ushort(input[i]);
        unsigned short prev = (i > 0) ? __bfloat162ushort(input[i-1]) : 0;
        short delta = current - prev;

        int num_bits;
        if (delta == 0) num_bits = 1;
        else if (delta >= -8 && delta <= 7) num_bits = 5;
        else if (delta >= -128 && delta <= 127) num_bits = 9;
        else num_bits = 17;

        int local_bit_pos = atomicAdd(&shared_bit_pos, num_bits);
        __syncthreads();

        if (num_bits == 1) {
            write_bits(output, local_bit_pos, 0, 1);
        } else if (num_bits == 5) {
            write_bits(output, local_bit_pos, 0b10, 2);
            write_bits(output, local_bit_pos, delta & 0xF, 3);
        } else if (num_bits == 9) {
            write_bits(output, local_bit_pos, 0b110, 3);
            write_bits(output, local_bit_pos, delta & 0xFF, 6);
        } else {
            write_bits(output, local_bit_pos, 0b111, 3);
            write_bits(output, local_bit_pos, delta & 0xFFFF, 14);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(output_bit_size, shared_bit_pos);
    }

    grid.sync();

    if (grid.thread_rank() == 0) {
        *output_bit_size = cg::reduce(grid, *output_bit_size, cg::plus<int>());
    }
}

__global__ void bfloat16_decompress_kernel(
    const unsigned int* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int input_bit_size,
    const int output_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bit_pos = 0;

    unsigned short prev = 0;
    for (int i = 0; i < output_size && bit_pos < input_bit_size; ++i) {
        unsigned int prefix = read_bits(input, bit_pos, 3);
        short delta;

        if (prefix == 0) {
            delta = 0;
        } else if ((prefix & 0b110) == 0b10) {
            delta = read_bits(input, bit_pos, 3);
            if (delta & 0b100) delta |= 0xFFF8;  // Sign extend
        } else if (prefix == 0b110) {
            delta = read_bits(input, bit_pos, 6);
            if (delta & 0b100000) delta |= 0xFF80;  // Sign extend
        } else {
            delta = read_bits(input, bit_pos, 14);
            if (delta & 0b10000000000000) delta |= 0xC000;  // Sign extend
        }

        unsigned short current = prev + delta;
        if (i == tid) {
            output[i] = ushort_to_bfloat16(current);
        }
        prev = current;
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

    const int block_size = 256;
    const int grid_size = std::min(65535, (input_size + block_size - 1) / block_size);

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

    const int block_size = 256;
    const int grid_size = std::min(65535, (output_size + block_size - 1) / block_size);

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
