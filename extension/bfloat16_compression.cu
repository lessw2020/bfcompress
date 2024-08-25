#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <iostream>
#include <sstream>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
               << cudaGetErrorString(error); \
            throw std::runtime_error(ss.str()); \
        } \
    } while(0)

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
    unsigned long long* __restrict__ output_size,
    const int input_size
) {
    cg::thread_block block = cg::this_thread_block();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_id = threadIdx.x;

    __shared__ unsigned short s_input[1024];
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
                          (blockIdx.x > 0 && tid > 0) ? __bfloat162ushort(input[tid - 1]) : 0;
    short delta = static_cast<short>(s_input[local_id]) - static_cast<short>(prev);
    unsigned int compressed = compress_delta(delta);
    
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
        unsigned long long write_offset = (blockIdx.x > 0) ? atomicAdd(output_size, static_cast<unsigned long long>(s_prefix_sum[1023])) : 0;
        write_offset += (local_id > 0) ? s_prefix_sum[local_id - 1] : 0;
        
        unsigned int bits_to_write = (compressed == 0) ? 1 : ((compressed & (1U << 5)) ? 17 : ((compressed & (1U << 4)) ? 9 : 5));
        unsigned long long* output64 = reinterpret_cast<unsigned long long*>(output);
        unsigned long long compressed64 = static_cast<unsigned long long>(compressed) << (write_offset % 64);
        atomicOr(&output64[write_offset / 64], compressed64);

        #ifdef DEBUG
        if (tid < 10 || (tid % 1000 == 0)) {  // Print debug info for first 10 elements and every 1000th element
            printf("Debug Compress: tid=%d, input=%hu, delta=%hd, compressed=%u, bits=%u\n", 
                   tid, __bfloat162ushort(input[tid]), delta, compressed, bits_to_write);
        }
        #endif
    }

    // Update total bit size
    if (threadIdx.x == 0) {
        atomicAdd(output_size, static_cast<unsigned long long>(s_prefix_sum[1023]));
    }
}

__global__ void bfloat16_decompress_kernel(
    const unsigned int* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const unsigned long long input_bit_size,
    const int output_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;

    unsigned long long bit_offset = 0;
    unsigned short prev = 0;

    for (int i = 0; i <= tid && bit_offset < input_bit_size; ++i) {
        unsigned long long word_offset = bit_offset / 32;
        unsigned int shift = bit_offset % 32;
        unsigned int word = input[word_offset];
        if (shift > 0 && (shift + 17) > 32) {  // We might need bits from the next word
            word |= input[word_offset + 1] << (32 - shift);
        }
        word >>= shift;

        unsigned int prefix = word & 0x7;
        bit_offset += 3;

        short delta;
        if (prefix == 0) {
            delta = 0;
            bit_offset += 1;
        } else if (prefix <= 3) {
            delta = (short)((word >> 3) & 0xF);
            if (delta & 0x8) delta |= 0xFFF0;
            bit_offset += 4;
        } else if (prefix <= 5) {
            delta = (short)((word >> 3) & 0xFF);
            if (delta & 0x80) delta |= 0xFF00;
            bit_offset += 8;
        } else {
            delta = (short)((word >> 3) & 0xFFFF);
            bit_offset += 16;
        }

        unsigned short current = prev + delta;
        if (i == tid) {
            output[i] = ushort_to_bfloat16(current);
            #ifdef DEBUG
            printf("Debug Decompress: tid=%d, prev=%hu, delta=%hd, current=%hu, output=%hu\n", 
                   tid, prev, delta, current, __bfloat162ushort(output[i]));
            #endif
            return;
        }
        prev = current;
    }
}

std::tuple<torch::Tensor, torch::Tensor> compress_bfloat16(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "Input tensor must be BFloat16");

    const int input_size = input.numel();
    const int max_output_size = (input_size * 17 + 63) / 64;  // Worst case: 17 bits per value, rounded up to 64-bit words
    auto options = torch::TensorOptions().device(input.device());
    auto output = torch::zeros({max_output_size}, options.dtype(torch::kInt64));
    auto output_bit_size = torch::zeros({1}, options.dtype(torch::kInt64));

    const int block_size = 1024;
    const int grid_size = (input_size + block_size - 1) / block_size;

    bfloat16_compress_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<unsigned int*>(output.data_ptr()),
        reinterpret_cast<unsigned long long*>(output_bit_size.data_ptr()),
        input_size
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long actual_bit_size = output_bit_size.item<unsigned long long>();
    int actual_size = (actual_bit_size + 63) / 64;

    std::cout << "Debug: input_size = " << input_size << std::endl;
    std::cout << "Debug: actual_bit_size = " << actual_bit_size << std::endl;
    std::cout << "Debug: actual_size = " << actual_size << std::endl;

    if (actual_size == 0) {
        std::cout << "Warning: Compression resulted in empty tensor. Returning original data." << std::endl;
        return std::make_tuple(input.view(torch::kInt16), torch::tensor({input_size * 16}, options.dtype(torch::kInt64)));
    }

    #ifdef DEBUG
    std::cout << "Debug: First few elements of output tensor: ";
    for (int i = 0; i < std::min(10, actual_size); ++i) {
        std::cout << output[i].item<int64_t>() << " ";
    }
    std::cout << std::endl;
    #endif

    return std::make_tuple(output.slice(0, 0, actual_size), output_bit_size);
}

torch::Tensor decompress_bfloat16(torch::Tensor input, torch::Tensor bit_size, int64_t output_size) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kInt64, "Input tensor must be Int64");

    auto options = torch::TensorOptions().device(input.device()).dtype(torch::kBFloat16);
    auto output = torch::empty({output_size}, options);

    const int block_size = 1024;
    const int grid_size = (output_size + block_size - 1) / block_size;

    bfloat16_decompress_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const unsigned int*>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        bit_size.item<unsigned long long>(),
        static_cast<int>(output_size)
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compress_bfloat16", [](torch::Tensor input) {
        try {
            return compress_bfloat16(input);
        } catch (const std::exception& e) {
            throw pybind11::error_already_set();
        }
    }, "Compress BFloat16 tensor");

    m.def("decompress_bfloat16", [](torch::Tensor input, torch::Tensor bit_size, int64_t output_size) {
        try {
            return decompress_bfloat16(input, bit_size, output_size);
        } catch (const std::exception& e) {
            throw pybind11::error_already_set();
        }
    }, "Decompress BFloat16 tensor");
}
