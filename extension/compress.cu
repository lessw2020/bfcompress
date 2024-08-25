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

    // Update total bit size
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output_size = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(output_size, s_prefix_sum[1023]);
    }
}

std::tuple<torch::Tensor, torch::Tensor> compress_bfloat16(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "Input tensor must be BFloat16");

    const int input_size = input.numel();
    const int max_output_size = (input_size * 17 + 31) / 32;  // Worst case: 17 bits per value
    auto options = torch::TensorOptions().device(input.device());
    auto output = torch::zeros({max_output_size}, options.dtype(torch::kInt32));
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

    std::cout << "Debug: input_size = " << input_size << std::endl;
    std::cout << "Debug: actual_bit_size = " << actual_bit_size << std::endl;
    std::cout << "Debug: actual_size = " << actual_size << std::endl;

    // Ensure we're not returning an empty tensor
    if (actual_size == 0) {
        std::cout << "Warning: Compression resulted in empty tensor. Returning original data." << std::endl;
        return std::make_tuple(input.view(torch::kInt16), torch::tensor({input_size * 16}, options.dtype(torch::kInt32)));
    }

    // Print first few elements of the output tensor
    std::cout << "Debug: First few elements of output tensor: ";
    for (int i = 0; i < std::min(10, actual_size); ++i) {
        std::cout << output[i].item<int>() << " ";
    }
    std::cout << std::endl;

    return std::make_tuple(output.slice(0, 0, actual_size), output_bit_size);
}
