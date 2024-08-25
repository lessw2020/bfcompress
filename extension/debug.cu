std::tuple<torch::Tensor, torch::Tensor> compress_bfloat16(torch::Tensor input) {
    // ... (previous code remains the same)

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

    // Print first few elements of the output tensor
    std::cout << "Debug: First few elements of output tensor: ";
    for (int i = 0; i < std::min(10, actual_size); ++i) {
        std::cout << output[i].item<int>() << " ";
    }
    std::cout << std::endl;

    return std::make_tuple(output.slice(0, 0, actual_size), output_bit_size);
    }
