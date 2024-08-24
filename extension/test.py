import torch
import bfloat16_compression
import time
import numpy as np

def benchmark_compression(input_size, num_iterations=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA is not available. Skipping benchmark.")
        return

    # Generate random input data
    input_tensor = torch.randn(input_size, dtype=torch.bfloat16, device=device)

    # Warm-up run
    compressed_tensor, bit_size = bfloat16_compression.compress_bfloat16(input_tensor)
    decompressed_tensor = bfloat16_compression.decompress_bfloat16(compressed_tensor, bit_size, input_size)

    # Compression benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        compressed_tensor, bit_size = bfloat16_compression.compress_bfloat16(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    compression_time = (end_time - start_time) / num_iterations

    # Decompression benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        decompressed_tensor = bfloat16_compression.decompress_bfloat16(compressed_tensor, bit_size, input_size)
    torch.cuda.synchronize()
    end_time = time.time()
    decompression_time = (end_time - start_time) / num_iterations

    # Calculate throughput
    input_size_mb = input_size * 2 / (1024 * 1024)  # 2 bytes per bfloat16
    compression_throughput = input_size_mb / compression_time
    decompression_throughput = input_size_mb / decompression_time

    # Calculate compression ratio
    compressed_size = (bit_size.item() + 7) // 8  # Convert bits to bytes
    original_size = input_tensor.numel() * input_tensor.element_size()
    compression_ratio = original_size / compressed_size

    # Verify correctness
    exact_matches = torch.sum(input_tensor.bitwise_equal(decompressed_tensor)).item()
    
    print(f"Input size: {input_size} elements ({input_size_mb:.2f} MB)")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Compression throughput: {compression_throughput:.2f} MB/s")
    print(f"Decompression throughput: {decompression_throughput:.2f} MB/s")
    print(f"Number of exact matches: {exact_matches} out of {input_size}")
    
    if exact_matches < input_size:
        mismatch_indices = torch.where(~input_tensor.bitwise_equal(decompressed_tensor))[0]
        print("\nFirst 10 mismatches:")
        for idx in mismatch_indices[:10]:
            print(f"Index {idx}:")
            print(f"  Input:  {input_tensor[idx].item()}")
            print(f"  Output: {decompressed_tensor[idx].item()}")

    # Check for NaN or Inf values
    nan_count = torch.isnan(decompressed_tensor).sum().item()
    inf_count = torch.isinf(decompressed_tensor).sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"\nWarning: Decompressed data contains {nan_count} NaN values and {inf_count} Inf values")

def run_benchmarks():
    input_sizes = [1000, 10000, 100000, 1000000, 10000000]
    for size in input_sizes:
        print(f"\nBenchmarking with input size: {size}")
        benchmark_compression(size)

if __name__ == "__main__":
    run_benchmarks()
