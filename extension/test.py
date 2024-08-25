import torch
import bfloat16_compression
import time
import numpy as np

def benchmark_compression(input_size, num_iterations=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA is not available. Skipping benchmark.")
        return

    print(f"Generating input tensor of size {input_size}...")
    input_tensor = torch.randn(input_size, dtype=torch.bfloat16, device=device)
    print("Input tensor generated.")

    print("Running compression...")
    compressed_tensor, output_size = bfloat16_compression.compress_bfloat16(input_tensor)
    print("Compression completed.")

    print("Running decompression...")
    decompressed_tensor = bfloat16_compression.decompress_bfloat16(compressed_tensor, input_size)
    print("Decompression completed.")

    compression_times = []
    decompression_times = []
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        compressed_tensor, _ = bfloat16_compression.compress_bfloat16(input_tensor)
        torch.cuda.synchronize()
        compression_times.append(time.time() - start_time)

        torch.cuda.synchronize()
        start_time = time.time()
        decompressed_tensor = bfloat16_compression.decompress_bfloat16(compressed_tensor, input_size)
        torch.cuda.synchronize()
        decompression_times.append(time.time() - start_time)

    compression_time = np.mean(compression_times)
    decompression_time = np.mean(decompression_times)

    input_size_mb = input_size * 2 / (1024 * 1024)  # 2 bytes per bfloat16
    compression_throughput = input_size_mb / compression_time
    decompression_throughput = input_size_mb / decompression_time

    compressed_size = compressed_tensor.numel() * compressed_tensor.element_size()
    original_size = input_tensor.numel() * input_tensor.element_size()
    compression_ratio = original_size / compressed_size

    print(f"\nResults:")
    print(f"Input size: {input_size} elements ({input_size_mb:.2f} MB)")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Compression throughput: {compression_throughput:.2f} MB/s")
    print(f"Decompression throughput: {decompression_throughput:.2f} MB/s")

    # Compare tensors element-wise
    exact_matches = torch.sum(input_tensor == decompressed_tensor).item()
    print(f"Number of exact matches: {exact_matches} out of {input_size}")

    if exact_matches < input_size:
        mismatch_indices = torch.where(input_tensor != decompressed_tensor)[0]
        print("\nFirst 10 mismatches:")
        for idx in mismatch_indices[:10]:
            input_val = input_tensor[idx].item()
            output_val = decompressed_tensor[idx].item()
            input_hex = input_tensor[idx].view('int16').item()
            output_hex = decompressed_tensor[idx].view('int16').item()
            print(f"Index {idx}:")
            print(f"  Input:  {input_val} (hex: {input_hex:04x})")
            print(f"  Output: {output_val} (hex: {output_hex:04x})")

    nan_count = torch.isnan(decompressed_tensor).sum().item()
    inf_count = torch.isinf(decompressed_tensor).sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"\nWarning: Decompressed data contains {nan_count} NaN values and {inf_count} Inf values")

def run_benchmarks():
    input_sizes = [1000, 10000, 100000, 1000000]
    for size in input_sizes:
        print(f"\nBenchmarking with input size: {size}")
        benchmark_compression(size)

if __name__ == "__main__":
    run_benchmarks()
