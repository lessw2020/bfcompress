import torch
import bfloat16_compression
import time
import numpy as np
import struct

def bfloat16_to_hex(x):
    return f"{struct.unpack('<H', struct.pack('<e', x))[0]:04x}"

def benchmark_compression(input_size, num_iterations=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA is not available. Skipping benchmark.")
        return

    print(f"Generating input tensor of size {input_size}...")
    input_tensor = torch.randn(input_size, dtype=torch.bfloat16, device=device)
    print("Input tensor generated.")

    print("Running compression...")
    compressed_tensor, bit_size = bfloat16_compression.compress_bfloat16(input_tensor)
    print("Compression completed.")

    print("Running decompression...")
    decompressed_tensor = bfloat16_compression.decompress_bfloat16(compressed_tensor, bit_size, input_size)
    print("Decompression completed.")

    compression_times = []
    decompression_times = []
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start_time = time.time()
        compressed_tensor, bit_size = bfloat16_compression.compress_bfloat16(input_tensor)
        torch.cuda.synchronize()
        compression_times.append(time.time() - start_time)

        torch.cuda.synchronize()
        start_time = time.time()
        decompressed_tensor = bfloat16_compression.decompress_bfloat16(compressed_tensor, bit_size, input_size)
        torch.cuda.synchronize()
        decompression_times.append(time.time() - start_time)

    compression_time = np.mean(compression_times)
    decompression_time = np.mean(decompression_times)

    input_size_mb = input_size * 2 / (1024 * 1024)  # 2 bytes per bfloat16
    compression_throughput = input_size_mb / compression_time
    decompression_throughput = input_size_mb / decompression_time

    compressed_size = (bit_size.item() + 7) // 8  # Convert bits to bytes
    original_size = input_tensor.numel() * input_tensor.element_size()
    compression_ratio = original_size / compressed_size

    print(f"\nResults:")
    print(f"Input size: {input_size} elements ({input_size_mb:.2f} MB)")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Compression throughput: {compression_throughput:.2f} MB/s")
    print(f"Decompression throughput: {decompression_throughput:.2f} MB/s")

    exact_matches = torch.sum(input_tensor == decompressed_tensor).item()
    print(f"Number of exact matches: {exact_matches} out of {input_size}")

    if exact_matches < input_size:
        mismatch_indices = torch.where(input_tensor != decompressed_tensor)[0]
        print("\nFirst 10 mismatches:")
        for idx in mismatch_indices[:10]:
            input_val = input_tensor[idx].item()
            output_val = decompressed_tensor[idx].item()
            input_hex = bfloat16_to_hex(input_val)
            output_hex = bfloat16_to_hex(output_val)
            print(f"Index {idx}:")
            print(f"  Input:  {input_val} (hex: {input_hex})")
            print(f"  Output: {output_val} (hex: {output_hex})")

    nan_count = torch.isnan(decompressed_tensor).sum().item()
    inf_count = torch.isinf(decompressed_tensor).sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"\nWarning: Decompressed data contains {nan_count} NaN values and {inf_count} Inf values")

    # Calculate and print error statistics
    abs_error = torch.abs(input_tensor - decompressed_tensor)
    max_error = torch.max(abs_error).item()
    mean_error = torch.mean(abs_error).item()
    mse = torch.mean((input_tensor - decompressed_tensor) ** 2).item()
    print(f"\nError statistics:")
    print(f"Max absolute error: {max_error}")
    print(f"Mean absolute error: {mean_error}")
    print(f"Mean squared error: {mse}")

    return compression_ratio, compression_throughput, decompression_throughput, exact_matches / input_size

def run_benchmarks():
    input_sizes = [1000, 10000, 100000, 1000000]
    results = []
    for size in input_sizes:
        print(f"\nBenchmarking with input size: {size}")
        result = benchmark_compression(size)
        results.append((size,) + result)

    print("\nSummary of results:")
    print("Input Size | Compression Ratio | Compression Throughput (MB/s) | Decompression Throughput (MB/s) | Accuracy")
    print("-" * 100)
    for size, ratio, comp_throughput, decomp_throughput, accuracy in results:
        print(f"{size:10d} | {ratio:18.2f} | {comp_throughput:30.2f} | {decomp_throughput:31.2f} | {accuracy:8.6f}")

if __name__ == "__main__":
    run_benchmarks()
