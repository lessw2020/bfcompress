__global__ void bfloat16_decompress_kernel(
    const unsigned int* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const int input_bit_size,
    const int output_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bit_offset = 0;

    unsigned short prev = 0;
    for (int i = 0; i < output_size && bit_offset < input_bit_size; ++i) {
        unsigned int word = input[bit_offset / 32];
        unsigned int shift = bit_offset % 32;
        unsigned int prefix = (word >> shift) & 0x7;
        bit_offset += 3;

        short delta;
        if (prefix == 0) {
            delta = 0;
            bit_offset += 1;
        } else if (prefix <= 3) {
            delta = (short)((word >> (shift + 3)) & 0xF);
            if (delta & 0x8) delta |= 0xFFF0;
            bit_offset += 4;
        } else if (prefix <= 5) {
            delta = (short)((word >> (shift + 3)) & 0xFF);
            if (delta & 0x80) delta |= 0xFF00;
            bit_offset += 8;
        } else {
            delta = (short)((word >> (shift + 3)) & 0xFFFF);
            bit_offset += 16;
        }

        unsigned short current = prev + delta;
        if (i == tid) {
            output[i] = ushort_to_bfloat16(current);
            printf("Debug: tid=%d, prev=%hu, delta=%hd, current=%hu\n", tid, prev, delta, current);
        }
        prev = current;
    }
}
