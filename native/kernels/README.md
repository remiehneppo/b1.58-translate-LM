# Independent RUVI0001 ternary kernel

These files are new implementations from the requested packing contract. No ASR
source was read or copied. The S3 instructions follow the Espressif ESP32-S3
Technical Reference Manual v1.8, sections 1.8.58 (SRS.ACCX), 1.8.88 (VLD.128.IP),
1.8.146 (VMULAS.S8.ACCX), and 1.8.215 (ZERO.ACCX):
https://www.espressif.com/sites/default/files/documentation/esp32-s3_technical_reference_manual_en.pdf

## API (`ruvi::kernels`, `kernel.h`)

- `size_t padded_k(size_t k)` returns align16(k), or zero outside [1,32768].
- `size_t weight_scratch_bytes(size_t k, size_t tile_rows)` returns
  tile_rows * align16(k), or zero unless tile_rows is 8, 16, or 32.
- `Status run(const Problem&, const Options&, const Scratch&)` computes
  `output[token*output_stride+n] = float(exact_int32_sum) *
  (weight_scale / multipliers[token*multiplier_stride])`.
- `bool esp32_s3_available()` reports compile-time S3 support.
- `Status dot_s3_aligned(const int8_t*, const int8_t*, size_t, int32_t*)`
  checks alignment and size before calling the private assembly leaf.

`Problem.weights` points at the RUVI0001 **payload**, not a file header. Element
`n*k+j` uses the low-to-high two-bit pair in byte `index/4`; rows have no byte
padding. Codes 00/01/10 represent -1/0/+1; 11 is rejected within the selected
partition. One finite `weight_scale` applies to the entire tensor. Inputs are
already quantized signed int8; each token must expose align16(k) bytes, with
`input_stride` in bytes. Input padding may contain arbitrary values and never
contributes. Outputs and multiplier strides/capacities are in float elements.
Multipliers are finite and positive; there is no quantization or allocation.

`Options.n_begin,n_count` select a contiguous N partition; default SIZE_MAX count
means the remainder. Output retains the full global N column indexing. Separate
callers can run disjoint partitions with separate weight scratch and, if used,
separate input-copy scratch. No writable buffer may overlap another buffer.

The portable path accepts unaligned int8 buffers. S3 requires 16-byte-aligned
weight scratch and aligned token input addresses. Alternatively an optional
16-byte-aligned input-copy scratch of align16(k) bytes copies one token at a time
for unaligned input/strides. `automatic` falls back to portable when S3 or
alignment is unavailable; explicit `esp32_s3` reports unsupported on host or
invalid_argument for bad S3 alignment. Scratch capacities are checked explicitly.
Weight unpacking is once per tile and shared across all M tokens. The integer
accumulation bound for ternary weights is 32768*128 = 4194304.

## Standalone host tests

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  native/kernels/kernel.cpp native/kernels/kernel_s3.cpp \
  native/kernels/test_kernel.cpp -o /tmp/ruvi-test-kernel
/tmp/ruvi-test-kernel

g++ -std=c++17 -O1 -g -Wall -Wextra -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  native/kernels/kernel.cpp native/kernels/kernel_s3.cpp \
  native/kernels/test_kernel.cpp -o /tmp/ruvi-test-kernel-asan
/tmp/ruvi-test-kernel-asan
```

Test coverage includes continuous odd-K row offsets, M>1, input/output and
multiplier strides, untouched output padding, partial tiles, N partitions,
aligned direct input, unaligned input with optional microcopy, deterministic
random data, extreme int8/ternary values at K=32767/32768, invalid codes,
capacities/multipliers, and the checked S3 entry.

## S3 compilation

Compile `kernel.cpp`, `kernel_s3.cpp`, and `kernel_s3.S` with the ESP32-S3 Xtensa
toolchain; define `RUVI_KERNEL_ESP32_S3=1` for all three, or use the IDF
`CONFIG_IDF_TARGET_ESP32S3` definition. The `.S` leaf supports windowed and call0
ABIs. It uses EE signed-int8 multiplication with ACCX and an explicit `bnez`
software branch, never hardware loop instructions. The private unchecked leaf
must only be called via the checked wrapper. EE q0/q1/ACCX are volatile state;
firmware must preserve PIE state when preempting the kernel with another PIE
user. There are no firmware/build-system changes in this directory addition.
Host tests do not assemble or execute the S3 instructions; S3 cross-build and
hardware execution remain required to validate that path.
