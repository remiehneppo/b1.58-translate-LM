# RuVi native research runtime

Mục tiêu: chạy **cùng artifact/model** trên host, Luckfox Pico Pro, ESP32-S3 N16R8
và EBAZ4205. Chưa đặt yêu cầu tốc độ sử dụng thực tế. Không giảm layer, vocab hay
hidden size riêng cho ESP32. Các board chạy inference; training vẫn thực hiện trên host.

## Trạng thái kiểm chứng

- Host Linux x86_64: đã build và chạy checkpoint demo thực tế.
- Greedy tokens hai hướng vi2ru/ru2vi khớp CPU PyTorch; kiểm tra logits từng bước.
- ESP32-S3 N16R8: đã flash và chạy một lượt inference bằng ESP-IDF 5.5.4; 8 token khớp host. Xem [kết quả](../docs/benchmarks/esp32-inference.json).
- Luckfox/EBAZ: có đường cross-build bên dưới, **chưa có toolchain/board để kiểm chứng**.
- Không sử dụng NPU hoặc FPGA ở giai đoạn này.

Tokenizer/dataset không thay đổi. Runtime native nhận **prefix token IDs** và trả
**generated token IDs**; tokenize/decode văn bản hiện dùng tokenizer Python trên host.
Đây là kiểm chứng model inference, chưa phải ứng dụng dịch văn bản standalone trên MCU.

## Kiến trúc Kernel & Tối ưu hóa trên ESP32-S3

Hệ thống bổ sung module `native/kernels/` với kiến trúc 3 tầng:

1. **Portable Tiled Kernel (`ruvi::kernels::run`)**:
   - Giải nén trọng số tam phân 2-bit (RUVI0001: continuous bit stream, `00=-1`, `01=0`, `10=+1`) thành vector `int8_t` trong buffer SRAM tạm theo từng tile hàng (`tile_rows` = 8, 16, hoặc 32).
   - Tối ưu hóa giải nén bằng bảng tra cứu 256 phần tử (`s_unpack_lut`), xử lý 4 trọng số trong 1 thao tác đọc/ghi 32-bit khi hàng bắt đầu tại ranh giới byte, và fallback bitwise chính xác cho các ranh giới lệch.
   - Tránh giải nén lại nhiều lần: một tile trọng số giải nén được dùng chung cho toàn bộ token trong batch prefill.

2. **ESP32-S3 128-bit EE SIMD Assembly (`ruvi_kernel_s3_dot_raw`)**:
   - Sử dụng tập lệnh mở rộng Energy-Efficient (EE) 128-bit của Xtensa LX7 (`ee.vld.128.ip`, `ee.vmulas.s8.accx`).
   - Xoay vòng thanh ghi 4x qua 8 thanh ghi vector `q0..q7` (64 phần tử / vòng lặp) để triệt tiêu độ trễ load-to-use.
   - Tuyệt đối không dùng `loop`/`loopnez` để tránh ghi đè thanh ghi vòng lặp phần cứng của GCC; sử dụng vòng lặp rẽ nhánh phần mềm `addi` + `bnez`.
   - Căn chỉnh địa chỉ 16-byte nghiêm ngặt với padding số 0 ở đuôi hàng (`padded_k = (K + 15) & ~15`), không cần vòng lặp scalar xử lý phần dư.

3. **Phân tầng Bộ nhớ (Memory Tiering)**:
   - **Flash XIP**: Trọng số mô hình giữ nguyên trên Flash SPI, ánh xạ trực tiếp vào CPU address space qua `esp_partition_mmap`. Không sao chép toàn bộ model vào RAM.
   - **Internal SRAM**: Cấp phát buffer tile (`tile_scratch`) bằng `heap_caps_aligned_alloc(16, ..., MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)`. Việc đọc trọng số int8 từ Internal SRAM tránh làm nghẽn bus và cache của Octal PSRAM.
   - **Octal PSRAM**: Cấp phát KV-cache, hidden prefix, và workspace qua `heap_caps_malloc(..., MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)`.

4. **Tối ưu hóa Luồng Inference**:
   - **RoPE Precomputation**: Bảng `sin`/`cos` được tính toán trước tại thời điểm `bind_workspace`, loại bỏ hoàn toàn các hàm lượng giác siêu việt (`pow`, `sin`, `cos`) trong vòng lặp giải mã.
   - **Fused RMSNorm & Quantization**: Trong pha decode, chuẩn hóa RMSNorm và lượng tử hóa kích hoạt int8 chỉ thực hiện một lần cho cả 3 phép chiếu Q, K, V dạng tam phân.
   - **Cache Locality trong Attention**: Hoán đổi thứ tự vòng lặp tính tổng có trọng số của vector Value, chuẩn hóa trọng số một lần và duyệt bộ nhớ với bước nhảy đơn vị (unit stride).

## Build và chạy host

Từ root dự án, dùng Python environment đã cài torch/transformers:

```sh
cmake -S native -B native/build
cmake --build native/build -j2
python scripts/export_native.py \
  --checkpoint ru-vi-tatoeba-demo/ruvi_10m_quantized.pt \
  --output native/artifacts/demo.bin
native/build/ruvi-native --model native/artifacts/demo.bin \
  --prompt 4,100,200,300,3 --context 128 --max-new 8 --repeat 3
python scripts/check_native.py \
  --checkpoint ru-vi-tatoeba-demo/ruvi_10m_quantized.pt \
  --image native/artifacts/demo.bin --text 'Xin chào, bạn khỏe không?' \
  --direction vi2ru --max-new 16
python scripts/check_native.py \
  --checkpoint ru-vi-tatoeba-demo/ruvi_10m_quantized.pt \
  --image native/artifacts/demo.bin --text 'Привет! Как дела?' \
  --direction ru2vi --max-new 16
python -m unittest discover -s tests -v
```

Exporter mặc định giữ phần không ternary ở BF16, ternary ở packed 2-bit cùng scale
FP32. Với checkpoint `ruvi_10m_quantized.pt` hiện tại, phần không ternary đã là BF16;
export không làm giảm thêm precision của weight. `--float-storage fp32` cũng được hỗ trợ.
Cùng file `.bin` được dùng trên mọi board; manifest `.bin.json` chứa SHA256, CRC32,
config và tensor inventory để đối chiếu artifact.

Khi export checkpoint shadow chưa đóng gói, weight BitLinear được lượng tử hóa theo
công thức đang dùng trong model; phần không ternary có thể được làm tròn sang BF16.
Oracle kiểm chứng dùng đúng weight/precision được lưu trong image, tính toán FP32 trên CPU.
Không nạp checkpoint pickle không đáng tin vào exporter.

## Benchmark

```sh
python scripts/benchmark_native.py --image native/artifacts/demo.bin \
  --contexts 64,128,256 --prompt-lengths 16,64,128 --max-new 8 --repeat 3 \
  --output native/artifacts/benchmark-host.json
```

Workload là token IDs tổng hợp có seed cố định, không đo chất lượng dịch. JSON lưu
platform, artifact SHA256, prompt IDs và số bước decode thực tế. EOS vẫn kết thúc sinh
bình thường; không ép model sinh đủ số token để làm đẹp throughput.

- `load_ms`: mmap + kiểm tra header, CRC và tensor, chưa cấp workspace.
- `prefill_*`: bidirectional prefix qua mọi layer và tính logits token đầu.
- `ttft_p50_ms`: thời gian từ lúc bắt đầu prefill tới argmax token đầu, không gồm load.
- `decode_tokens_per_second`: số bước decode sau token đầu / tổng thời gian decode.
- `total_p50_ms`: prefill + generation; nếu bật `--logits` thì có thêm chi phí copy trace.
- `workspace_bytes`: toàn bộ KV-cache, hidden prefix, scratch, logits và activation int8.
- `peak_rss_kib`: VmHWM của tiến trình Linux, gồm cả page file model cư trú. Không được
  coi con số này là PSRAM cần trên ESP32; weight ESP32 đọc trực tiếp qua flash cache.

Một lần chạy đầu không được gọi là cold-cache benchmark: page cache có thể đã nóng.
Ba repeats chỉ là smoke benchmark, không đủ để kết luận P95 ổn định.

Lõi hiện là scalar reference: GQA không nhân bản K/V, attention không cấp ma trận S×S,
prefill giữ một mảng hidden rồi cập nhật tại chỗ sau khi đã dựng K/V cả prefix. Decode
không cấp phát thêm buffer. Scale weight được áp dụng đúng một lần. Scratch và KV dùng
FP32; không thay precision theo board. Packed weights được giải mã từng phần tử ngay
trong dot product; không có bản float/int8 đầy đủ của weight ternary trong RAM.

Rounding/accumulation khác BLAS/SDPA có thể làm lệch các ngưỡng lượng tử hóa. Parity
không phải bit-exact: checker yêu cầu greedy IDs khớp và logits nằm trong atol/rtol
công bố (demo mặc định atol=0.02, rtol=0.005). Regression model nhỏ dùng atol=2e-5.
Parity pass không chứng minh chất lượng bản dịch, hoặc bảo đảm mọi prompt cho cùng IDs.

## Luckfox Pico RV1103 (ARM Cortex-A7 NEON) và EBAZ4205

Luckfox Pico / Pico Mini / Pico Plus trang bị vi xử lý **Rockchip RV1103** (ARM Cortex-A7 single-core @ 1.2 GHz, 32KB L1 D-Cache, 128KB L2 Cache, 64MB DDR2, Linux 5.10).

Runtime `ruvi-native` tích hợp kernel SIMD **ARMv7-A NEON** (`native/kernels/kernel_rv1103.S` và `kernel_rv1103.cpp`) mở rộng 64-byte unrolled dot product với 2 bộ tích lũy 32-bit song song (`vpadal.s16`) và giải nén tile 16 hàng nằm trọn trong L1 D-Cache (6.14 KB).

Chọn **compiler và libc khớp OS/image đang chạy**. Ví dụ Luckfox SDK Buildroot dùng
`arm-rockchip830-linux-uclibcgnueabihf-g++`; Ubuntu ARM có thể dùng GNU armhf toolchain (`arm-linux-gnueabihf-g++`).
Không dùng binary uClibc trên hệ glibc hoặc ngược lại.
[Hướng dẫn cross-compile của Luckfox](https://wiki.luckfox.com/zh/Luckfox-Pico-Plus-Mini/Cross-Compile/).

```sh
# Cách 1: Sử dụng toolchain file cấu hình sẵn cho RV1103:
export RUVI_TOOLCHAIN_PREFIX=/path/to/luckfox-pico/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-
cmake -S native -B native/build-rv1103 \
  -DCMAKE_TOOLCHAIN_FILE=native/toolchain-rv1103.cmake \
  -DRUVI_TOOLCHAIN_PREFIX="$RUVI_TOOLCHAIN_PREFIX"
cmake --build native/build-rv1103 -j$(nproc)

# Cách 2: Trỏ trực tiếp tới compiler g++:
export RUVI_ARM_CXX=/path/to/arm-rockchip830-linux-uclibcgnueabihf-g++
cmake -S native -B native/build-arm \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=arm \
  -DCMAKE_CXX_COMPILER="$RUVI_ARM_CXX"
cmake --build native/build-arm -j$(nproc)
```

Chuyển executable `ruvi-native`, model artifact `ruvi_best_int8.bin` (4.88 MB) và manifest lên board qua ADB/SSH/SD-card:

```sh
adb push native/build-rv1103/ruvi-native /userdata/
adb push native/artifacts/ruvi_best_int8.bin /userdata/
adb push native/artifacts/ruvi_best_int8.bin.json /userdata/

# Đăng nhập vào shell board:
adb shell   # hoặc ssh root@172.32.0.93 (password: luckfox)
cd /userdata

# Cách 1: Dịch trực tiếp văn bản tự nhiên (CLI 1 lần):
./ruvi-native --model ruvi_best_int8.bin --text "Xin chào, bạn khỏe không?" --dir vi2ru --backend rv1103
./ruvi-native --model ruvi_best_int8.bin --text "Привет, как дела?" --dir ru2vi --backend rv1103

# Cách 2: Chế độ tương tác gõ trực tiếp (Interactive Chat Mode):
./ruvi-native --model ruvi_best_int8.bin -i --backend rv1103
# [vi2ru] > Tôi có một con bò.
# => У меня есть коров. (8.6 tok/s)
# [vi2ru] > :dir ru2vi
# [ru2vi] > Открой сейф.
# => Mở két sắt ra. (8.5 tok/s)

# Cách 3: Chạy với token IDs thô để đo đạc benchmark:
./ruvi-native --model ruvi_best_int8.bin --prompt 4,12,34,56,2 --context 128 --max-new 8 --backend rv1103
```

Chỉ sau khi CPU NEON benchmark hoàn tất mới đánh giá NPU/FPGA. Chi tiết thiết kế kiến trúc xem [ADR 0006](../docs/adr/0006-luckfox-pico-rv1103-neon-kernel.md).

## ESP32-S3 N16R8

Firmware nằm dưới `native/esp32/`; dùng ESP-IDF 5.5.4 đã được build-check. Mặc định:
16 MB flash, Octal PSRAM, context 128, max-new 8. `heap_caps_malloc(...SPIRAM...)`
cấp workspace; `esp_partition_mmap` đọc model trực tiếp từ flash.
[Flash mapping](https://docs.espressif.com/projects/esp-idf/en/v5.1.3/esp32s3/api-reference/peripherals/spi_flash/index.html),
[external RAM](https://documentation.espressif.com/projects/esp-idf/en/v6.0.3/esp32s3/api-guides/external-ram.html).

Từ root, sau khi activate ESP-IDF của bạn:

```sh
idf.py -C native/esp32 -B "$PWD/build" set-target esp32s3 build
idf.py -C native/esp32 -B "$PWD/build" menuconfig
# RuVi research inference: nhập prefix IDs từ tokenizer của đúng checkpoint.
# Sau khi đổi cấu hình:
idf.py -C native/esp32 -B "$PWD/build" build
```

`partitions.csv` dành 15 MiB tại offset `0x100000` cho model, app nằm trước vùng này.
Không nhúng model vào `.rodata` hay copy vào PSRAM. Flash image riêng, sau khi đã
chọn đúng cổng/board và chấp nhận ghi firmware:

```sh
export RUVI_PORT=/dev/ttyUSB0
idf.py -C native/esp32 -B "$PWD/build" -p "$RUVI_PORT" flash
python -m esptool --chip esp32s3 -p "$RUVI_PORT" write_flash \
  0x100000 native/artifacts/demo.bin
idf.py -C native/esp32 -B "$PWD/build" -p "$RUVI_PORT" monitor
```

Firmware báo CRC, model bytes, workspace bytes, load/prefill/decode và generated IDs
qua serial. Đối chiếu CRC với manifest và IDs với checker host cho **cùng prompt/max-new**.
Lần thử đầu giữ context 128; model/config/weight không đổi. mmap hoặc cấp PSRAM thất bại
được báo rõ, không fallback sang model nhỏ hơn. Việc build thành công chưa chứng minh
flash mapping đủ chỗ hoặc inference hoàn tất trên board thực tế.

Task watchdog không tự khởi tạo trong cấu hình research này vì scalar inference có thể
chạy rất lâu. Không dùng cấu hình này làm firmware sản phẩm trước khi thêm scheduling
và watchdog policy phù hợp. Board kiểm chứng có flash Quad 16 MB ở 3,3 V và Octal PSRAM 8 MB.
Khi dùng module khác cần xác nhận thông số tương ứng trước khi flash.

## Binary format RUVI0001

Little-endian, header 80 byte: magic 8 byte; 12 uint32 (vocab, dim, layers, heads,
kv_heads, head_dim, ff, max_context, eos, pad, tag_ru, tag_vi); float32 theta/eps;
uint32 image length/CRC32; 8 byte padding. CRC32 tính toàn image với trường CRC bằng 0.
CRC dùng phát hiện hỏng dữ liệu; SHA256 manifest dùng nhận diện artifact.

Mỗi tensor có header 32 byte: uint32 rows/cols/kind, float32 scale, uint32 payload bytes,
12 byte padding; payload pad tới bội 16 byte. Kind 0=FP32, 1=BF16, 2=ternary 2-bit
(-1→0, 0→1, +1→2; code 3 bị từ chối). Thứ tự tensor cố định như exporter, không có
lookup tên lúc inference. Embedding được dùng lại cho output head.

Parser kiểm tra version, config, checksum, shape, dtype, kích thước và finite values;
không đọc padding partition như tensor. V1 hỗ trợ batch 1, greedy decode, Prefix-LM,
GQA, RoPE xen kẽ, RMSNorm và ReLU², không hỗ trợ projection bias.
