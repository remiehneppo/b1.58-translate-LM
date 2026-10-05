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

## Luckfox Pico Pro và EBAZ4205

Chọn **compiler và libc khớp OS/image đang chạy**. Ví dụ Luckfox SDK Buildroot dùng
`arm-rockchip830-linux-uclibcgnueabihf-g++`; Ubuntu ARM có thể dùng GNU armhf toolchain.
Không dùng binary uClibc trên hệ glibc hoặc ngược lại.
[Hướng dẫn cross-compile của Luckfox](https://wiki.luckfox.com/zh/Luckfox-Pico-Plus-Mini/Cross-Compile/).

```sh
# Trỏ tới g++ của SDK đúng với board/image:
export RUVI_ARM_CXX=/absolute/path/to/arm-rockchip830-linux-uclibcgnueabihf-g++
cmake -S native -B native/build-arm \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=arm \
  -DCMAKE_CXX_COMPILER="$RUVI_ARM_CXX"
cmake --build native/build-arm -j2
```

Chuyển executable `native/build-arm/ruvi-native`, `demo.bin` và manifest lên board
bằng phương thức hiện có. EBAZ mod SD có thể lưu file trên SD. Chạy đúng command host
với cùng prompt/context; lưu JSON stdout cùng thông tin board, OS, CPU clock và artifact
SHA256. Chỉ sau khi CPU benchmark hoàn tất mới đánh giá NPU/FPGA.

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
