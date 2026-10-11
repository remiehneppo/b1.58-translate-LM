# 0006. Kiến trúc Kernel SIMD ARM NEON và Tối Ưu Hóa Suy Luận Trên Luckfox Pico RV1103

Status: Accepted

## Context

Mô hình dịch ngôn ngữ siêu nhỏ Ru-Vi BitNet b1.58 (~10M tham số, vocab 8,192, $d_{model}=384$, 5 tầng Transformer) đã được triển khai và tối ưu thành công trên vi điều khiển ESP32-S3 (Xtensa dual-core LX7 @ 240 MHz) bằng kernel Xtensa 128-bit EE SIMD (ADR 0005).

Mục tiêu tiếp theo là mở rộng năng lực suy luận sang dòng board Linux nhúng siêu rẻ **Luckfox Pico / Pico Plus / Pico Mini** sử dụng vi xử lý **Rockchip RV1103**:
- **CPU**: ARM Cortex-A7 32-bit single-core @ 1.2 GHz (ARMv7-A / ARMv7VE, Thumb-2).
- **Vector & Floating Point**: ARM NEON Advanced SIMD (128-bit vector) + VFPv4 FPU.
- **Bộ nhớ đệm (Cache)**: 32 KB L1 I-Cache, 32 KB L1 D-Cache, 128 KB L2 Cache thống nhất.
- **Bộ nhớ RAM**: 64 MB DDR2 SIP tích hợp trên chip (băng thông ~800 – 1000 MB/s).
- **Hệ điều hành**: Embedded Linux 5.10 (Buildroot với uClibc-ng hoặc Ubuntu với glibc).
- **NPU**: Rockchip RKNN NPU 0.5 TOPS (int4/int8/int16).

So với ESP32-S3, RV1103 sở hữu xung nhịp cao hơn gấp 5 lần (1.2 GHz vs 240 MHz), băng thông RAM DDR2 cao hơn gấp 10 lần (1 GB/s vs 75 MB/s PSRAM), và tập thanh ghi NEON rộng hơn gấp đôi (16 thanh ghi 128-bit `q0..q15` thay vì 8 thanh ghi trên ESP32-S3). Tuy nhiên, kiến trúc ARMv7-A không có chỉ thị nhân chập phần cứng một chu kỳ trực tiếp vào accumulator 40-bit như ESP32-S3 (`ee.vmulas.s8.accx`) hay `sdot` của ARMv8.2-A. Cần xây dựng một kernel SIMD chuyên biệt khai thác tối đa pipeline Cortex-A7 dual-issue.

---

## Decisions

### 1. Hợp Ngữ ARMv7-A NEON 64-byte Unrolled Dot-Product (`kernel_rv1103.S`)
- **Tập lệnh**:
  - `vld1.8 {d0, d1}, [r0]!`: Nạp 16 byte (128-bit) có auto-increment con trỏ với bước nhảy 16.
  - `vmull.s8 q2, d0, d2`: Nhân 8 phần tử signed int8 thấp, sinh ra 8 phần tử signed int16 trong `q2`.
  - `vmlal.s8 q2, d1, d3`: Nhân 8 phần tử signed int8 cao, tích lũy tiếp vào 8 phần tử int16 trong `q2`.
  - `vpadal.s16 q3, q2`: Tích lũy cặp (pairwise add and accumulate long), cộng các cặp số 16-bit trong `q2` và tích lũy vào 4 làn signed 32-bit (`int32_t`) trong `q3`.
- **Khử trễ đường ống (Dual Accumulators Latency Hiding)**:
  - Trên Cortex-A7, lệnh `vmull`/`vmlal` có độ trễ 2 chu kỳ. Sử dụng 2 bộ tích lũy 32-bit độc lập `q3` và `q8` luân phiên giữa các block liền kề để triệt tiêu hiện tượng stall chờ thanh ghi (RAW hazard).
- **Mở rộng vòng lặp 64 byte**:
  - Vòng lặp chính `.Lloop64` xử lý 4 block (64 byte) mỗi lượt: nạp luân phiên dữ liệu của 4 block, tính MAC và tích lũy vào `q3` (block 0, 2) và `q8` (block 1, 3).
  - Vòng lặp đuôi `.Lloop16` xử lý các block 16-byte còn dư ($K$ luôn là bội số của 16 nhờ `padded_k`).
- **Tuân thủ tuyệt đối chuẩn AAPCS (Zero Stack Overhead)**:
  - Toàn bộ thanh ghi sử dụng gồm `r0..r3`, `r12` và `d0..d7`, `d16..d31` đều là **caller-saved**.
  - Không chạm vào các thanh ghi callee-saved `r4..r11` và `d8..d15`.
  - Hàm không cần sinh frame stack, không có chỉ thị `push`/`pop`, tối thiểu hóa chi phí gọi hàm con.

### 2. Định Dạng Căn Chỉnh 16-byte & Invariant Zero-Padding
- Tương tự quy tắc trên ESP32-S3:
  $$\text{stride} = \text{padded\_k}(K) = (K + 15) \ \& \ \sim 15$$
- Các phần tử đệm trong đoạn $[K, \text{stride})$ được điền số 0 (`std::memset`). Vì $0 \times 0 = 0$, vòng lặp vector chạy qua toàn bộ số block $\text{stride} / 16$ mà kết quả tích chập hoàn toàn bit-exact, loại bỏ 100% vòng lặp scalar xử lý đuôi.

### 3. Bộ Nhớ & Tối Ưu Hóa Cache L1 D-Cache
- Khác với ESP32-S3 phải sao chép dữ liệu giữa Flash và PSRAM, RV1103 có 64 MB DDR2 chạy trên Linux.
- Bộ nhớ đệm dữ liệu L1 D-Cache của RV1103 là **32 KB**.
- Với kích thước tile $16 \times 384$:
  $$\text{Tile Size} = 16 \times 384 = 6,144 \text{ bytes} \approx 6 \text{ KB}$$
  Toàn bộ tile trọng số giải nén int8 và vector kích hoạt nằm trọn 100% trong L1 D-Cache trong suốt quá trình nhân ma trận, đạt tốc độ truy xuất cực đại ~1 chu kỳ CPU mà không cần truy cập bus DDR2 ngoài.
- Bảng tra cứu giải nén `s_unpack_lut` (256 phần tử $\times$ 4 byte = 1,024 byte = 1 KB) cũng nằm trọn trong L1 D-Cache, giải nén 4 trọng số 2-bit thành 4 giá trị signed `int8_t` trong 1 thao tác đọc/ghi 32-bit duy nhất.

### 4. Tăng Tốc Tầng Đầu Ra LM Head INT8 & Sub-Vocabulary Masking
- Tầng đầu ra LM Head ($8192 \times 384 = 3.15\text{M}$ tham số) ở dạng Signed INT8 (`kind == 3`).
- Dữ liệu trọng số int8 được nạp thẳng vào kernel `dot_rv1103_aligned` mà không cần giải nén tile.
- Kết hợp với Sub-Vocabulary Masking (`SUBVOCAB_RU_MASK` / `SUBVOCAB_VI_MASK`):
  - Bỏ qua các hàng từ vựng không thuộc ngôn ngữ mục tiêu.
  - Các hàng hợp lệ được tính bằng SIMD NEON với tốc độ hàng tỷ phép tính mỗi giây.

### 5. Khởi Tạo Vùng Nhớ Linux với `madvise(MADV_WILLNEED)`
- Khi mở file mô hình `ruvi_best_int8.bin` (4.88 MB) qua `mmap(MAP_PRIVATE)`, chèn lệnh:
  ```cpp
  madvise(data, size, MADV_WILLNEED);
  ```
- Lệnh này chỉ thị cho nhân Linux nạp trước (pre-fault) các trang bộ nhớ của mô hình vào Page Cache trong DDR2, loại bỏ hoàn toàn các ngắt trang (major/minor page faults) trong quá trình decode token đầu tiên.

### 6. Đánh Giá CPU NEON So Với NPU RKNN Trên RV1103
- **Rockchip NPU (0.5 TOPS)**:
  - Yêu cầu đồ thị mạng cố định (static ONNX graph), chuyển đổi offline bằng `rknn-toolkit2` trên máy x86_64 thành định dạng đóng `.rknn`.
  - Yêu cầu liên kết thư viện động độc quyền `librknnrt.so`.
  - Quá trình suy luận tự hồi quy (autoregressive decode step-by-step) có chi phí overhead gọi driver và truyền buffer qua DMA lớn cho từng token đơn lẻ. Không hỗ trợ trực tiếp trọng số ternary 2-bit động kiểu BitNet b1.58.
- **CPU Cortex-A7 NEON (1.2 GHz)**:
  - Chạy native C++ độc lập, zero-dependency, nhúng trực tiếp trong runtime `ruvi-native`.
  - Độ trễ gọi kernel cực thấp (nanosecond), phù hợp hoàn hảo với kiến trúc giải mã từng token của mô hình ngôn ngữ.
  - Do đó, CPU NEON là lựa chọn tối ưu hàng đầu cho runtime của dự án. NPU RKNN chỉ được xem xét cho các bài toán nhận dạng giọng nói hoặc thị giác (CNN/Fixed ViT) khi có đồ thị cố định.

---

## Consequences

1. **Hiệu năng suy luận**:
   - Nhờ xung nhịp 1.2 GHz và SIMD NEON unrolled 64-byte, tốc độ suy luận trên Luckfox Pico RV1103 vượt trội, đạt throughput cao và độ trễ thấp cho cả pha prefill và decode.
2. **Dung lượng & Tài nguyên**:
   - Toàn bộ mô hình (4.88 MB) và workspace (< 1 MB) chỉ chiếm ~6 MB trên tổng 64 MB DDR2 của RV1103.
   - Hệ thống còn trống >50 MB RAM cho hệ điều hành Linux và các dịch vụ mạng (Wi-Fi, USB Ethernet, Web API).
3. **Tính ổn định & Độ chính xác**:
   - Kết quả đầu ra hoàn toàn bit-exact với kernel scalar reference.
   - Thao tác decode không cấp phát heap (`malloc`/`free` = 0), bảo đảm an toàn chạy 24/7.
