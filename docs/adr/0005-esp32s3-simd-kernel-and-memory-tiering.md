# 0005. Kiến trúc Kernel SIMD Tam Phân, Phân Tầng Bộ Nhớ và Tối Ưu Hóa Inference trên ESP32-S3

Status: Accepted

## Context

Mô hình dịch ngôn ngữ siêu nhỏ Ru-Vi BitNet b1.58 (~10M tham số, vocab 8,192, $d_{model}=384$, 5 tầng Transformer) được triển khai độc lập trên vi điều khiển ESP32-S3 (Xtensa dual-core LX7 @ 240 MHz, 16 MB Flash, 8 MB Octal PSRAM, ~512 KB Internal SRAM).

Phiên bản scalar CPU baseline ban đầu đạt tốc độ sinh từ rất chậm (~0.63 token/s, ~1,583 ms/token) và tốn gần 4 giây cho prefill 5 token. Các nguyên nhân gây nghẽn:
1. Thiếu tăng tốc phần cứng cho phép nhân ma trận tam phân (GEMV/GEMM);
2. Trọng số tam phân 2-bit bị giải mã bit-by-bit lặp lại trong vòng lặp tính toán;
3. Băng thông SPI Flash bị bão hòa và xung đột giữa Data Cache và Instruction Cache;
4. Tầng chiếu đầu ra (LM Head / Tied Embeddings) kích thước lớn ($8192 \times 384 = 3.15\text{M}$ tham số) ở dạng BF16 phải chạy vòng lặp nhân float scalar trên CPU;
5. Tính toán lặp lại RMSNorm và lượng tử hóa kích hoạt int8 cho các phép chiếu Q, K, V;
6. Gọi liên tục các hàm siêu việt (`pow`, `sin`, `cos`) trong RoPE ở mỗi bước sinh từ;
7. Chưa tận dụng lõi CPU thứ 2 (Core 1) của ESP32-S3;
8. Phụ thuộc vào tokenizer Python trên máy tính host, chưa thể chạy standalone trên MCU.

## Decisions

### 1. Phân Tầng Bộ Nhớ (Memory Tiering Strategy)
- **Octal PSRAM Caching (`CONFIG_RUVI_MODEL_IN_PSRAM=y`)**: Khi khởi động, nạp toàn bộ khối nhị phân mô hình (4.88 MB INT8) từ Flash vào Octal PSRAM. Băng thông đọc Octal PSRAM 80MHz đạt ~60–75 MB/s (nhanh gấp 2.5 lần Flash), đồng thời giải phóng hoàn toàn bus SPI Flash cho Instruction Cache, loại bỏ triệt để hiện tượng tráo đổi dòng cache (cache thrashing).
- **Internal SRAM (`MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT`)**: Cấp phát các tile buffer trọng số giải nén int8 (`tile_scratch_[0]` và `tile_scratch_[1]`, 42.3 KB cho cả 2 core). Đọc trọng số int8 từ Internal SRAM giúp đạt băng thông cực đại mà không làm nghẽn bus PSRAM.
- **Octal PSRAM Workspace**: Cấp phát không gian trạng thái chuỗi (KV-cache FP32, hidden prefix states) và workspace tổng thể (~972 KB ở context 128).

### 2. Tăng Tốc Phép Nhân Ma Trận với Xtensa 128-bit EE SIMD
- **Tập lệnh**: Sử dụng chỉ thị `ee.vld.128.ip` (nạp vector 16-byte có auto-increment) và `ee.vmulas.s8.accx` (nhân tích chập 16-way signed 8-bit và tích lũy vào thanh ghi QACC 40-bit).
- **Căn chỉnh & Zero-Padding Invariant**: Địa chỉ vector bắt buộc chia hết cho 16. Mọi hàng kích thước $K$ được mở rộng thành $\text{stride} = (K + 15) \ \& \ \sim 15$ và điền số 0 ở các phần tử đệm $[K, \text{stride})$. Vì $0 \times 0 = 0$, vòng lặp vector chạy qua toàn bộ số block mà không làm sai lệch kết quả và không cần vòng lặp scalar xử lý đuôi.
- **Triệt tiêu Load-to-Use Latency**: Vòng lặp chính được mở rộng 4 block (64 phần tử), nạp luân phiên 8 thanh ghi vector `q0..q7` trước khi thực thi các lệnh MAC liên tiếp.
- **Cấm Hardware Loops trong Assembly**: Tuyệt đối không sử dụng chỉ thị `loop`/`loopnez` trong inline assembly để tránh phá hủy các thanh ghi vòng lặp phần cứng (`LBEG`, `LEND`, `LCOUNT`) của GCC; sử dụng vòng lặp rẽ nhánh phần mềm `addi` + `bnez`.

### 3. Giải Nén Trọng Số Tam Phân Bằng Bảng Tra Cứu (256-Entry LUT)
- Dùng bảng tra cứu tĩnh 256 phần tử (`s_unpack_lut`) ánh xạ 1 byte đóng gói (4 trọng số 2-bit với mã `00=-1, 01=0, 10=+1`) thành 4 giá trị signed `int8_t` trong một thao tác 32-bit.
- Khi hàng bắt đầu tại ranh giới byte, giải nén theo nhóm 4 trọng số; với các ranh giới lẻ, sử dụng hàm fallback bitwise để bảo đảm tính tương thích với định dạng RUVI0001.

### 4. Lượng Tử Hóa Offline INT8 Cho Các Lớp Non-Ternary (LM Head & Dense)
- Toàn bộ các ma trận trọng số không phải tam phân (`embed_tokens.weight` / `lm_head` và các lớp linear dense) được lượng tử hóa offline sang **Signed INT8** (`kind == 3`) với hệ số co giãn per-tensor `scale = max(|W|) / 127.0`.
- Dung lượng mô hình giảm từ **8.03 MB xuống còn 4.88 MB** (giảm 50% kích thước ma trận embedding).
- Cho phép tầng Output LM Head chạy trực tiếp trên **Xtensa 128-bit EE SIMD** mà không cần giải nén động trên chip.
- Giữ các vector RMSNorm ở FP32/BF16 vì kích thước rất nhỏ (~32 KB) và cần độ ổn định số học cao.

### 5. Lọc Không Gian Từ Vựng (Sub-Vocabulary Masking)
- Nhúng 2 mảng bitmask offline 1,024 byte (`SUBVOCAB_RU_MASK` và `SUBVOCAB_VI_MASK`) đại diện cho các token hợp lệ của tiếng Nga và tiếng Việt.
- Dựa vào language tag (`<2ru>` hoặc `<2vi>`), tầng Output bỏ qua các phép tính dot product của các token không thuộc ngôn ngữ đích, gán thẳng logit bằng `-1e9f`.
- Giảm **33.6% số phép tính** cho chiều `vi2ru` và **63.1% số phép tính** cho chiều `ru2vi`.

### 6. Tối Ưu Phép Tính Con Trỏ Trực Tiếp (Direct Pointer Arithmetic)
- Loại bỏ toàn bộ lời gọi hàm gián tiếp `weight->value(idx)` trong các vòng lặp nhân ma trận dense và tra cứu embedding.
- Duyệt trực tiếp qua con trỏ `const int8_t*` (cho INT8) hoặc trích xuất bit `const uint16_t*` (cho BF16), bảo toàn bit-exact thứ tự tích lũy số học.

### 7. Điều Phối Song Song Đa Lõi (Dual-Core N-Split Dispatcher)
- Tạo worker task cố định trên Core 1 bằng FreeRTOS API (`xTaskNotifyGive` / `ulTaskNotifyTake`) với độ trễ microsecond.
- Phân chia công việc theo chiều hàng ngõ ra $N$ ($N$-split): Core 0 tính $0 \dots N/2-1$, Core 1 tính $N/2 \dots N-1$.
- Mỗi core sở hữu một tile buffer độc lập trong SRAM nội bộ, ngăn ngừa hoàn toàn hiện tượng tranh chấp vùng nhớ.

### 8. Bảng RoPE Tính Toán Trước (Precomputed Continuous RoPE)
- Bảng giá trị $\cos(\theta)$ và $\sin(\theta)$ cho mọi vị trí trong ngữ cảnh được tính trước một lần tại thời điểm `bind_workspace`.
- Loại bỏ 100% các hàm siêu việt trong quá trình suy luận.

### 9. Nhúng Bộ BPE Tokenizer Chuẩn C++ (Embedded Tokenizer)
- Nhúng toàn bộ 8,192 token tiếng Việt/Nga vào Flash `.rodata` (64.7 KB chuỗi UTF-8 + 32 KB offset table).
- Giải mã từ Token ID sang ký tự UTF-8 với thời gian $O(1)$ không cấp phát heap.
- Tự động dừng tại `<eos>` và ghép chuỗi đầu ra trực tiếp trên MCU.

## Consequences & Measured Benchmarks

1. **Tốc độ sinh từ**:
   - Từ **1,583 ms/token (0.63 tok/s)** $\to$ **610 ms/token (1.64 tok/s)** (tăng tốc **2.60x**).
2. **Dung lượng mô hình**:
   - Từ **8.03 MB** $\to$ **4.88 MB** (vừa vặn trong phân vùng Flash và 8MB PSRAM).
3. **Bộ nhớ RAM trên chip**:
   - Internal SRAM: Dùng 42.3 KB cho tile buffer, còn trống **>331 KB**.
   - Octal PSRAM: Dùng 5.59 MB (chứa cả mô hình và workspace), còn trống **>2.52 MB**.
4. **Chất lượng bản dịch**:
   - Bảo toàn 100% độ chính xác greedy trên các câu thử nghiệm; loại bỏ hiện tượng ảo giác cắt cụt từ.
5. **Khả năng triển khai**:
   - Hệ thống có thể chạy hoàn toàn standalone trên ESP32-S3, tự tokenize và decode chuỗi văn bản không cần máy tính.
