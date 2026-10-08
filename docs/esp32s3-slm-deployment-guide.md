# Hướng Dẫn Kỹ Thuật: Triển Khai Mô Hình Ngôn Ngữ Siêu Nhỏ (SLM) & BitNet b1.58 Trên ESP32-S3

Tài liệu này tổng hợp toàn bộ các quyết định kiến trúc, cạm bẫy phần cứng và kỹ thuật tối ưu hóa đã được kiểm chứng thực tế khi triển khai mô hình ngôn ngữ 10M tham số (Ru-Vi BitNet b1.58) trên vi điều khiển **ESP32-S3** (Xtensa dual-core LX7 @ 240 MHz, 16 MB Flash, 8 MB Octal PSRAM, ~512 KB Internal SRAM).

Tài liệu này được biên soạn làm cẩm nang chuẩn cho các dự án triển khai mạng nơ-ron (LLM/SLM, ASR, TTS, Vision) trên ESP32-S3 trong tương lai.

---

## 1. Bản Đồ Phần Cứng & Phân Tầng Bộ Nhớ (Memory Hierarchy)

ESP32-S3 có 3 tầng bộ nhớ với đặc tính truy xuất rất khác nhau:

| Tầng bộ nhớ | Dung lượng khả dụng | Băng thông thực tế | Độ trễ | Vai trò tối ưu trong suy luận |
|:---|:---:|:---:|:---:|:---|
| **Internal SRAM** | ~380 – 512 KB | Rất cao (~1 chu kỳ) | Cực thấp | **Tile buffer trọng số giải nén int8** (`tile_scratch`), activation buffer, vector accumulator |
| **Octal PSRAM** | 8 MB (8-bit @ 80MHz) | ~60 – 75 MB/s | Trung bình (~20-30 chu kỳ) | **Lưu toàn bộ mô hình (PSRAM Caching)**, KV-cache FP32, hidden prefix buffer |
| **SPI Flash (XIP)** | 16 MB (Quad @ 80MHz) | ~25 – 35 MB/s | Cao (qua Flash Cache) | Lưu trữ vĩnh viễn firmware, bảng vocab tokenizer, partition gốc |

### Bài học sống còn về Flash Cache Thrashing:
- Khi trọng số mô hình được đọc trực tiếp từ Flash (`esp_partition_mmap`), bus Flash và Flash Cache bị **chia sẻ giữa Instruction Cache (mã máy thực thi) và Data Cache (trọng số)**.
- Quá trình nhân ma trận liên tục nạp megabytes dữ liệu trọng số khiến các dòng cache mã máy bị tráo đổi liên tục (cache flush). CPU bị treo (stall) chờ nạp lại mã lệnh từ Flash.
- **Giải pháp**: Nếu PSRAM còn đủ chỗ (mô hình $\le 5$ MB trên bản 8MB PSRAM), hãy **sao chép toàn bộ mô hình vào Octal PSRAM khi khởi động** (`CONFIG_RUVI_MODEL_IN_PSRAM=y`). Bus Flash lúc này được giải phóng 100% cho Instruction Cache, tăng tốc độ decode ngay lập tức **~10–15%** và giữ thời gian thực thi ổn định tuyệt đối.

---

## 2. Quy Tắc Phần Cứng & Cạm Bẫy Hợp Ngữ (Hardware Pitfalls)

Khi viết kernel tăng tốc bằng tập lệnh mở rộng Xtensa 128-bit Energy-Efficient (EE) SIMD, **bắt buộc tuân thủ 4 quy tắc vàng**:

### Quy tắc 1: CẤM TUYỆT ĐỐI chỉ thị `loop` / `loopnez` trong Inline Assembly
- **Hiện tượng**: Xtensa LX7 chỉ có duy nhất **một bộ thanh ghi vòng lặp phần cứng** (`LBEG`, `LEND`, `LCOUNT`). Trình biên dịch GCC tự động phát ra các vòng lặp phần cứng cho các vòng lặp C/C++ bên ngoài (duyệt batch, layer, channel).
- **Hậu quả**: Nếu khối assembly của bạn dùng `loop` hoặc `loopnez`, các thanh ghi này sẽ bị ghi đè ngầm. Khi assembly trả về, vòng lặp C của hàm cha bị phá hủy, dẫn đến treo chip hoặc ghi đè bộ nhớ mà không báo lỗi compiler.
- **Quy chuẩn**: Luôn sử dụng vòng lặp rẽ nhánh phần mềm bằng cờ:
  ```assembly
  addi %[counter], %[counter], -1
  bnez %[counter], 1b
  ```

### Quy tắc 2: Ranh giới căn chỉnh 16-byte (16-Byte Alignment) & Zero-Padding
- **Hạn chế**: Lệnh nạp vector `ee.vld.128.ip` **bắt buộc địa chỉ bộ nhớ phải chia hết cho 16**. Nếu nạp từ địa chỉ lẻ, chip sẽ kích hoạt ngoại lệ phần cứng (LoadStoreAlignmentCause) hoặc đọc dữ liệu rác.
- **Xử lý số chiều lẻ**: Chiều vector $K$ không phải lúc nào cũng chia hết cho 16.
  $$\text{stride} = (K + 15) \ \& \ \sim 15$$
  Mọi hàng trọng số và vector kích hoạt phải được cấp phát với kích thước `stride` và **điền số 0 vào các phần tử đệm $[K, \text{stride})$**.
- **Lợi ích toán học**: Vì $0 \times 0 = 0$, vòng lặp vector chạy qua toàn bộ số block $\text{stride} / 16$ mà kết quả tích chập hoàn toàn bit-exact, **loại bỏ 100% vòng lặp scalar xử lý phần đuôi**.

### Quy tắc 3: Xoay vòng thanh ghi 4x (4x Register Rotation) triệt tiêu Load Latency
- Lệnh nạp vector vào thanh ghi `q0..q7` có độ trễ trước khi dữ liệu sẵn sàng cho phép nhân.
- Mở rộng vòng lặp nạp luân phiên 4 block (64 phần tử) liên tiếp:
  ```assembly
  ee.vld.128.ip q0, a2, 16
  ee.vld.128.ip q1, a3, 16
  ee.vld.128.ip q2, a2, 16
  ee.vld.128.ip q3, a3, 16
  ee.vld.128.ip q4, a2, 16
  ee.vld.128.ip q5, a3, 16
  ee.vld.128.ip q6, a2, 16
  ee.vld.128.ip q7, a3, 16
  ee.vmulas.s8.accx q0, q1
  ee.vmulas.s8.accx q2, q3
  ee.vmulas.s8.accx q4, q5
  ee.vmulas.s8.accx q6, q7
  ```
  Kỹ thuật này lấp đầy pipeline của CPU, giúp đạt thông lượng đỉnh **~320 MMAC/s mỗi core**.

### Quy tắc 4: Giới hạn bộ tích lũy 40-bit (QACC Headroom)
- Lệnh `ee.vmulas.s8.accx` tích lũy tích signed 8-bit vào thanh ghi QACC 40-bit.
- Giá trị tích cực đại của 1 cặp phần tử: $127 \times 127 = 16,129$.
- Với bất kỳ lớp nơ-ron nào có chiều $K \le 4096$, tổng tích lũy tối đa:
  $$\text{Max Acc} = 4096 \times 16,129 \approx 6.6 \times 10^7 < 2^{27}$$
  Con số này nằm trọn trong số nguyên có dấu 32-bit (`int32_t`). Do đó, chỉ cần đọc thanh ghi `rur.accx_0` là kết quả toán học chính xác 100%, không cần xử lý tràn số hay đọc thanh ghi phụ `accx_1`.

---

## 3. Chiến Lược Lượng Tử Hóa & Đóng Gói Trọng Số (Quantization & Binary Layout)

### 3.1. Đóng gói tam phân 2-bit (Ternary 2-Bit Packing)
- Mã hóa 2 bit / trọng số, 4 trọng số / byte:
  - `00` $\to -1$
  - `01` $\to 0$
  - `10` $\to +1$
  - `11` $\to$ Dự phòng / Bất hợp lệ (Trình nạp từ chối ngay để chống hỏng dữ liệu).
- **Giải nén tốc độ cao bằng bảng 256 phần tử (`s_unpack_lut`)**:
  Thay vì dùng các phép dịch bit và trừ scalar (`(b >> (i * 2) & 3) - 1`) tốn hàng tá lệnh CPU, bảng tra cứu `s_unpack_lut` ánh xạ 1 byte thành 4 giá trị signed `int8_t` trong **1 lệnh đọc/ghi 32-bit duy nhất**.

### 3.2. Lượng tử hóa Offline INT8 cho các lớp Non-Ternary (Embedding & LM Head)
- **Vấn đề**: Trong các mô hình ngôn ngữ (như BitNet b1.58), ma trận nhúng từ vựng `embed_tokens` thường dùng chung với `lm_head` (Tied Embeddings). Với bộ từ vựng 8k và $d_{model}=384$, ma trận này có 3.15 triệu tham số. Nếu giữ BF16, nó chiếm 6.29 MB và CPU phải nhân float scalar từng phần tử rất chậm.
- **Quy chuẩn**: Lượng tử hóa offline ma trận này sang **Signed INT8** (`kind == 3`) với hệ số co giãn per-tensor $scale = \max(|W|) / 127.0$:
  - Giảm dung lượng từ **6.29 MB xuống 3.15 MB**.
  - Không cần giải nén trên chip: dữ liệu nạp trực tiếp vào tập lệnh SIMD 128-bit `ee.vmulas.s8.accx`.
  - Độ suy hao số học cực nhỏ ($< 0.003$), bảo toàn nguyên vẹn 100% token đầu ra.

---

## 4. Tối Ưu Hóa Thuật Toán & Luồng Suy Luận (Algorithmic Optimizations)

### 4.1. Lọc không gian từ vựng (Sub-Vocabulary Masking)
- Đối với các tác vụ có phân tách ngôn ngữ mục tiêu (như dịch thuật song hướng, nhận dạng giọng nói theo ngữ cảnh):
  - Nhúng bảng bitmask offline đại diện cho các token hợp lệ của ngôn ngữ mục tiêu.
  - Khi sinh từ, tầng Output LM Head bỏ qua hoàn toàn các phép tính dot product của các token không thuộc ngôn ngữ đích, gán thẳng logit bằng `-1e9f`.
  - **Hiệu quả**: Giảm ngay **33% – 63% khối lượng tính toán** của tầng LM Head mà không ảnh hưởng đến độ chính xác.

### 4.2. Khử lời gọi hàm bằng Direct Pointer Arithmetic
- Trong các vòng lặp nhân ma trận lớn (hàng triệu lượt lặp), tuyệt đối không gọi hàm lấy giá trị trung gian (như `tensor.value(index)`).
- Ép kiểu con trỏ trực tiếp sang `const int8_t*` (với INT8) hoặc trích xuất bit `const uint16_t*` (với BF16) ngay trong thân vòng lặp. Việc này giúp tiết kiệm 10–15 chu kỳ CPU cho mỗi phần tử, giảm hàng trăm mili-giây trên mỗi token decode.

### 4.3. Bảng RoPE tính toán trước (Precomputed Continuous RoPE)
- Tránh gọi các hàm toán học siêu việt (`std::pow`, `std::sin`, `std::cos`) trong vòng lặp giải mã.
- Tính toán trước toàn bộ bảng $\sin(\theta)$ và $\cos(\theta)$ cho mọi vị trí ngữ cảnh một lần duy nhất khi cấp phát workspace.

### 4.4. Fused RMSNorm & Quantization
- Trong khối Multi-Head Attention, vector hidden state được chuẩn hóa RMSNorm một lần và lượng tử hóa sang vector int8 một lần dùng chung cho cả 3 phép chiếu Q, K, V.

---

## 5. Song Song Hóa Đa Lõi (Dual-Core $N$-Split Parallelism)

ESP32-S3 sở hữu 2 lõi CPU LX7 @ 240 MHz. Để khai thác trọn vẹn cả 2 lõi mà không gây xung đột:

1. **Khởi tạo Worker Task cố định trên Core 1**:
   Sử dụng FreeRTOS API `xTaskCreatePinnedToCore` để ghim worker task lên Core 1, đồng bộ hóa bằng `xTaskNotifyGive` và `ulTaskNotifyTake` (độ trễ microsecond, không tốn tài nguyên chờ).
2. **Chiến lược phân chia $N$-split**:
   - Chia ma trận theo **chiều hàng ngõ ra $N$** (output features), KHÔNG chia theo chiều tích lũy $K$.
   - Core 0 tính các hàng $0 \dots N/2 - 1$.
   - Core 1 tính song song các hàng $N/2 \dots N - 1$.
   - **Bảo toàn bit-exact**: Mỗi phần tử đầu ra giữ nguyên vẹn thứ tự cộng dồn số học, không có hiện tượng sai khác số thực giữa đơn lõi và đa lõi.
3. **Bộ nhớ Scratch độc lập**:
   Mỗi lõi phải sở hữu một tile buffer riêng biệt trong Internal SRAM (`tile_scratch_[0]` và `tile_scratch_[1]`). Tuyệt đối không cho 2 lõi chia sẻ buffer ghi.

---

## 6. Tính Ổn Định Hệ Thống & Watchdog (Production Safety)

1. **Task Watchdog Timer (TWDT)**:
   - Một lượt suy luận mô hình có thể kéo dài hàng trăm mili-giây đến vài giây.
   - Để tránh kích hoạt ngắt Watchdog làm reset chip, chèn `vTaskDelay(1)` sau mỗi token sinh ra hoặc ở ranh giới giữa các tầng Transformer.
   - Việc này nhường 1 tick cho FreeRTOS scheduler chạy task IDLE, nạp lại watchdog và xử lý các ngắt ngoại vi (Wi-Fi, Bluetooth, nút bấm, màn hình).
2. **Kiểm thử bộ nhớ an toàn (Zero Heap Allocations)**:
   - Toàn bộ bộ nhớ phục vụ suy luận phải được cấp phát một lần duy nhất tại `bind_workspace()` và `bind_scratch()`.
   - Trong quá trình `prefill()` và `decode()`, số lượng lời gọi `malloc`, `new`, `free` phải bằng **0**.
   - Kiểm tra rò rỉ bộ nhớ định kỳ bằng `heap_caps_get_minimum_free_size()`.

---

## 7. Bảng Số Liệu Đo Lường Thực Tế Trên ESP32-S3 (Model 10M BitNet)

Các kết quả đo đạc thực tế trên phần cứng ESP32-S3 (QFN56 r0.2, 8MB Octal PSRAM @ 80MHz, 16MB Flash @ 80MHz) qua từng giai đoạn tối ưu:

| Cấu hình triển khai | Thời gian nạp | Decode ms/token | Tốc độ sinh từ | RAM SRAM trống | RAM PSRAM trống |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline (Scalar C++, Flash XIP, 1 Core)** | 3,516 ms | **1,583 ms** | 0.63 tok/s | 356 KB | 6.78 MB |
| **2. Dual-Core SIMD (BF16 Head, Flash XIP)** | 3,516 ms | **1,160 ms** | 0.86 tok/s | 331 KB | 7.40 MB |
| **3. Dual-Core SIMD + INT8 Head + SubVocab (Flash)** | 1,644 ms | **662 ms** | 1.51 tok/s | 331 KB | 7.40 MB |
| **4. Toàn bộ Model lưu trên Octal PSRAM (Hoàn thiện)** | **1,934 ms** | **610 ms** | **1.64 tok/s** | **331 KB** | **2.52 MB** |

> **Tổng kết**: Tốc độ sinh từ tăng từ **1,583 ms $\to$ 610 ms/token (tăng tốc 2.60x)**. Dung lượng mô hình giảm từ **8.03 MB $\to$ 4.88 MB (giảm 40%)**. Bản dịch tiếng Việt $\leftrightarrow$ Nga đạt độ chính xác cao và chạy hoàn toàn độc lập trên vi điều khiển.
