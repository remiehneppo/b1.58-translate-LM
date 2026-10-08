# Kế Hoạch Triển Khai: Hệ Thống Dịch Giọng Nói Ngoại Tuyến (ASR + Translator) Trên ESP32-S3

Tài liệu này tổng hợp kế hoạch hành động và lộ trình kỹ thuật để ghép nối 2 mô hình BitNet b1.58:
1. **Conformer BitNet ASR** (Nhận dạng giọng nói tiếng Việt từ micro I2S).
2. **RuVi 10M BitNet SLM** (Dịch máy câu văn bản sang tiếng Nga hoặc ngược lại).

Hệ thống theo dõi chi tiết (PRD và Issues) được lưu tại `.scratch/offline-speech-translation/`.

---

## 1. Ràng Buộc Phần Cứng Sống Còn Cần Nhớ

- **Dung lượng Flash (16 MB)**: Đủ chỗ cho cả 2 mô hình (ASR ~4.56 MB + Translator ~4.88 MB + App ~0.5 MB = ~9.94 MB < 16 MB).
- **Dung lượng Octal PSRAM (8 MB)**: 
  - ASR yêu cầu **~7.06 MB workspace** cho âm thanh 6 giây $\implies$ **Model ASR BẮT BUỘC phải đọc từ Flash (Flash XIP)**.
  - Sau khi ASR nhận dạng xong và trả về chuỗi văn bản, **giải phóng 100% ASR workspace**.
  - Translator sau đó có toàn bộ 8 MB PSRAM sạch sẽ: Nạp 4.88 MB mô hình dịch vào PSRAM + 0.97 MB KV-cache $\implies$ Dịch với tốc độ cực đại **610 ms / token**.
- **Chân nối Micro I2S (INMP441)**: Octal PSRAM chiếm dụng GPIO 33-37. Micro I2S **bắt buộc cắm vào GPIO 4 (WS), GPIO 5 (SCK), GPIO 6 (SD)**. Tuyệt đối không cắm vào GPIO 36/37.

---

## 2. Danh Sách Các Hạng Mục Công Việc (TODO Roadmap)

### Giai đoạn 1: Phân Vùng Flash & Đóng Gói Binary Blobs
- [ ] Cập nhật `native/esp32/partitions.csv`:
  - `0x010000`: `factory` (App firmware 1 MB)
  - `0x110000`: `asr_model` (ASR weights blob 5 MB)
  - `0x610000`: `trans_model` (RuVi INT8 weights blob 5 MB)
- [ ] Viết script nạp `tools/flash_dual_models.py` hoặc lệnh esptool nạp cả 2 blobs vào chip một lần.
- [ ] Chi tiết: `.scratch/offline-speech-translation/issues/01-flash-partitioning-and-model-blobs.md`.

### Giai đoạn 2: Trình Quản Lý Bộ Nhớ PSRAM 2 Pha (Time-Sharing Memory Manager)
- [ ] Xây dựng module `memory_manager.h / .cpp`:
  - **Pha 1 (ASR)**: Cấp phát 3.16 MB – 7.06 MB PSRAM cho ASR workspace (hỗ trợ âm thanh từ 3s đến 6s). Model ASR chạy Flash XIP qua `esp_partition_mmap`.
  - Kết thúc Pha 1: Giải phóng toàn bộ ASR workspace, thu hồi >7.5 MB PSRAM.
  - **Pha 2 (Translator)**: Cấp phát 4.88 MB PSRAM nạp model dịch + 0.97 MB KV-cache.
  - Kết thúc Pha 2: Giải phóng vùng nhớ Translator, trở về trạng thái chờ thu âm tiếp theo.
- [ ] Chi tiết: `.scratch/offline-speech-translation/issues/02-two-phase-psram-memory-manager.md`.

### Giai đoạn 3: Tích Hợp Micro I2S & Bộ Lọc Im Lặng
- [ ] Ghép driver I2S chuẩn ESP-IDF thu âm 16kHz Mono từ INMP441 trên GPIO 4/5/6.
- [ ] Tích hợp xử lý bit shift (24-bit left-aligned $\to$ signed 16-bit PCM).
- [ ] Tích hợp cơ chế VAD / Silence Threshold: tự động phát hiện khi người dùng bắt đầu nói và kết thúc nói để kích hoạt ASR.
- [ ] Chi tiết: `.scratch/offline-speech-translation/issues/03-i2s-mic-frontend-integration.md`.

### Giai đoạn 4: Ghép Nối Pipeline End-to-End & Giao Diện Người Dùng
- [ ] Ghép nối luồng liên tục:
  $$\text{I2S Audio} \xrightarrow{\text{ASR (Flash)}} \text{Text Tiếng Việt} \xrightarrow{\text{C++ Tokenizer}} \text{Token IDs} \xrightarrow{\text{RuVi (PSRAM)}} \text{Text Tiếng Nga}$$
- [ ] Đảm bảo an toàn Watchdog: gọi `vTaskDelay(1)` giữa các pha và sau mỗi token decode.
- [ ] Xuất log JSON đo đạc độ trễ chi tiết của từng bước qua UART.
- [ ] (Tùy chọn) Ghép nút bấm Push-to-Talk và hiển thị kết quả lên màn hình OLED/LCD nhỏ.
- [ ] Chi tiết: `.scratch/offline-speech-translation/issues/04-end-to-end-pipeline-and-ui.md`.

---

## 3. Dự Báo Trải Nghiệm & Thông Số Mục Tiêu

- **Thời lượng thu âm**: 1 đến 6 giây (tiếng Việt).
- **Độ trễ ASR**: ~1.8 – 2.2 giây cho câu 3 giây (Real-Time Factor ~0.6x).
- **Độ trễ Dịch máy**: ~3.5 giây cho câu 6 token (~610 ms/token).
- **Tổng độ trễ từ khi dứt lời đến khi có bản dịch**: **~5.5 – 6.0 giây**.
- **Tính độc lập**: Hoạt động 100% offline trên thiết bị cầm tay pin 3.7V.
