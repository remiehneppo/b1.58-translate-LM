# Issue 03: Tích Hợp Micro I2S (INMP441) Tránh Xung Đột Chân Octal PSRAM

Status: ready-for-agent

## Context
Trên ESP32-S3, kích hoạt Octal PSRAM (bắt buộc để có 8MB RAM) sẽ chiếm dụng cố định các chân **GPIO 33, 34, 35, 36, 37**.
Nếu cố gắng cấu hình I2S vào các chân tài liệu mặc định (WS=36, BCLK=37), chip sẽ gặp sự cố phần cứng ngay khi khởi động (`rst:0x8 TG1WDT_SYS_RST`).

## Acceptance Criteria
1. Nối dây micro I2S INMP441 vào các chân GPIO còn tự do:
   - `I2S_PIN_WS` = **GPIO 4**
   - `I2S_PIN_BCLK` = **GPIO 5**
   - `I2S_PIN_DATA_IN` = **GPIO 6**
2. Cấu hình driver ESP-IDF I2S tiêu chuẩn:
   - Tần số lấy mẫu: 16,000 Hz, 1 kênh (Mono), định dạng 32-bit slot (24-bit dữ liệu từ INMP441, dịch phải 16 bit sang signed 16-bit PCM).
3. Tích hợp bộ lọc ngưỡng im lặng (Silence Gating / VAD đơn giản):
   - Nếu năng lượng âm thanh trong cửa sổ thu âm thấp hơn ngưỡng im lặng, bỏ qua không đưa vào ASR để tiết kiệm pin và CPU.
4. Kiểm tra thu âm 3s – 6s không gây xung đột bus hoặc treo hệ thống khi Octal PSRAM đang hoạt động liên tục.

## Comments
- Tham khảo mã nguồn thu âm và xử lý bit shift tại `esp32-kernel/main/asr_main.c`.
