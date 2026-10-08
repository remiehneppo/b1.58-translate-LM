# Issue 04: Tích Hợp End-to-End Pipeline Dịch Giọng Nói (ASR -> Tokenizer -> Translator)

Status: ready-for-agent

## Context
Sau khi đã có module ASR (thu âm & nhận dạng tiếng Việt), C++ Tokenizer và module Translator (RuVi BitNet 10M), cần ghép nối toàn bộ thành một luồng hoàn chỉnh chạy liên tục trong vòng lặp chính của firmware, bảo đảm an toàn Watchdog và hiển thị kết quả.

## Acceptance Criteria
1. **Luồng End-to-End**:
   - Khi có giọng nói: Thu âm I2S $\to$ Chạy ASR $\to$ Nhận chuỗi tiếng Việt UTF-8 (ví dụ: `"Mở két sắt ra."`).
   - C++ Tokenizer chuyển chuỗi thành Token IDs kèm mã ngôn ngữ `<2ru>`: `[4, ...] + [3]`.
   - Translator thực hiện prefill và autoregressive decode (dual-core SIMD) $\to$ Nhận mảng Token IDs tiếng Nga.
   - C++ Tokenizer giải mã thành chuỗi tiếng Nga UTF-8: `"Открой сейф."`.
2. **Đo Lường Độ Trễ**:
   - Ghi nhận và xuất qua UART thời gian thực thi của từng công đoạn: Thu âm (s), ASR (ms), Tokenize (ms), Translate (ms), Tổng trễ phản hồi (ms).
   - Tổng trễ từ khi dứt lời nói đến khi có bản dịch $\le 6$ giây.
3. **An Toàn RTOS & Watchdog**:
   - Chèn `vTaskDelay(1)` giữa các pha và sau mỗi token decode để nhường CPU cho task IDLE và nạp Watchdog Timer (TWDT).
   - Hệ thống chạy liên tục 50 lần dịch mà không bị reset hay rò rỉ RAM.
4. **Mở Rộng Giao Diện**:
   - Xuất kết quả song ngữ qua UART serial chuẩn JSON.
   - (Tùy chọn) Gửi bản dịch qua SPI hiển thị màn hình LCD ST7789 hoặc I2C SSD1306.

## Comments
- Có thể bổ sung nút bấm (Push-to-talk) trên GPIO 0 hoặc GPIO 41 để người dùng bấm giữ khi nói.
