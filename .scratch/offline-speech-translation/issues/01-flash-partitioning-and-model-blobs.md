# Issue 01: Cấu hình Phân Vùng Flash 16MB và Nạp 2 Mô Hình (ASR & Translator)

Status: ready-for-agent

## Context
ESP32-S3 N16R8 sở hữu 16 MB SPI Flash. Để chạy cả 2 mô hình nhận dạng giọng nói (ASR 4.56 MB) và dịch máy (RuVi 4.88 MB), phân vùng Flash hiện tại cần được mở rộng để chứa cả 2 binary blobs tại các offset cố định, không đè lên firmware app hay NVS.

## Acceptance Criteria
1. File `native/esp32/partitions.csv` được cấu hình lại với các phân vùng:
   - `nvs`: 24 KB tại `0x9000`
   - `phy_init`: 4 KB tại `0xf000`
   - `factory`: 1 MB tại `0x10000`
   - `asr_model`: 5 MB tại `0x110000`
   - `trans_model`: 5 MB tại `0x610000`
   - `storage`: phần còn lại (~4.8 MB) cho tài nguyên hoặc lưu audio
2. Cập nhật script nạp `esptool`:
   - Ghi ASR blob (`model_blob.bin` từ dự án ASR) vào offset `0x110000`.
   - Ghi Translator blob (`ruvi_best_int8.bin`) vào offset `0x610000`.
3. Firmware đọc đúng header `RUVI0001` tại phân vùng `trans_model` và layout header ASR tại phân vùng `asr_model`.

## Comments
- Cần chú ý offset phân vùng phải căn chỉnh theo block 64 KB (Flash erase block size).
