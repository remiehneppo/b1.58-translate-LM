# Issue 02: Trình Quản Lý Bộ Nhớ Octal PSRAM 2 Pha (Two-Phase Memory Manager)

Status: ready-for-agent

## Context
ESP32-S3 chỉ có 8 MB Octal PSRAM. 
- ASR yêu cầu ~7.06 MB workspace cho câu nói dài 6 giây (Model ASR bắt buộc chạy Flash XIP).
- Translator yêu cầu 4.88 MB mô hình nạp vào PSRAM + 0.97 MB KV-cache workspace = 5.85 MB.
Tổng nhu cầu nếu cùng nạp là >12.9 MB (gây tràn RAM ngay lập tức).
Do hai tác vụ chạy lần lượt (ASR $\to$ Dịch), ta cần một trình quản lý bộ nhớ PSRAM 2 pha (Two-Phase Memory Manager) nhằm chia sẻ vùng nhớ theo thời gian mà không gây phân mảnh heap.

## Acceptance Criteria
1. Thiết kế module `memory_manager.h / .cpp`:
   - `phase1_asr_enter(size_t window_seconds)`: Cấp phát ASR workspace (từ 3.16 MB cho 3s đến 7.06 MB cho 6s) từ PSRAM. Model ASR được ánh xạ trực tiếp từ Flash (`esp_partition_mmap`), không nạp vào PSRAM.
   - `phase1_asr_exit()`: Giải phóng toàn bộ ASR workspace, trả lại >7.5 MB PSRAM trống.
   - `phase2_trans_enter()`: Cấp phát 4.88 MB PSRAM nạp model Translator từ Flash, cấp phát 0.97 MB workspace KV-cache.
   - `phase2_trans_exit()`: Giải phóng model Translator và workspace KV-cache.
2. Kiểm tra `heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM)` sau mỗi chu kỳ chuyển pha: bộ nhớ phải hồi phục nguyên vẹn, không có rò rỉ bộ nhớ (memory leak).
3. Đảm bảo Internal SRAM tile buffer (42.3 KB) được giữ cố định hoặc tái sử dụng an toàn cho cả kernel ASR và Translator.

## Comments
- Việc dùng `heap_caps_aligned_alloc` với block lớn có thể gây phân mảnh nếu cấp phát nhiều lần; khuyến nghị dùng 1 con trỏ tĩnh hoặc pool cố định cho từng pha.
