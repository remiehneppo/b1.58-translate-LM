# PRD: Tích Hợp Pipeline Dịch Giọng Nói Ngoại Tuyến (Offline Speech-to-Speech / Speech-to-Text) Trên ESP32-S3

Status: ready-for-agent

## 1. Mục Tiêu (Objective)
Xây dựng hệ thống dịch giọng nói cầm tay hoàn toàn độc lập (Standalone Offline Pocket Speech Translator) trên vi điều khiển ESP32-S3 N16R8 (16MB Flash, 8MB Octal PSRAM), kết hợp 2 mô hình BitNet b1.58:
1. **Model 1: Conformer BitNet ASR** (Nhận dạng tiếng Việt từ micro I2S, cửa sổ âm thanh lên đến 6 giây).
2. **Model 2: RuVi 10M BitNet SLM** (Dịch máy câu văn bản tiếng Việt sang tiếng Nga hoặc ngược lại).

Toàn bộ hệ thống chạy tuần tự, offline 100%, không cần kết nối Wi-Fi hay máy chủ đám mây.

---

## 2. Ràng Buộc Phần Cứng & Kỹ Thuật (Constraints)

1. **Ràng buộc Flash ROM (16 MB)**:
   - Tổng dung lượng của cả 2 mô hình là ~9.44 MB (ASR 4.56 MB + Translator 4.88 MB).
   - Firmware app: ~0.5 MB.
   - Bắt buộc cấu hình phân vùng tùy biến `partitions.csv`:
     - `factory`: 1 MB (`0x010000`)
     - `asr_model`: 5 MB (`0x100000`)
     - `trans_model`: 5 MB (`0x600000`)
     - Phân vùng còn lại (~5 MB): NVS, SPIFFS/Storage.
2. **Ràng buộc Octal PSRAM (8 MB) — Bài học ASR sống còn**:
   - Model ASR cho cửa sổ âm thanh 6 giây yêu cầu **workspace lên tới ~7.06 MB (6.74 MiB)**.
   - Do đó, **Model ASR BẮT BUỘC phải chạy Flash XIP** (đọc trực tiếp từ Flash qua `esp_partition_mmap`), không được copy vào PSRAM vì $4.56\text{ MB} + 7.06\text{ MB} = 11.62\text{ MB} > 8\text{ MB}$ sẽ gây tràn RAM.
   - Giải pháp: **Two-Phase PSRAM Time-Sharing (Chia sẻ RAM 2 pha)**:
     - **Pha 1 (ASR)**: Cấp phát ~7.1 MB PSRAM cho ASR workspace. Sau khi ra text, giải phóng 100% PSRAM.
     - **Pha 2 (Translator)**: Cấp phát 4.88 MB PSRAM nạp model dịch + 0.97 MB KV-cache. Dịch xong, giải phóng PSRAM.
3. **Ràng buộc Chân GPIO (Octal PSRAM Conflict)**:
   - Octal PSRAM chiếm dụng các chân GPIO 33, 34, 35, 36, 37.
   - Micro I2S (INMP441) **BẮT BUỘC nối vào các chân tự do** (khuyến nghị: WS=4, SCK=5, SD=6), cấm cắm vào GPIO 36/37.
4. **Tập Lệnh & Tăng Tốc**:
   - Dùng chung kernel Xtensa 128-bit EE SIMD (`ee.vmulas.s8.accx`) và 42.3 KB Internal SRAM tile scratch cho cả 2 mô hình.
   - Điều phối song song 2 lõi CPU (Core 0 và Core 1).

---

## 3. Kiến Trúc Luồng Thực Thi (Execution Pipeline)

```text
[Người dùng bấm nút / Nói vào Micro I2S]
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ PHA 1: THU ÂM & ASR (VIETNAMESE SPEECH-TO-TEXT)         │
│ 1. Cấp phát ASR Workspace ~3.16 MB - 7.06 MB trong PSRAM│
│ 2. Thu âm 16kHz PCM qua I2S (1 - 6 giây)               │
│ 3. Trích xuất Log-Mel Filterbank                       │
│ 4. Chạy Conformer BitNet Encoder (Flash XIP)           │
│ 5. CTC Greedy Decoder nhả chuỗi UTF-8 tiếng Việt       │
│ 6. GIẢI PHÓNG TOÀN BỘ ASR WORKSPACE                    │
└────────────────────────────────────────────────────────┘
                 │
                 ▼ (Chuỗi tiếng Việt UTF-8: "Xin chào, bạn khỏe không?")
┌────────────────────────────────────────────────────────┐
│ PHA 2: DỊCH THUẬT (TEXT-TO-TEXT TRANSLATION)           │
│ 1. C++ Tokenizer tách chuỗi thành Token IDs (< 1 ms)   │
│ 2. Cấp phát 4.88 MB PSRAM nạp model + 0.97 MB KV-cache │
│ 3. Prefill Prefix-LM (Dual-core SIMD)                  │
│ 4. Autoregressive Decode với Sub-Vocab Masking         │
│ 5. C++ Tokenizer giải mã Token IDs -> UTF-8 tiếng Nga  │
│ 6. GIẢI PHÓNG PSRAM TRANSLATOR                         │
└────────────────────────────────────────────────────────┘
                 │
                 ▼ (Chuỗi tiếng Nga: "Привет, ты хорошо себя чувствуешь?")
[Hiển thị màn hình OLED/LCD / Gửi qua Serial / Phát âm thanh DAC]
```

---

## 4. Kế Hoạch Triển Khai (Issue Breakdown)

1. `issues/01-flash-partitioning-and-model-blobs.md`: Tái cấu trúc partition table, tích hợp nạp cả 2 binary blobs ASR và Translator vào Flash 16MB.
2. `issues/02-two-phase-psram-memory-manager.md`: Xây dựng trình quản lý cấp phát / thu hồi PSRAM 2 pha an toàn, ngăn ngừa phân mảnh heap (fragmentation).
3. `issues/03-i2s-mic-frontend-integration.md`: Ghép driver micro I2S INMP441 trên các chân GPIO an toàn, xử lý silence gating / VAD.
4. `issues/04-end-to-end-pipeline-and-ui.md`: Tích hợp logic liên kết ASR $\to$ Translator, hiển thị bản dịch qua UART / màn hình LCD, bảo đảm Watchdog an toàn.
