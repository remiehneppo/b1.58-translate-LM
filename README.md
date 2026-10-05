# b1.58 Translate LM

Dự án nghiên cứu mô hình dịch Việt ↔ Nga dùng BitNet b1.58 và runtime C++ cho thiết bị biên. Mục tiêu hiện tại là chạy cùng model trên host và ESP32-S3; Luckfox Pico Pro và EBAZ4205 là các đích tiếp theo.

## Kiến trúc

- Decoder-only Prefix-LM: attention hai chiều cho câu nguồn, causal cho câu đích.
- BitLinear dùng trọng số tam phân `{-1, 0, 1}`, activation 8-bit và STE khi training.
- GQA, Continuous RoPE, RMSNorm, FFN ReLU² và tied embeddings.
- Sandwich precision: tầng đầu/cuối giữ trọng số float theo cấu hình; các tầng giữa dùng BitLinear.
- Cấu hình mặc định: vocab 8192, hidden 384, FFN 1280, 5 tầng, 6 Q-head/2 KV-head, tổng 10.033.792 tham số.

Tokenizer và dữ liệu đang được xây dựng. Checkpoint demo cũ có vocab 3021, hidden 256 và 11 tầng; benchmark ESP32 bên dưới dùng checkpoint này, không phải cấu hình mặc định 10M.

## Cài đặt và kiểm tra

Python 3.10+, PyTorch, cùng compiler C++17 nếu chạy kiểm tra native. Chọn bản PyTorch phù hợp CPU/CUDA của máy trước khi cài package.

```sh
python -m pip install -e ".[teacher]"
python -m unittest discover -s tests -v
```

Các test tự tạo model nhỏ và dữ liệu tạm; không cần dataset hoặc checkpoint demo. Test CUDA chạy khi có GPU. Bộ test native cần `g++`.

## Training và export

Các lệnh chạy từ root dự án. Chuẩn bị tokenizer và JSONL có `src`, `tgt`, `dir` (`vi2ru` hoặc `ru2vi`). Xem `python scripts/train.py --help` để chọn đường dẫn, variant và tham số training.

```sh
python scripts/train.py --data distill_vi_ru_tatoeba.jsonl \
  --tokenizer ru_vi_tokenizer_8k --output ruvi_10m_checkpoint.pt
python scripts/export_quantized.py --checkpoint ruvi_10m_checkpoint.pt \
  --output ruvi_10m_quantized.pt --pack_2bit
python scripts/translate.py --checkpoint ruvi_10m_quantized.pt \
  --tokenizer ru_vi_tokenizer_8k --dir vi2ru --text 'Xin chào'
```

Training hỗ trợ resume với optimizer, scheduler và RNG; validation NLL được tính theo số token đích. `export_bf16.py` xuất baseline không tam phân từ shadow weights. `compare.py` đối chiếu checkpoint trên cùng dữ liệu held-out. Checkpoint PyTorch có thể chứa pickle: chỉ nạp file tin cậy.

Teacher generation là bước tùy chọn; cài thêm `python -m pip install -e '.[teacher]'` để dùng script `generate_teacher_openai.py`; truyền key qua biến môi trường `OPENAI_API_KEY` hoặc `--api_key`. Dataset, checkpoint, firmware build và backup flash được giữ ngoài Git. Hai file tokenizer demo nhỏ được giữ để đối chiếu các artifact demo có sẵn trên máy.

## Native và ESP32

```sh
cmake -S native -B native/build
cmake --build native/build -j2
python scripts/export_native.py --checkpoint ruvi_10m_quantized.pt \
  --output native/artifacts/model.bin
native/build/ruvi-native --model native/artifacts/model.bin \
  --prompt 4,100,200,300,3 --context 128 --max-new 8
```

Runtime nhận token IDs; tokenize và giải mã văn bản hiện thực hiện trên host. Cùng image `RUVI0001` dùng trên mọi board, với packed ternary, BF16/FP32 cho phần float, scale FP32 và CRC32. Xem [hướng dẫn native và ESP-IDF](native/README.md) để build, export, kiểm tra parity và flash.

ESP32-S3 N16R8 đã hoàn tất một lượt inference trên CPU 240 MHz, model 6.076.240 byte đọc từ flash, workspace 1.594.676 byte trong PSRAM:

| Phép đo | Kết quả |
|---|---:|
| Load và kiểm tra image | 2,708 giây |
| Prefill 5 token | 5,535 giây |
| Decode 7 bước | 9,193 giây |
| Tốc độ decode | 0,761 token/giây |
| Tổng tính toán inference, sinh 8 token | 14,728 giây |

8 token đầu ra khớp runtime host. Đây là một lần đo với prefix tổng hợp trên kernel scalar, không đo chất lượng dịch hay phân phối latency. Bằng chứng: [JSON](docs/benchmarks/esp32-inference.json), [log](docs/benchmarks/esp32-inference.log). Luckfox/EBAZ chưa được chạy kiểm chứng.

## Cấu trúc

| Thư mục | Vai trò |
|---|---|
| `ruvi_bitnet/` | Model, dữ liệu Prefix-LM, checkpoint và packing dùng chung |
| `scripts/` | Training, export, inference, parity, benchmark và công cụ dữ liệu |
| `native/` | Runtime C++17, host CLI và firmware ESP-IDF |
| `tests/` | Regression training, export, cached decode và native parity |
| `docs/adr/` | Quyết định kiến trúc |
| `docs/benchmarks/` | Kết quả đo được giữ trong Git |

[CONTEXT.md](CONTEXT.md) mô tả thuật ngữ dự án.
