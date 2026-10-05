# So sánh BF16 / tam phân

Checkpoint hiện có `ru-vi-tatoeba-demo/ruvi_10m_checkpoint.pt` chứa **trọng số FP32 shadow đã học bằng STE**, không phải baseline BF16 được huấn luyện độc lập. Bản BF16 xuất từ đây dùng cùng trọng số nhưng tắt cả lượng tử hóa trọng số lẫn activation ở các lớp BitLinear. Bản tam phân mặc định lưu trọng số đúng {-1, 0, 1} bằng BF16, lưu hệ số scale FP32 riêng và không lượng tử hóa lại lúc suy luận; `--pack_2bit` vẫn cho phép đóng gói tùy chọn. Đây là ablation tác động lượng tử hóa, không phải so sánh hai mô hình huấn luyện công bằng.

```bash
python scripts/export_bf16.py --checkpoint ru-vi-tatoeba-demo/ruvi_10m_checkpoint.pt
python scripts/export_quantized.py --checkpoint ru-vi-tatoeba-demo/ruvi_10m_checkpoint.pt
python scripts/compare.py --ternary ru-vi-tatoeba-demo/ruvi_10m_quantized.pt --bf16 ru-vi-tatoeba-demo/ruvi_10m_bf16.pt --data held_out.jsonl
python scripts/translate.py --checkpoint ru-vi-tatoeba-demo/ruvi_10m_bf16.pt --tokenizer ru-vi-tatoeba-demo --text 'Xin chào'
```

`compare.py` đo NLL trung bình theo token (nhỏ hơn là tốt hơn) trên **cùng các cặp** `src`, `tgt`, `dir`; dùng tập chưa thấy khi huấn luyện, không dùng tập train mặc định để kết luận chất lượng tổng quát. Cần `torch`, `transformers` và tokenizer đi kèm checkpoint. Dữ liệu gốc có 1.035.824 dòng, nên có thể dùng `--skip 900000 --samples 100` **chỉ nếu** xác nhận các dòng này chưa nằm trong tập train; nếu không, chuẩn bị `held_out.jsonl` riêng. NLL không thay thế đánh giá bản dịch thực tế (BLEU/COMET hoặc chấm tay).

Muốn baseline học độc lập: trên GPU CUDA hỗ trợ BF16 chạy `python scripts/train.py --bf16 --data TRAIN.jsonl --tokenizer ru-vi-tatoeba-demo --epochs 5 --seed 42` (mặc định ghi `ruvi_10m_bf16_trained.pt`). Chạy lại bản tam phân với **cùng dữ liệu, thứ tự, seed và hyperparameter**, đặt `--output` riêng; so sánh trên held-out. `--bf16` dùng autocast BF16, optimizer và master weights vẫn FP32; bản export lưu trọng số BF16 thực sự. Không resume chéo checkpoint hai kiến trúc.


## Model và training regression

Chạy `python -m unittest discover -s tests -v` để kiểm tra export/loader parity,
KD mask, EOS, context, scale serialization và train liên tục so với ngắt/resume.
Test CUDA tự bỏ qua khi không có GPU; test BF16 backward chạy khi GPU hỗ trợ.

`--max_seq_len` (mặc định 512) là giới hạn cứng của model: forward quá dài hoặc
prompt cộng số token yêu cầu vượt giới hạn sẽ báo lỗi, không âm thầm cắt nội dung.
Khi resume, dùng cùng cấu hình model; kéo dài `--epochs` sẽ cập nhật LR theo lịch mới
trước optimizer step đầu tiên. Checkpoint mới lưu RNG sau validation để resume đúng
thứ tự shuffle/dropout. Checkpoint cũ lưu RNG trước validation nên không bảo đảm
khớp từng bước với một run liên tục; best loss cũ được reset vì cách tính metric đã đổi.

Train CE dùng label smoothing đã cấu hình; validation NLL không smoothing và được
trung bình theo số target token để chọn best checkpoint.

Runtime logic dùng chung loader strict cho cả artifact packed/unpacked, giữ RMSNorm
và scale. Weight logic lưu int8 thay cho hai mask int32; mỗi intermediate bitwise được
giới hạn 512 KiB. Đây vẫn là reference kernel có vòng lặp, không phải kernel packed
native hay bằng chứng tốc độ trên edge. Loader vẫn giải nén weight packed trước khi chạy.
Trạng thái inference lưu lại có buffer `export_scale`; hai exporter chỉ nhận checkpoint
shadow để tránh lượng tử hóa lại hoặc diễn giải sai scale.
