# 0004. Giữ nguyên độ chính xác cao cho tầng Transformer cuối cùng (Layer L-1)

## Context
Tương tự như tầng đầu vào (Layer 0), tầng Transformer cuối cùng trước lớp chiếu Logits (`layers.10`) chịu trách nhiệm tổng hợp các biểu diễn trừu tượng thành các đặc trưng tách biệt rõ ràng để nhân với ma trận `lm_head`. Nếu lượng tử hóa tầng này về tam phân {-1, 0, 1}, phân phối xác suất đầu ra trên 8,192 token bị làm phẳng (flattened distribution), gây giảm độ tin cậy của token sinh ra và mất ổn định hàm mất mát Cross-Entropy.

## Decision
Áp dụng chiến lược "Sandwich Precision" kinh điển của mạng nhị phân/tam phân (BNN/TNN):
- Giữ nguyên cả tầng đầu tiên (`layers.0`) và tầng cuối cùng (`layers.10`) ở định dạng tuyến tính chuẩn (`nn.Linear` FP16/INT8).
- Toàn bộ 9 tầng trung gian (`layers.1` đến `layers.9`) sử dụng `BitLinear` tam phân 1.58-bit.
- Lớp `lm_head` vốn đã chia sẻ trọng số với `embed_tokens` nên cũng được giữ ở độ chính xác FP16/INT8.

## Consequences
- Cải thiện đáng kể độ sắc nét của phân phối xác suất từ vựng (logits distribution), giảm hiện tượng lặp từ và câu vô nghĩa khi sinh.
- Dung lượng checkpoint nén 1.58-bit tăng thêm ~1.4 MB (tổng dung lượng ~8.8 MB ở FP16 hoặc ~5.0 MB ở INT8), hoàn toàn nằm trong mục tiêu < 10 MB cho thiết bị biên.
- Tổng tham số mô hình giữ nguyên 10,037,120 (~10.04M).
