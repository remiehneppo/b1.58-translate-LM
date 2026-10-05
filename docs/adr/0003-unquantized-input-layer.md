# 0003. Giữ nguyên độ chính xác cao cho tầng Transformer đầu vào (Layer 0)

## Context
Trong kiến trúc mạng trọng số cực thấp (Binary / Ternary Neural Networks), các vector đặc trưng sau lớp embedding mang lượng entropy cao và biểu diễn liên tục. Nếu lượng tử hóa ngay lập tức các ma trận tuyến tính (Q, K, V, O và FFN) ở tầng đầu tiên về tam phân {-1, 0, 1}, mạng sẽ gặp hiện tượng nghẽn cổ chai biểu diễn (representation bottleneck) và gradient ở các bước đầu bị suy hao nặng nề.

## Decision
Giữ nguyên tầng Transformer đầu tiên (`layers.0`) ở định dạng tuyến tính chuẩn (`nn.Linear` độ chính xác FP16 / INT8), không lượng tử hóa về tam phân 1.58-bit. Các tầng tiếp theo (`layers.1` đến `layers.10`) tiếp tục sử dụng `BitLinear` 1.58-bit.

## Consequences
- Loại bỏ hiện tượng mất mát thông tin đột ngột ngay sau tầng embedding, giúp mô hình hội tụ nhanh và ổn định hơn đáng kể.
- Chi phí lưu trữ tăng nhẹ (~720 KB ở định dạng INT8 hoặc ~1.4 MB ở FP16 cho toàn bộ Layer 0), nhưng tổng dung lượng mô hình sau khi nén vẫn chỉ ~4-7 MB, hoàn toàn đáp ứng ngưỡng biên (< 10 MB).
- Tổng số lượng tham số mô hình giữ nguyên ở mức ~10.04M.
