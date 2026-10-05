# 0001. Sử dụng 2-Linear FFN với Squared ReLU thay vì SwiGLU

## Context
Kiến trúc BitNet b1.58 mặc định sử dụng SwiGLU FFN (3 ma trận chiếu và hàm SiLU). Tuy nhiên, mô hình được thiết kế để triển khai trên các thiết bị biên cực đoan (MCU/NPU/CPU nhúng) với ngân sách giới hạn 10M tham số.

## Decision
Sử dụng kiến trúc 2-Linear FFN với hàm kích hoạt Squared ReLU (`ReLU(x)^2`, intermediate_size = 1024, d_model = 256) thay vì SwiGLU 3 ma trận.

## Consequences
- Giảm số phép nhân ma trận FFN từ 3 xuống 2 ma trận mỗi tầng, tiết kiệm bộ nhớ và độ trễ tính toán.
- Loại bỏ hoàn toàn các phép toán siêu việt (exp/sigmoid trong SiLU), chỉ cần phép so sánh ngưỡng và nhân số nguyên, tối ưu tối đa cho bộ xử lý biên.
- Cho phép mở rộng mô hình lên 11 tầng sâu trong cùng ngân sách 10M tham số.
