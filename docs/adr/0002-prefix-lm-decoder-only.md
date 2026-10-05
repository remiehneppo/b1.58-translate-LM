# 0002. Sử dụng kiến trúc Decoder-Only với Prefix-LM Mask thay cho Encoder-Decoder

## Context
Mô hình dịch máy truyền thống (như MarianMT, NLLB-200) thường sử dụng kiến trúc Encoder-Decoder với các khối Cross-Attention riêng biệt. Tuy nhiên, ở ngân sách cực nhỏ 10M tham số trên thiết bị biên, việc phân chia tham số thành 2 khối tách biệt làm giảm chiều sâu của từng khối.

## Decision
Sử dụng kiến trúc thuần Decoder-only kết hợp Prefix-LM Mask (chú ý 2 chiều ở tiền tố câu nguồn, chú ý nhân quả ở câu đích) và Continuous RoPE.

## Consequences
- Tận dụng 100% tham số (toàn bộ 11 tầng) cho cả pha mã hóa ngữ cảnh nguồn lẫn pha sinh câu dịch.
- Loại bỏ hoàn toàn các ma trận Cross-Attention riêng biệt, giúp tối đa hóa kích thước hidden và chiều sâu mạng.
- Đơn giản hóa cơ chế KV-cache trong quá trình decode trên runtime biên (tương tự các LLM hiện đại).
