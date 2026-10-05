# Vi-Ru Edge Translator (10M BitNet b1.58)

Mô hình ngôn ngữ siêu nhỏ (SLM ~10M tham số) chuyên biệt cho tác vụ dịch máy song hướng Việt - Nga (Vi <-> Ru) tối ưu hóa cho thiết bị biên (Edge Devices).

## Language

**BitLinear Attention**:
Cơ chế Multi-Head Attention trong đó các ma trận chiếu (Q, K, V, Output) sử dụng trọng số ternary {-1, 0, 1} (BitLinear 1.58-bit) và kích hoạt 8-bit, trong khi phép tính Softmax và phân phối attention giữ độ chính xác chuẩn (FP16/INT8).
_Avoid_: Ternary Attention Map, Fully Ternary Attention

**Tied Embeddings**:
Kỹ thuật dùng chung một ma trận trọng số duy nhất giữa lớp embed_tokens và lm_head để tiết kiệm ngân sách tham số cho các transformer block.
_Avoid_: Untied Head, Separate LM Head

**Grouped-Query Attention (GQA)**:
Cấu hình attention gom nhóm các query heads chia sẻ chung ít key-value heads hơn (6 Q-heads, 2 KV-heads, head_dim=64), giúp giảm hai phần ba dung lượng KV-cache và băng thông bộ nhớ khi decode.
_Avoid_: Full MHA, MQA

**Prefix-LM Mask**:
Mặt nạ attention hỗn hợp cho Decoder-only: áp dụng chú ý 2 chiều (bidirectional) đầy đủ trên các token tiền tố (câu nguồn + mã ngôn ngữ), và áp dụng chú ý nhân quả (causal) tự hồi quy trên các token đích sinh ra.
_Avoid_: Pure Causal Mask, Full Bidirectional Mask

**Continuous RoPE**:
Cơ chế nhúng vị trí quay (Rotary Position Embedding) đánh số vị trí liên tục từ 0 đến N-1 trên toàn bộ chuỗi (tiền tố nguồn và mục tiêu đích) để tương thích tối đa với các runtime biên chuẩn.
_Avoid_: Reset RoPE, ALiBi

**Squared ReLU FFN**:
Khối Feed-Forward gồm 2 ma trận chiếu tuyến tính BitLinear (intermediate_size = 1280 ở bản 10M) với hàm phi tuyến ReLU^2, loại bỏ hoàn toàn các hàm phi tuyến siêu việt để tối ưu hóa cho phần cứng biên.
_Avoid_: SwiGLU, GELU FFN

**Language Tag**:
Token điều khiển đặc biệt đứng đầu chuỗi tiền tố (`<2ru>` hoặc `<2vi>`) chỉ thị hướng dịch mục tiêu, thay thế cho prompt văn bản tự nhiên để tiết kiệm tối đa chiều dài ngữ cảnh và tham số mô hình.
_Avoid_: Text Prompting, Natural Language Prefix

**BPE-8k Tokenizer**:
Bộ tokenizer Byte-fallback BPE kích thước chính xác 8,192 token (lũy thừa 2), được huấn luyện cân bằng 50% ngữ liệu tiếng Việt và 50% ngữ liệu tiếng Nga, đảm bảo 100% không bị token `<unk>`.
_Avoid_: Unigram-8k, Open-ended Vocab

**STE Distillation**:
Phương pháp huấn luyện trực tiếp mô hình trọng số 1.58-bit từ đầu bằng kỹ thuật Straight-Through Estimator kết hợp chưng cất tri thức (Knowledge Distillation) từ mô hình dịch máy giáo viên lớn hơn.
_Avoid_: Two-stage Pretrain-QAT, Scratch CE Only

**Sequence-Level KD**:
Phương pháp chưng cất tri thức bằng cách cho mô hình giáo viên dịch toàn bộ ngữ liệu để tạo nhãn giả (pseudo-labels) chất lượng cao cho học sinh học, giải quyết triệt để vấn đề lệch bộ từ vựng (mismatch vocabulary) giữa giáo viên (256k vocab) và học sinh (8k vocab).
_Avoid_: Word-Level Token KL, Cross-Vocab Logit Matching

**Sandwich Precision**:
Thiết kế kiến trúc giữ nguyên tầng Transformer đầu vào (`layers.0`) và tầng Transformer đầu ra (`layers.{n_layers-1}`; bản 10M là `layers.4`) ở độ chính xác cao (`nn.Linear` FP16/INT8), trong khi các tầng trung gian sử dụng `BitLinear` tam phân 1.58-bit. Thiết kế này bảo toàn entropy thông tin sau embedding và giữ sắc nét phân phối xác suất trước `lm_head`.
_Avoid_: Full-Ternary All-Layers, Uniform Quantization

## Relationships

- **BitLinear Attention** thực hiện các phép chiếu Q, K, V, O bằng **BitLinear b1.58** layer trước khi tính ma trận chú ý chuẩn qua Softmax.
- **Tied Embeddings** liên kết không gian vector đầu vào và đầu ra của mô hình; bản 10M dùng embedding ~3.15M tham số và 5 tầng Transformer.
- **Prefix-LM Mask** kết hợp cùng **Continuous RoPE** cho phép khối Decoder-only hoạt động như Encoder cho câu nguồn (full context) và Decoder cho câu đích (autoregressive) mà không cần thêm kiến trúc cross-attention riêng biệt.
- **Squared ReLU FFN** kết hợp **GQA** giúp bản gốc đạt 5 tầng với 10.033.792 tham số (d_model=384, d_ff=1280).

## Example dialogue

> **Dev:** "Liệu ta có lượng tử hóa ma trận Softmax attention xuống {-1, 0, 1} không?"
> **Domain expert:** "Không, ở quy mô 10M tham số, ma trận attention map phải giữ FP16/INT8 để đảm bảo alignment dịch thuật, chỉ các phép chiếu tuyến tính (Q, K, V, O) mới dùng trọng số ternary."
>
> **Dev:** "Tại sao chúng ta phải dùng Tied Embeddings thay vì tách riêng như LLaMA 3?"
> **Domain expert:** "Với vocab 8k và d_model 384, mỗi ma trận embedding tốn ~3.15M tham số. Dùng Tied Embeddings tránh nhân đôi chi phí này, giữ ngân sách cho 5 tầng Transformer."
>
> **Dev:** "Khi sinh từ trên thiết bị biên, Prefix-LM mask có làm chậm quá trình decode không?"
> **Domain expert:** "Không, khi decode sinh từng token (step-by-step), token mới chỉ cần attend vào toàn bộ KV-cache đã tính trước đó. Prefix mask chỉ hoạt động ở pha prefill câu nguồn."

## Flagged ambiguities

- "Ternary Attention" dễ gây hiểu nhầm là lượng tử hóa ma trận Softmax attention — đã thống nhất: đây là **BitLinear Attention** (chỉ lượng tử hóa các phép chiếu tuyến tính Q, K, V, O).
