# scripts/train_tokenizer.py
import argparse
import os
import sys

try:
    from tokenizers import Tokenizer, normalizers, processors
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast
except ImportError:
    print(
        "[Error] Cần cài đặt thư viện 'tokenizers' và 'transformers' trước:\n"
        "        pip install tokenizers transformers\n"
    )
    sys.exit(1)


def train_ru_vi_tokenizer(
    corpus_file: str = "corpus_tokenizer.txt",
    output_dir: str = "ru_vi_tokenizer_8k",
    vocab_size: int = 8192,
):
    if not os.path.exists(corpus_file) or os.path.getsize(corpus_file) == 0:
        raise FileNotFoundError(f"File corpus không tồn tại hoặc rỗng: {corpus_file}")
    if vocab_size < 100:
        raise ValueError("vocab_size phải lớn hơn hoặc bằng 100")

    print(f"[*] Bắt đầu huấn luyện Byte-fallback BPE trên '{corpus_file}'...")
    print(f"[*] Mục tiêu vocab size: {vocab_size}")

    # 1. Khởi tạo BPE model
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 2. Khai báo Special Tokens theo thứ tự định sẵn:
    # 0: <pad>, 1: <bos>, 2: <eos>, 3: <sep>, 4: <2ru>, 5: <2vi>, 6: <unk>
    special_tokens = ["<pad>", "<bos>", "<eos>", "<sep>", "<2ru>", "<2vi>", "<unk>"]

    # 3. Cấu hình Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    # 4. Huấn luyện
    tokenizer.train([corpus_file], trainer)

    # 5. Đóng gói PreTrainedTokenizerFast
    os.makedirs(output_dir, exist_ok=True)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        additional_special_tokens=["<sep>", "<2ru>", "<2vi>"],
    )

    hf_tokenizer.save_pretrained(output_dir)
    print(f"[+] Huấn luyện thành công! Đã lưu tokenizer tại: {output_dir}")

    # 6. Kiểm tra encode / decode mẫu
    test_phrase = "<2vi> Привет! Cảm ơn bạn rất nhiều.<sep>"
    enc = hf_tokenizer.encode(test_phrase)
    tokens = hf_tokenizer.convert_ids_to_tokens(enc)
    dec = hf_tokenizer.decode(enc)

    print("\n--- Kiểm tra Tokenizer ---")
    print(f"Input:   {test_phrase}")
    print(f"IDs:     {enc}")
    print(f"Tokens:  {tokens}")
    print(f"Decoded: {dec}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Byte-fallback BPE Tokenizer 8,192 cho Vi-Ru")
    parser.add_argument("--corpus", type=str, default="corpus_tokenizer.txt", help="Đường dẫn file corpus text")
    parser.add_argument("--output_dir", type=str, default="ru_vi_tokenizer_8k", help="Thư mục lưu tokenizer")
    parser.add_argument("--vocab_size", type=int, default=8192, help="Kích thước từ vựng")
    args = parser.parse_args()

    train_ru_vi_tokenizer(args.corpus, args.output_dir, args.vocab_size)
