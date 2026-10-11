# scripts/translate.py
import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

# Thêm thư mục gốc dự án vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import PreTrainedTokenizerFast
from ruvi_bitnet import RuViBitNet10M, RuViConfig
from ruvi_bitnet.checkpoint import checkpoint_config, load_checkpoint_weights

_SUBVOCAB_CACHE = {}


def get_subvocab_mask(tokenizer: PreTrainedTokenizerFast, direction: str, device: torch.device) -> torch.Tensor:
    cache_key = (id(tokenizer), direction, str(device))
    if cache_key in _SUBVOCAB_CACHE:
        return _SUBVOCAB_CACHE[cache_key]

    vocab_size = len(tokenizer)

    def has_cyrillic(t: str) -> bool:
        return any("CYRILLIC" in unicodedata.name(c, "") for c in t)

    def has_latin(t: str) -> bool:
        return any("LATIN" in unicodedata.name(c, "") for c in t)

    target_is_ru = direction == "vi2ru"
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for i in range(vocab_size):
        if i in (0, 1, 2, 3, 4, 5, 6):
            mask[i] = True
        else:
            s = tokenizer.decode([i])
            cyr = has_cyrillic(s)
            lat = has_latin(s)
            if target_is_ru:
                if (cyr and not lat) or (not cyr and not lat):
                    mask[i] = True
            else:
                if (lat and not cyr) or (not cyr and not lat):
                    mask[i] = True

    _SUBVOCAB_CACHE[cache_key] = mask
    return mask


def split_sentences(text: str) -> list[str]:
    """Tách đoạn văn thành các câu riêng biệt dựa trên dấu kết thúc câu."""
    raw_chunks = re.split(r"(?<=[.!?;\n])\s+", text.strip())
    return [c.strip() for c in raw_chunks if c.strip()]


def translate_segment(
    text: str,
    direction: str,
    model: RuViBitNet10M,
    tokenizer: PreTrainedTokenizerFast,
    cfg: RuViConfig,
    device: torch.device,
    max_new_tokens: int = 64,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    subvocab_mask: Optional[torch.Tensor] = None,
) -> str:
    tag_id = cfg.tag_ru_id if direction == "vi2ru" else cfg.tag_vi_id
    src_ids = tokenizer.encode(text, add_special_tokens=False)

    # Prefix: [TAG] + src + [SEP]
    prompt_ids = torch.tensor([[tag_id] + src_ids + [cfg.sep_token_id]], dtype=torch.long, device=device)

    # Đảm bảo max_new_tokens đủ lớn cho câu nguồn nhưng không vượt max_seq_len
    P = prompt_ids.shape[1]
    safe_max_new = min(max_new_tokens, cfg.max_seq_len - P)
    if safe_max_new <= 0:
        return ""

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=not cfg.quantize_hidden_layers
        or any(getattr(m, "export_scale", None) is not None for m in model.modules()),
    ):
        out_ids = model.generate(
            prompt_ids,
            max_new_tokens=safe_max_new,
            eos_token_id=cfg.eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            subvocab_mask=subvocab_mask,
        )

    # Lấy các token sinh ra sau [SEP]
    generated_ids = out_ids[0, prompt_ids.shape[1] :].tolist()
    if cfg.eos_token_id in generated_ids:
        generated_ids = generated_ids[: generated_ids.index(cfg.eos_token_id)]

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def translate_text(
    text: str,
    direction: str,
    model: RuViBitNet10M,
    tokenizer: PreTrainedTokenizerFast,
    cfg: RuViConfig,
    device: torch.device,
    max_new_tokens: int = 64,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    use_subvocab: bool = True,
    chunk_sentences: bool = True,
) -> str:
    subvocab_mask = get_subvocab_mask(tokenizer, direction, device) if use_subvocab else None

    if chunk_sentences:
        chunks = split_sentences(text)
        if len(chunks) > 1:
            translated_chunks = [
                translate_segment(
                    chunk,
                    direction,
                    model,
                    tokenizer,
                    cfg,
                    device,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    subvocab_mask=subvocab_mask,
                )
                for chunk in chunks
            ]
            return " ".join(translated_chunks)

    return translate_segment(
        text,
        direction,
        model,
        tokenizer,
        cfg,
        device,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        subvocab_mask=subvocab_mask,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ru-vi-tatoeba/checkpoint.pt")
    parser.add_argument("--tokenizer", default="ru-vi-tatoeba")
    parser.add_argument("--text", type=str, default="Xin chào, bạn khỏe không?")
    parser.add_argument("--dir", choices=["vi2ru", "ru2vi"], default="vi2ru")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--no-chunk", action="store_true", help="Không chia câu cho văn bản dài")
    parser.add_argument("--no-subvocab", action="store_true", help="Không áp dụng sub-vocabulary mask")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    cfg = checkpoint_config(ckpt)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    model = RuViBitNet10M(cfg).to(device)
    load_checkpoint_weights(model, ckpt)
    model.eval()

    res = translate_text(
        args.text,
        args.dir,
        model,
        tokenizer,
        cfg,
        device,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
        use_subvocab=not args.no_subvocab,
        chunk_sentences=not args.no_chunk,
    )
    print(f"[{args.dir.upper()}] Input:  {args.text}")
    print(f"[{args.dir.upper()}] Output: {res}")
