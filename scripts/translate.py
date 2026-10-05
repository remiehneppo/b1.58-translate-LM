# scripts/translate.py
import argparse
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import PreTrainedTokenizerFast
from ruvi_bitnet import RuViBitNet10M, RuViConfig
from ruvi_bitnet.checkpoint import checkpoint_config, load_checkpoint_weights


def translate_text(
    text: str,
    direction: str,
    model: RuViBitNet10M,
    tokenizer: PreTrainedTokenizerFast,
    cfg: RuViConfig,
    device: torch.device,
) -> str:
    tag_id = cfg.tag_ru_id if direction == "vi2ru" else cfg.tag_vi_id
    src_ids = tokenizer.encode(text, add_special_tokens=False)

    # Prefix: [TAG] + src + [SEP]
    prompt_ids = torch.tensor([[tag_id] + src_ids + [cfg.sep_token_id]], dtype=torch.long, device=device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=not cfg.quantize_hidden_layers or any(getattr(m, "export_scale", None) is not None for m in model.modules())):
        out_ids = model.generate(prompt_ids, max_new_tokens=64, eos_token_id=cfg.eos_token_id)

    # Lấy các token sinh ra sau [SEP]
    generated_ids = out_ids[0, prompt_ids.shape[1] :].tolist()
    if cfg.eos_token_id in generated_ids:
        generated_ids = generated_ids[: generated_ids.index(cfg.eos_token_id)]

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ruvi_10m_checkpoint.pt")
    parser.add_argument("--tokenizer", default="ru_vi_tokenizer_8k")
    parser.add_argument("--text", type=str, default="Xin chào, bạn khỏe không?")
    parser.add_argument("--dir", choices=["vi2ru", "ru2vi"], default="vi2ru")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    cfg = checkpoint_config(ckpt)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    model = RuViBitNet10M(cfg).to(device)
    load_checkpoint_weights(model, ckpt)
    model.eval()

    res = translate_text(args.text, args.dir, model, tokenizer, cfg, device)
    print(f"[{args.dir.upper()}] Input:  {args.text}")
    print(f"[{args.dir.upper()}] Output: {res}")
