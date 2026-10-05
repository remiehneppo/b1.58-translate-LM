"""Compare any two trusted local RuVi checkpoints on the same held-out translation pairs."""
import argparse
from collections import Counter
from contextlib import nullcontext
from itertools import islice
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from ruvi_bitnet import RuViBitNet10M
from ruvi_bitnet.distill import TranslationPrefixDataset, collate_prefix_fn
from ruvi_bitnet.model import BitLinear
from ruvi_bitnet.checkpoint import checkpoint_config, load_checkpoint_weights


def char_f1(pred, ref):
    """Character unigram F1; a lightweight overlap proxy, not chrF/BLEU/COMET."""
    a, b = Counter(pred.replace(" ", "")), Counter(ref.replace(" ", ""))
    return 2 * sum((a & b).values()) / (sum(a.values()) + sum(b.values())) if a or b else 1.0


def evaluate(path, samples, tokenizer, device, batch_size, max_new_tokens):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # Trusted local checkpoints only.
    cfg = checkpoint_config(ckpt)
    special_ids = (tokenizer.pad_token_id, tokenizer.eos_token_id,
                   tokenizer.convert_tokens_to_ids("<sep>"), tokenizer.convert_tokens_to_ids("<2ru>"),
                   tokenizer.convert_tokens_to_ids("<2vi>"))
    if cfg.vocab_size != len(tokenizer) or (cfg.pad_token_id, cfg.eos_token_id, cfg.sep_token_id,
                                            cfg.tag_ru_id, cfg.tag_vi_id) != special_ids:
        raise ValueError(f"Tokenizer does not match checkpoint {path}")
    model = RuViBitNet10M(cfg).to(device)
    load_checkpoint_weights(model, ckpt)
    model.eval()
    ternary = sum(m.weight.numel() for m in model.modules() if isinstance(m, BitLinear))
    counts = {key: {"nll_sum": 0.0, "tokens": 0, "correct": 0, "exact": 0,
                    "char_f1_sum": 0.0, "generated_tokens": 0, "generation_seconds": 0.0, "pairs": 0}
              for key in ("all", "vi2ru", "ru2vi")}
    loader = DataLoader(TranslationPrefixDataset(samples, cfg), batch_size=batch_size,
                        collate_fn=lambda b: collate_prefix_fn(b, cfg.pad_token_id))
    amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" and (
        not cfg.quantize_hidden_layers or ckpt.get("is_quantized", False)) else nullcontext()
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    sync()
    start = time.perf_counter()
    with torch.inference_mode(), amp:
        offset = 0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)[:, 1:]
            logits = model(ids, mask=batch["mask"].to(device))["logits"][:, :-1].float()
            losses = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
            predictions = logits.argmax(dim=-1)
            for i, sample in enumerate(samples[offset:offset + len(ids)]):
                valid = labels[i] != -100
                for key in ("all", sample["direction"]):
                    c = counts[key]
                    c["nll_sum"] += losses[i][valid].sum().item()
                    c["tokens"] += valid.sum().item()
                    c["correct"] += (predictions[i][valid] == labels[i][valid]).sum().item()
            offset += len(ids)
        sync()
        teacher_seconds = time.perf_counter() - start
        for sample in samples:
            prompt = torch.tensor([[cfg.tag_ru_id if sample["direction"] == "vi2ru" else cfg.tag_vi_id,
                                    *sample["src_ids"], cfg.sep_token_id]], device=device)
            sync()
            start = time.perf_counter()
            generated = model.generate(prompt, max_new_tokens=max_new_tokens)[0, prompt.shape[1]:].tolist()
            sync()
            seconds = time.perf_counter() - start
            if cfg.eos_token_id in generated:
                generated = generated[:generated.index(cfg.eos_token_id)]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
            reference = sample["tgt"].strip()
            for key in ("all", sample["direction"]):
                c = counts[key]
                c["pairs"] += 1
                c["exact"] += prediction == reference
                c["char_f1_sum"] += char_f1(prediction, reference)
                c["generated_tokens"] += len(generated)
                c["generation_seconds"] += seconds
    def summarize(c):
        return {"pairs": c["pairs"], "target_tokens": c["tokens"],
                "nll": c["nll_sum"] / c["tokens"] if c["tokens"] else None,
                "perplexity": math.exp(min(100.0, c["nll_sum"] / c["tokens"])) if c["tokens"] else None,
                "token_accuracy": c["correct"] / c["tokens"] if c["tokens"] else None,
                "exact_match": c["exact"] / c["pairs"] if c["pairs"] else None,
                "char_f1_proxy": c["char_f1_sum"] / c["pairs"] if c["pairs"] else None,
                "generated_tokens": c["generated_tokens"],
                "decode_tokens_per_second": c["generated_tokens"] / c["generation_seconds"] if c["generation_seconds"] else None,
                "decode_ms_per_pair": 1000 * c["generation_seconds"] / c["pairs"] if c["pairs"] else None}
    result = {"checkpoint": str(path), "checkpoint_mb": Path(path).stat().st_size / 1048576,
              "checkpoint_type": "packed ternary" if ckpt.get("pack_2bit") else ("exported ternary" if ckpt.get("is_quantized") else "training/raw"),
              "parameters": sum(p.numel() for p in model.parameters()), "ternary_parameters": ternary,
              "layers": cfg.n_layers, "d_model": cfg.d_model, "d_ff": cfg.d_ff,
              "q_heads": cfg.n_heads, "kv_heads": cfg.n_kv_heads,
              "teacher_forced_seconds": teacher_seconds,
              "metrics": {key: summarize(value) for key, value in counts.items()}}
    del model, ckpt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", "--ternary", dest="a", required=True, help="First checkpoint")
    parser.add_argument("--b", "--bf16", dest="b", required=True, help="Second checkpoint")
    parser.add_argument("--data", required=True, help="Held-out JSONL with src, tgt, dir")
    parser.add_argument("--tokenizer", default="ru-vi-tatoeba-demo")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output", help="Optional JSON summary path")
    args = parser.parse_args()
    if args.samples <= 0 or args.skip < 0 or args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("samples, batch_size and max_new_tokens must be positive; skip nonnegative")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    with open(args.data, encoding="utf-8") as f:
        valid_lines = (line.strip() for line in f if line.strip())
        items = [json.loads(line) for line in islice(valid_lines, args.skip, args.skip + args.samples)]
    if not items:
        parser.error("empty evaluation dataset")
    for item in items:
        if item["dir"] not in ("vi2ru", "ru2vi"):
            parser.error("dir must be vi2ru or ru2vi")
    samples = [{"src_ids": tokenizer.encode(x["src"], add_special_tokens=False),
                "tgt_ids": tokenizer.encode(x["tgt"], add_special_tokens=False),
                "tgt": x["tgt"], "direction": x["dir"]} for x in items]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    results = [evaluate(path, samples, tokenizer, device, args.batch_size, args.max_new_tokens)
               for path in (args.a, args.b)]
    summary = {"data": args.data, "samples": len(samples), "device": str(device), "models": results}
    if args.output:
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("checkpoint_mb", "parameters", "ternary_parameters", "layers", "d_model", "d_ff",
                "q_heads", "kv_heads", "teacher_forced_seconds"):
        print(f"{key:26} {str(results[0][key]):>18} {str(results[1][key]):>18}")
    for direction in ("all", "vi2ru", "ru2vi"):
        print(f"\n{direction}: {results[0]['metrics'][direction]['pairs']} pairs")
        for key in ("nll", "perplexity", "token_accuracy", "exact_match", "char_f1_proxy",
                    "decode_tokens_per_second", "decode_ms_per_pair"):
            a, b = (r["metrics"][direction][key] for r in results)
            print(f"{key:26} {a if a is None else f'{a:.4f}':>18} {b if b is None else f'{b:.4f}':>18}")
    print("\nA:", args.a, "\nB:", args.b)
    print("char_f1_proxy is character unigram overlap, not chrF/BLEU/COMET; run on held-out data.")
