# scripts/train.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import json
import os
import random
import math

import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from ruvi_bitnet import RuViBitNet10M, RuViConfig
from ruvi_bitnet.model import BitLinear
from ruvi_bitnet.distill import TranslationPrefixDataset, collate_prefix_fn


def report_bitlinear(model):
    print("[*] BitLinear Layer Statistics:")
    for i, layer in enumerate(model.layers):
        modules = [m for m in layer.modules() if isinstance(m, BitLinear)]
        if not modules:
            continue
        with torch.no_grad():
            ternary = torch.cat([(m.weight / m.weight.abs().mean().clamp(min=1e-5)).round().clamp(-1, 1).flatten() for m in modules])
            fractions = [(ternary == v).float().mean().item() for v in (-1, 0, 1)]
            betas = [m.weight.abs().mean().item() for m in modules]
            avg_beta = sum(betas) / len(betas)
            grads = [m.weight.grad.norm().item() for m in modules if m.weight.grad is not None]
        grad_str = f", grad_norm={sum(grads) / len(grads):.3g}" if grads else ""
        print(f"  layer {i:2d}: -1/0/+1 = {fractions[0]:.2f}/{fractions[1]:.2f}/{fractions[2]:.2f}, avg_scale={avg_beta:.4f}{grad_str}")


def save_checkpoint(path, state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def restore_scheduler(scheduler, optimizer, state, total_steps, warmup_ratio):
    step = state["step"]
    if not 0 <= step <= total_steps:
        raise ValueError("Saved step is outside the requested training schedule")
    if (state.get("total_steps") == total_steps and "scheduler_state_dict" in state
            and state.get("warmup_ratio", warmup_ratio) == warmup_ratio):
        scheduler.load_state_dict(state["scheduler_state_dict"])
        rates = scheduler.get_last_lr()
    else:
        scheduler.last_epoch = step
        rates = [base * factor(step) for base, factor in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
        scheduler._last_lr = rates
        scheduler._step_count = step + 1
    for group, rate in zip(optimizer.param_groups, rates):
        group["lr"] = rate


def load_dataset(jsonl_path: str, tokenizer: PreTrainedTokenizerFast):
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            src_ids = tokenizer.encode(item["src"], add_special_tokens=False)
            tgt_ids = tokenizer.encode(item["tgt"], add_special_tokens=False)
            samples.append({
                "src_ids": src_ids,
                "tgt_ids": tgt_ids,
                "direction": item.get("dir", "vi2ru"),
            })
    return samples


def train(
    data_path: str = "distill_vi_ru_tatoeba.jsonl",
    tokenizer_dir: str = "ru_vi_tokenizer_8k",
    output_model: str = "ruvi_10m_checkpoint.pt",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.1,
    dropout: float = 0.1,
    label_smoothing: float = 0.1,
    weight_decay: float = 0.1,
    seed: int = 42,
    warmup_ratio: float = 0.05,
    resume: str = None,
    report_stats: bool = True,
    bf16: bool = False,
    variant: str = "base",
    max_seq_len: int = 512,
):
    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError("epochs, batch_size and lr must be positive")
    if not 0 <= val_split < 1:
        raise ValueError("val_split must be in [0, 1)")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 training requires a BF16-capable CUDA GPU")
    print(f"[*] Training on device: {device} | Seed: {seed}")

    # 1. Load Tokenizer & Config
    if not os.path.exists(tokenizer_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục tokenizer: '{tokenizer_dir}'. Hãy kiểm tra lại đường dẫn.")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    variants = {
        "base": {},
        "wide": dict(d_model=768, d_ff=2560, n_heads=12, n_kv_heads=4),
        "deep": dict(n_layers=22),
    }
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}")
    sep_id = tokenizer.convert_tokens_to_ids("<sep>")
    tag_ru_id = tokenizer.convert_tokens_to_ids("<2ru>")
    tag_vi_id = tokenizer.convert_tokens_to_ids("<2vi>")
    if any(tok_id is None or tok_id == tokenizer.unk_token_id for tok_id in (sep_id, tag_ru_id, tag_vi_id)):
        raise ValueError("Tokenizer must contain valid <sep>, <2ru>, and <2vi> special tokens")

    cfg = RuViConfig(
        **variants[variant],
        vocab_size=len(tokenizer),
        max_seq_len=max_seq_len,
        dropout=dropout,
        label_smoothing=label_smoothing,
        quantize_hidden_layers=not bf16,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=sep_id,
        tag_ru_id=tag_ru_id,
        tag_vi_id=tag_vi_id,
    )

    # 2. Init 10M BitNet Model & Optimizer
    model = RuViBitNet10M(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)

    # 3. Train/Val Split & DataLoader
    samples = load_dataset(data_path, tokenizer)
    random.Random(seed).shuffle(samples)
    val_size = int(len(samples) * val_split)
    train_samples = samples[val_size:]
    val_samples = samples[:val_size]
    print(f"[*] Dataset: {len(train_samples)} train samples | {len(val_samples)} val samples")

    train_loader = DataLoader(
        TranslationPrefixDataset(train_samples, cfg),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_prefix_fn(b, pad_token_id=cfg.pad_token_id),
    )
    val_loader = DataLoader(
        TranslationPrefixDataset(val_samples, cfg),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_prefix_fn(b, pad_token_id=cfg.pad_token_id),
    ) if val_samples else None

    # 4. Training Loop
    if len(train_loader) == 0:
        raise ValueError("Training set is empty")
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start_epoch = 1
    best_val_loss = float("inf")
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)  # Only resume trusted local checkpoints.
        if "optimizer_state_dict" not in state:
            raise ValueError("Resume requires a training checkpoint, not an inference-only checkpoint")
        saved_cfg = state["config"]
        if isinstance(saved_cfg, dict):
            saved_cfg = RuViConfig(**saved_cfg)
        for attr in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim", "d_ff", "quantize_first_layer", "quantize_last_layer", "quantize_hidden_layers", "max_seq_len", "rope_theta", "eps", "dropout", "label_smoothing"):
            if getattr(saved_cfg, attr) != getattr(cfg, attr):
                raise ValueError(f"Resume config mismatch for {attr}: saved {getattr(saved_cfg, attr)} vs current {getattr(cfg, attr)}")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        state.setdefault("step", state.get("epoch", 0) * len(train_loader))
        restore_scheduler(scheduler, optimizer, state, total_steps, warmup_ratio)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            saved_rng = state["cuda_rng"]
            if len(saved_rng) != torch.cuda.device_count():
                raise ValueError("Exact resume requires the same CUDA device count")
            torch.cuda.set_rng_state_all([rng.cpu() for rng in saved_rng])
        start_epoch = state["epoch"] + 1
        # Legacy checkpoints used a batch-averaged validation loss.
        best_val_loss = state.get("best_val_loss", float("inf")) if "validation_nll" in state else float("inf")
    save_file = os.path.join(output_model, "ruvi_10m_checkpoint.pt") if os.path.isdir(output_model) else output_model
    last_file = save_file.replace(".pt", "_last.pt") if save_file.endswith(".pt") else save_file + "_last.pt"
    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)

    if report_stats and start_epoch == 1:
        print("[*] Initial BitLinear Statistics:")
        report_bitlinear(model)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        train_tokens = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                out = model(input_ids=input_ids, mask=mask, labels=labels)
                loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            tokens = out["target_tokens"].item()
            train_loss += loss.item() * tokens
            train_tokens += tokens
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / train_tokens

        # Validation NLL is unsmoothed and averaged over target tokens, not batches.
        improved = False
        if val_loader:
            model.eval()
            with torch.no_grad():
                val_loss_acc = 0.0
                val_tokens = 0
                for batch in val_loader:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                        out = model(
                            input_ids=batch["input_ids"].to(device),
                            mask=batch["mask"].to(device),
                            labels=batch["labels"].to(device),
                        )
                    tokens = out["target_tokens"].item()
                    val_loss_acc += out["loss"].item() * tokens
                    val_tokens += tokens
                avg_val_loss = val_loss_acc / val_tokens
            print(f"Epoch {epoch} - Train CE (smoothing={cfg.label_smoothing}): {avg_train_loss:.4f} | Val NLL: {avg_val_loss:.4f}")
            improved = avg_val_loss < best_val_loss
            if improved:
                best_val_loss = avg_val_loss
        else:
            print(f"Epoch {epoch} - Train CE (smoothing={cfg.label_smoothing}): {avg_train_loss:.4f}")

        # Capture RNG after validation: its DataLoader iterator consumes CPU RNG.
        ckpt_state = {
            "config": cfg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "step": epoch * len(train_loader),
            "best_val_loss": best_val_loss,
            "total_steps": total_steps,
            "warmup_ratio": warmup_ratio,
            "validation_nll": avg_val_loss if val_loader else None,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        if improved or not val_loader:
            save_checkpoint(save_file, ckpt_state)
            print(f"[+] Saved best model to: {save_file}")
        save_checkpoint(last_file, ckpt_state)

        if report_stats:
            report_bitlinear(model)

    print(f"[+] Training complete. Best checkpoint: {save_file} (best_val_loss: {best_val_loss:.4f}) | Latest: {last_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="distill_vi_ru_tatoeba.jsonl")
    parser.add_argument("--tokenizer", default="ru_vi_tokenizer_8k")
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int, default=512, help="Hard limit for forward and prompt plus generated tokens")
    parser.add_argument("--no_report_bitlinear", action="store_true", help="Tắt hiển thị thống kê BitLinear theo tầng")
    parser.add_argument("--bf16", action="store_true", help="Huấn luyện baseline nn.Linear với CUDA BF16 autocast (FP32 master weights)")
    parser.add_argument("--variant", choices=("base", "wide", "deep"), default="base",
                        help="base: 384 × 5 (~10M); wide: 768 × 5; deep: 384 × 22 (~33-34M)")
    args = parser.parse_args()

    train(
        args.data,
        args.tokenizer,
        args.output or f"ruvi_{args.variant}{'_bf16' if args.bf16 else ''}_checkpoint.pt",
        args.epochs,
        args.batch_size,
        args.lr,
        args.val_split,
        args.dropout,
        args.label_smoothing,
        args.weight_decay,
        seed=args.seed,
        warmup_ratio=args.warmup_ratio,
        resume=args.resume,
        report_stats=not args.no_report_bitlinear and not args.bf16,
        bf16=args.bf16,
        variant=args.variant,
        max_seq_len=args.max_seq_len,
    )
