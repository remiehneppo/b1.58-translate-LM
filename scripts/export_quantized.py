# scripts/export_quantized.py
import argparse
import os
import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from ruvi_bitnet.checkpoint import checkpoint_config, pack_ternary_to_2bit, require_shadow_weights


def export_model(
    checkpoint_path: str,
    output_path: str = "ruvi_10m_quantized.pt",
    pack_2bit: bool = False,
):
    print(f"[*] Đang tải checkpoint: {checkpoint_path}")
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    require_shadow_weights(ckpt)
    cfg = checkpoint_config(ckpt)
    if not cfg.quantize_hidden_layers:
        raise ValueError("Checkpoint BF16 không có BitLinear để xuất tam phân")

    state_dict = ckpt["model_state_dict"]
    quantized_state = {}

    for name, param in state_dict.items():
        is_first = name.startswith("layers.0.") and not cfg.quantize_first_layer
        is_last = name.startswith(f"layers.{cfg.n_layers - 1}.") and not cfg.quantize_last_layer
        is_linear = any(
            linear_tag in name for linear_tag in [".q_proj.", ".k_proj.", ".v_proj.", ".o_proj.", ".fc1.", ".fc2."]
        )
        is_bitlinear = is_linear and name.endswith(".weight") and param.ndim == 2 and not (is_first or is_last)

        if is_bitlinear:
            if getattr(cfg, "weight_quant", "ternary") == "int8":
                scale = param.abs().amax().clamp(min=1e-5) / 127.0
                w_q = torch.clamp(torch.round(param / scale), -128, 127).to(torch.int8)
                quantized_state[name] = w_q
                quantized_state[name + "_scale"] = scale
            else:
                # Quantize the BitLinear shadow weights; keep scale in FP32.
                beta = param.abs().mean().clamp(min=1e-5)
                w_scaled = param / beta
                w_q = torch.clamp(torch.round(w_scaled), -1.0, 1.0).to(torch.int8)

                if pack_2bit:
                    # Nén 2-bit: 4 weight / 1 byte
                    quantized_state[name + "_packed"] = pack_ternary_to_2bit(w_q)
                    quantized_state[name + "_shape"] = list(param.shape)
                else:
                    quantized_state[name] = w_q.to(torch.bfloat16)

                quantized_state[name + "_scale"] = beta  # Keep full precision to match training forward.

        else:
            # Embedding, sandwich layers and RMSNorm remain BF16.
            quantized_state[name] = param.to(torch.bfloat16)

    payload = {
        "config": cfg,
        "is_quantized": True,
        "pack_2bit": pack_2bit,
        "model_state_dict": quantized_state,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    orig_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    quant_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n--- Kết quả Xuất Mô Hình Biên ---")
    print(f"File gốc (Huấn luyện FP32): {orig_mb:.2f} MB")
    print(f"File xuất (Tam phân {'2-bit packed' if pack_2bit else 'BF16'}): {quant_mb:.2f} MB (Giảm {(1 - quant_mb / orig_mb) * 100:.1f}%)")
    print(f"[+] Đã lưu checkpoint triển khai tại: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xuất checkpoint BitNet sang trọng số tam phân 1.58-bit")
    parser.add_argument("--checkpoint", default="ru-vi-tatoeba-demo/ruvi_10m_checkpoint.pt")
    parser.add_argument("--output", default="ru-vi-tatoeba-demo/ruvi_10m_quantized.pt")
    parser.add_argument("--pack_2bit", action="store_true", help="Tuỳ chọn: pack 4 trọng số tam phân vào 1 byte uint8")
    args = parser.parse_args()

    export_model(args.checkpoint, args.output, args.pack_2bit)
