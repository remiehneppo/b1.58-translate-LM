"""Export the FP32 shadow weights of a BitNet checkpoint as an unquantized BF16 model."""
import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from ruvi_bitnet.checkpoint import checkpoint_config, require_shadow_weights


def export_bf16(checkpoint: str, output: str) -> None:
    # Checkpoints are trusted local files: their config may contain Python objects.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require_shadow_weights(ckpt)
    cfg = asdict(checkpoint_config(ckpt))
    cfg.update(quantize_hidden_layers=False, quantize_first_layer=False, quantize_last_layer=False)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "model_state_dict": {
        k: v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in ckpt["model_state_dict"].items()
    }, "precision": "bf16", "source_checkpoint": checkpoint}, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="ru-vi-tatoeba-demo/ruvi_10m_checkpoint.pt")
    parser.add_argument("--output", default="ru-vi-tatoeba-demo/ruvi_10m_bf16.pt")
    args = parser.parse_args()
    export_bf16(args.checkpoint, args.output)
    print(f"BF16 checkpoint: {args.output}")
