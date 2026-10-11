"""Shared checkpoint weight restoration and ternary packing.

Torch checkpoints may contain pickled configs; only load trusted files.
"""
from dataclasses import fields
import math

import torch

from .model import BitLinear, Int8Linear, RuViBitNet10M, RuViConfig


def checkpoint_config(checkpoint: dict) -> RuViConfig:
    config = checkpoint["config"]
    if isinstance(config, dict):
        valid_keys = {f.name for f in fields(RuViConfig)}
        filtered = {k: v for k, v in config.items() if k in valid_keys}
        return RuViConfig(**filtered)
    return config


def require_shadow_weights(checkpoint: dict) -> None:
    if checkpoint.get("is_quantized") or any(
        name.endswith(".export_scale") for name in checkpoint["model_state_dict"]
    ):
        raise ValueError("Export requires training shadow weights, not an inference state")


def pack_ternary_to_2bit(w_int8: torch.Tensor) -> torch.Tensor:
    """
    Nén 4 trọng số tam phân {-1, 0, 1} thành 1 byte uint8 (2-bit mỗi trọng số).
    Map: -1 -> 0, 0 -> 1, 1 -> 2
    """
    flat = (w_int8.flatten() + 1).to(torch.uint8)  # Chuyển về {0, 1, 2}
    pad_len = (4 - (len(flat) % 4)) % 4
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8, device=flat.device)])

    reshaped = flat.view(-1, 4)
    packed = (reshaped[:, 0]) | (reshaped[:, 1] << 2) | (reshaped[:, 2] << 4) | (reshaped[:, 3] << 6)
    return packed


def load_checkpoint_weights(model: RuViBitNet10M, ckpt: dict):
    state_dict = ckpt["model_state_dict"]
    if not ckpt.get("is_quantized", False):
        model.load_state_dict(state_dict)
        return

    # Retain integer ternary values and their scale for BitLinear; never requantize them.
    modules = dict(model.named_modules())
    restored = {}
    for k, v in state_dict.items():
        if k.endswith("_scale") or k.endswith("_shape"):
            continue

        if k.endswith("_packed"):
            base_k = k[:-7]
            shape = state_dict[base_k + "_shape"]
            module_name = base_k.rsplit(".", 1)[0]
            if not isinstance(modules.get(module_name), (BitLinear, Int8Linear)):
                raise ValueError(f"Quantized weight does not belong to BitLinear or Int8Linear: {base_k}")
            restored[module_name + ".export_scale"] = state_dict[base_k + "_scale"].float()

            num_el = math.prod(shape)
            w0 = (v & 0b00000011).to(torch.int8) - 1
            w1 = ((v >> 2) & 0b00000011).to(torch.int8) - 1
            w2 = ((v >> 4) & 0b00000011).to(torch.int8) - 1
            w3 = ((v >> 6) & 0b00000011).to(torch.int8) - 1
            unpacked = torch.stack([w0, w1, w2, w3], dim=1).flatten()[:num_el]
            restored[base_k] = unpacked.view(shape).float()
        elif k + "_scale" in state_dict:
            module_name = k.rsplit(".", 1)[0]
            if not isinstance(modules.get(module_name), (BitLinear, Int8Linear)):
                raise ValueError(f"Quantized weight does not belong to BitLinear or Int8Linear: {k}")
            restored[module_name + ".export_scale"] = state_dict[k + "_scale"].float()
            restored[k] = v.float()
        else:
            restored[k] = v.float()

    model.load_state_dict(restored)
