import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from ruvi_bitnet.model import RuViBitNet10M, BitLinear
from ruvi_bitnet.checkpoint import checkpoint_config, load_checkpoint_weights


def ternary_gemm_logic(x_int: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Integer ternary dot products with bounded bitwise scratch."""
    flat = x_int.reshape(-1, weight.shape[1])
    result = torch.zeros((flat.shape[0], weight.shape[0]), dtype=torch.int64, device=x_int.device)
    # ponytail: portable reference kernel, native packed kernel if edge latency matters.
    # Each int32 broadcast is at most 16 * 32 * 256 * 4 = 512 KiB.
    for t in range(0, flat.shape[0], 16):
        for o in range(0, weight.shape[0], 32):
            block = result[t:t + 16, o:o + 32]
            for i in range(0, weight.shape[1], 256):
                x = flat[t:t + 16, None, i:i + 256]
                w = weight[o:o + 32, i:i + 256]
                block.add_((x & -(w == 1).to(torch.int32)).sum(-1))
                block.sub_((x & -(w == -1).to(torch.int32)).sum(-1))
    return result.reshape(*x_int.shape[:-1], weight.shape[0])


class BitLinearLogic(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale_w", torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def load_ternary(self, w_ternary: torch.Tensor, scale: torch.Tensor):
        if w_ternary.shape != self.weight.shape or not ((w_ternary == -1) | (w_ternary == 0) | (w_ternary == 1)).all():
            raise ValueError("Logic weights must have the expected shape and ternary values")
        if scale.numel() != 1 or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Logic weight scale must be finite and positive")
        self.weight.copy_(w_ternary)
        self.scale_w.copy_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_x = 127.0 / torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1e-5)
        x_int = torch.clamp(torch.round(x * scale_x), -128, 127).to(torch.int32)

        # Tính toán lõi: chỉ có phép toán logic và cộng/trừ
        y_int = ternary_gemm_logic(x_int, self.weight)

        # Rescale về FP
        out = (y_int.float() * (self.scale_w.float() / scale_x.float())).to(x.dtype)
        return out if self.bias is None else out + self.bias


def load_model_ternary_logic(checkpoint_path: str, device: str = "cpu") -> RuViBitNet10M:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint_config(ckpt)

    if not ckpt.get("is_quantized"):
        raise ValueError("Logic runtime requires an exported ternary checkpoint")
    model = RuViBitNet10M(cfg).to(device)
    load_checkpoint_weights(model, ckpt)
    for name, module in list(model.named_modules()):
        if isinstance(module, BitLinear):
            parent_name, child_name = name.rsplit(".", 1)
            new_layer = BitLinearLogic(module.in_features, module.out_features, bias=module.bias is not None).to(device)
            new_layer.load_ternary(module.weight, module.export_scale)
            if module.bias is not None:
                with torch.no_grad():
                    new_layer.bias.copy_(module.bias)
            setattr(model.get_submodule(parent_name), child_name, new_layer)
    model.eval()
    return model


if __name__ == "__main__":
    print("[*] Đang khởi tạo và load model với kernel Logic/Cộng/Trừ...")
    model = load_model_ternary_logic("ru-vi-tatoeba-demo/ruvi_10m_quantized.pt")

    dummy_input = torch.randint(0, model.config.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(dummy_input)

    print(f"[+] Chạy inference thành công! Logits shape: {out['logits'].shape}")
