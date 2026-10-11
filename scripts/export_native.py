"""Export trusted RuVi checkpoints to a mmap-friendly, versioned native image."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from ruvi_bitnet.model import BitLinear, Int8Linear, RuViBitNet10M
from ruvi_bitnet.checkpoint import checkpoint_config, load_checkpoint_weights, pack_ternary_to_2bit

MAGIC = b'RUVI0001'
HEADER_SIZE = 80


def load_native_reference(checkpoint, float_storage='bf16'):
    if float_storage not in ('int8', 'bf16', 'fp32'):
        raise ValueError('float_storage must be int8, bf16 or fp32')
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)  # Trusted local files only.
    cfg = checkpoint_config(ckpt)
    model = RuViBitNet10M(cfg).eval()
    load_checkpoint_weights(model, ckpt)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, BitLinear):
                if module.export_scale is None:
                    beta = module.weight.abs().mean().clamp(min=1e-5)
                    module.weight.copy_((module.weight / beta).round().clamp(-1, 1))
                    module.export_scale = beta
            elif isinstance(module, Int8Linear):
                if module.export_scale is None:
                    scale = module.weight.abs().amax().clamp(min=1e-5) / 127.0
                    module.weight.copy_(torch.clamp(torch.round(module.weight / scale), -128, 127))
                    module.export_scale = scale
            elif float_storage == 'int8':
                for parameter in module.parameters(recurse=False):
                    if parameter.ndim == 2:
                        scale = parameter.abs().amax().clamp(min=1e-5) / 127.0
                        q = torch.clamp(torch.round(parameter / scale), -128, 127) * scale
                        parameter.copy_(q)
                    else:
                        parameter.copy_(parameter.to(torch.bfloat16).float())
            elif float_storage == 'bf16':
                for parameter in module.parameters(recurse=False):
                    parameter.copy_(parameter.to(torch.bfloat16).float())
    return model


def export_native(checkpoint, output, float_storage='bf16'):
    model = load_native_reference(checkpoint, float_storage)
    cfg = model.config
    tensors = [('embed_tokens.weight', model.embed_tokens.weight, None)]
    for i, layer in enumerate(model.layers):
        for name in ('attn_norm', 'attn.q_norm', 'attn.k_norm', 'attn.q_proj',
                     'attn.k_proj', 'attn.v_proj', 'attn.o_proj', 'ffn_norm', 'ffn.fc1', 'ffn.fc2'):
            module = layer.get_submodule(name)
            if getattr(module, 'bias', None) is not None:
                raise ValueError('Native v1 does not support biased projections')
            scale = None
            weight = module.weight.detach().float()
            if isinstance(module, BitLinear):
                if module.export_scale is None:
                    scale = weight.abs().mean().clamp(min=1e-5)
                    weight = (weight / scale).round().clamp(-1, 1)
                else:
                    scale = module.export_scale.float()
            elif isinstance(module, Int8Linear):
                if module.export_scale is None:
                    scale = weight.abs().amax().clamp(min=1e-5) / 127.0
                    weight = torch.clamp(torch.round(weight / scale), -128, 127)
                else:
                    scale = module.export_scale.float()
            tensors.append((f'layers.{i}.{name}.weight', weight, scale))
    tensors.append(('norm.weight', model.norm.weight, None))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, prefix=output.name + '.', suffix='.tmp')
    manifest = []
    try:
        with os.fdopen(fd, 'w+b') as stream:
            stream.write(bytes(HEADER_SIZE))
            for name, weight, scale in tensors:
                weight = weight.detach().cpu().contiguous()
                if not torch.isfinite(weight).all():
                    raise ValueError(f'Nonfinite weight: {name}')
                rows, cols = weight.shape if weight.ndim == 2 else (1, weight.numel())
                if scale is not None:
                    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() <= 0:
                        raise ValueError(f'Invalid scale: {name}')
                    if ((weight == -1) | (weight == 0) | (weight == 1)).all():
                        kind = 2
                        payload = pack_ternary_to_2bit(weight.to(torch.int8)).numpy().tobytes()
                    else:
                        kind = 3
                        payload = weight.to(torch.int8).numpy().tobytes()
                    beta = scale.item()
                elif float_storage == 'int8' and weight.ndim == 2:
                    kind = 3
                    scale_val = weight.abs().amax().clamp(min=1e-5).item() / 127.0
                    weight_q = torch.clamp(torch.round(weight / scale_val), -128, 127).to(torch.int8)
                    payload = weight_q.numpy().tobytes()
                    beta = scale_val
                elif float_storage == 'bf16' or (float_storage == 'int8' and weight.ndim != 2):
                    kind, beta = 1, 1.0
                    payload = weight.to(torch.bfloat16).view(torch.int16).numpy().astype('<u2').tobytes()
                else:
                    kind, beta = 0, 1.0
                    payload = weight.float().numpy().astype('<f4').tobytes()
                record = struct.pack('<IIIfI', rows, cols, kind, beta, len(payload)) + bytes(12)
                record += payload + bytes((-len(payload)) % 16)
                stream.write(record)
                manifest.append(dict(name=name, shape=[rows, cols], kind=kind, payload_bytes=len(payload)))
            size = stream.tell()
            attn_softcap = float(cfg.attn_logit_softcapping) if getattr(cfg, 'attn_logit_softcapping', None) is not None else 0.0
            final_softcap = float(cfg.final_logit_softcapping) if getattr(cfg, 'final_logit_softcapping', None) is not None else 0.0
            header = struct.pack('<8s12I2fII2f', MAGIC, cfg.vocab_size, cfg.d_model, cfg.n_layers,
                                 cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_ff, cfg.max_seq_len,
                                 cfg.eos_token_id, cfg.pad_token_id, cfg.tag_ru_id, cfg.tag_vi_id,
                                 cfg.rope_theta, cfg.eps, size, 0, attn_softcap, final_softcap)
            stream.seek(0)
            stream.write(header + bytes(HEADER_SIZE - len(header)))
            stream.seek(0)
            crc = 0
            while block := stream.read(1 << 20):
                crc = zlib.crc32(block, crc)
            stream.seek(68)
            stream.write(struct.pack('<I', crc))
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    report = dict(format='RUVI0001', config=asdict(cfg), source_checkpoint=str(checkpoint),
                  float_storage=float_storage, image_bytes=size, crc32=crc, sha256=digest, tensors=manifest)
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--float-storage', choices=['int8', 'bf16', 'fp32'], default='int8')
    args = parser.parse_args()
    report = export_native(args.checkpoint, args.output, args.float_storage)
    print(json.dumps({key: report[key] for key in ('format', 'image_bytes', 'sha256')}))
