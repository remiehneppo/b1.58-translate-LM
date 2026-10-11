import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RuViConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 11
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff: int = 1024
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    eps: float = 1e-6
    dropout: float = 0.1
    label_smoothing: float = 0.1
    init_std: float = 0.02
    quantize_first_layer: bool = False   # Sandwich Precision (ADR-0003: Layer 0 unquantized)
    quantize_last_layer: bool = False    # Sandwich Precision (ADR-0004: Layer L-1 unquantized)
    quantize_hidden_layers: bool = True  # Hidden layers use quantized linear
    weight_quant: str = "ternary"        # "ternary" (BitLinear 1.58-bit) hoặc "int8" (Int8Linear 8-bit)
    attn_logit_softcapping: Optional[float] = 50.0  # Gemma 2 Attention soft-capping
    final_logit_softcapping: Optional[float] = 30.0 # Gemma 2 Final logit soft-capping
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    sep_token_id: int = 3
    tag_ru_id: int = 4
    tag_vi_id: int = 5


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class PerHeadRMSNorm(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads * head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, S, num_heads, head_dim]
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        w = self.weight.view(1, 1, self.num_heads, self.head_dim)
        return x * torch.rsqrt(var + self.eps) * w


class BitLinear(nn.Linear):
    """
    BitNet b1.58 Linear Layer with Straight-Through Estimator (STE).
    Weights in {-1, 0, 1}, activations quantized to 8-bit.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("export_scale", None)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        scale = state_dict.get(prefix + "export_scale")
        self.export_scale = None if scale is None else torch.empty_like(scale, device=self.weight.device)
        super()._load_from_state_dict(state_dict, prefix, *args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Activation quantization (8-bit signed)
        scale_x = 127.0 / torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1e-5)
        x_q = torch.clamp(torch.round(x * scale_x), -128, 127) / scale_x
        x_ste = x + (x_q - x).detach()

        # Exported weights are exact {-1, 0, 1}; apply their saved scale once.
        if self.export_scale is not None:
            w_ste = self.weight * self.export_scale
        else:
            beta = self.weight.abs().mean().clamp(min=1e-5)
            w_scaled = self.weight / beta
            w_q = torch.clamp(torch.round(w_scaled), -1.0, 1.0) * beta
            w_ste = self.weight + (w_q - self.weight).detach()

        return F.linear(x_ste, w_ste, self.bias)


class Int8Linear(nn.Linear):
    """
    8-bit Weight & 8-bit Activation Linear Layer with Straight-Through Estimator (STE).
    Weights in [-128, 127], activations in [-128, 127].
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("export_scale", None)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        scale = state_dict.get(prefix + "export_scale")
        self.export_scale = None if scale is None else torch.empty_like(scale, device=self.weight.device)
        super()._load_from_state_dict(state_dict, prefix, *args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Activation quantization (8-bit signed)
        scale_x = 127.0 / torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1e-5)
        x_q = torch.clamp(torch.round(x * scale_x), -128, 127) / scale_x
        x_ste = x + (x_q - x).detach()

        # 2. Weight quantization (8-bit signed)
        if self.export_scale is not None:
            w_ste = self.weight * self.export_scale
        else:
            scale_w = torch.clamp(self.weight.abs().amax(), min=1e-5) / 127.0
            w_scaled = self.weight / scale_w
            w_q = torch.clamp(torch.round(w_scaled), -128.0, 127.0) * scale_w
            w_ste = self.weight + (w_q - self.weight).detach()

        return F.linear(x_ste, w_ste, self.bias)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def apply_rope(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        if offset < 0 or offset + seq_len > self.cos_cached.shape[0]:
            raise ValueError("RoPE positions exceed max_seq_len")
        cos = self.cos_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim/2]
        sin = self.sin_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(2)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2)


class BitLinearGQA(nn.Module):
    def __init__(self, config: RuViConfig, is_quantized: bool = True):
        super().__init__()
        if not is_quantized:
            linear_cls = nn.Linear
        elif config.weight_quant == "int8":
            linear_cls = Int8Linear
        else:
            linear_cls = BitLinear
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_rep = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout
        self.attn_logit_softcapping = config.attn_logit_softcapping

        self.q_proj = linear_cls(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = linear_cls(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = linear_cls(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = linear_cls(config.n_heads * config.head_dim, config.d_model, bias=False)

        self.q_norm = PerHeadRMSNorm(config.n_heads, config.head_dim, eps=config.eps)
        self.k_norm = PerHeadRMSNorm(config.n_kv_heads, config.head_dim, eps=config.eps)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, _ = x.shape

        q = self.q_norm(self.q_proj(x).view(B, S, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        q = rope.apply_rope(q, S, offset=offset)
        k = rope.apply_rope(k, S, offset=offset)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=1)
            v = torch.cat([v_prev, v], dim=1)
        new_kv_cache = (k, v)

        # GQA expansion
        if self.num_rep > 1:
            k_exp = k.repeat_interleave(self.num_rep, dim=2)
            v_exp = v.repeat_interleave(self.num_rep, dim=2)
        else:
            k_exp, v_exp = k, v

        # [B, H, S, D]
        q = q.transpose(1, 2)
        k_exp = k_exp.transpose(1, 2)
        v_exp = v_exp.transpose(1, 2)

        if self.attn_logit_softcapping is not None and self.attn_logit_softcapping > 0:
            scores = torch.matmul(q, k_exp.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = self.attn_logit_softcapping * torch.tanh(scores / self.attn_logit_softcapping)
            if mask is not None:
                if mask.dtype == torch.bool:
                    all_masked = (~mask).all(dim=-1, keepdim=True)
                    scores = scores.masked_fill(~mask, float("-inf"))
                    scores = scores.masked_fill(all_masked, 0.0)
                else:
                    scores = scores + mask
            attn_weights = F.softmax(scores, dim=-1)
            if mask is not None and mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(~mask, 0.0)
            if self.training and self.dropout > 0.0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            out = torch.matmul(attn_weights, v_exp)
        else:
            out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
            )
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), new_kv_cache


class SquaredReLUFFN(nn.Module):
    def __init__(self, config: RuViConfig, is_quantized: bool = True):
        super().__init__()
        if not is_quantized:
            linear_cls = nn.Linear
        elif config.weight_quant == "int8":
            linear_cls = Int8Linear
        else:
            linear_cls = BitLinear
        self.fc1 = linear_cls(config.d_model, config.d_ff, bias=False)
        self.fc2 = linear_cls(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.relu(self.fc1(x)).square()))


class TransformerBlock(nn.Module):
    def __init__(self, config: RuViConfig, is_quantized: bool = True):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.eps)
        self.attn = BitLinearGQA(config, is_quantized=is_quantized)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.eps)
        self.ffn = SquaredReLUFFN(config, is_quantized=is_quantized)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, new_kv = self.attn(self.attn_norm(x), rope, mask=mask, kv_cache=kv_cache, offset=offset)
        x = x + self.attn_dropout(h)
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


class RuViBitNet10M(nn.Module):
    def __init__(self, config: Optional[RuViConfig] = None):
        super().__init__()
        self.config = config or RuViConfig()
        if (self.config.n_heads <= 0 or self.config.n_kv_heads <= 0
                or self.config.n_heads % self.config.n_kv_heads
                or self.config.head_dim <= 0 or self.config.head_dim % 2
                or self.config.max_seq_len <= 0):
            raise ValueError("GQA needs positive divisible head counts, even head_dim and positive max_seq_len")
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.d_model)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.config.init_std)
        self.rope = RotaryEmbedding(self.config.head_dim, self.config.max_seq_len, self.config.rope_theta)
        def is_layer_quantized(i: int) -> bool:
            if not self.config.quantize_hidden_layers:
                return False
            if i == 0 and not self.config.quantize_first_layer:
                return False
            if i == self.config.n_layers - 1 and not self.config.quantize_last_layer:
                return False
            return True

        self.layers = nn.ModuleList([
            TransformerBlock(self.config, is_quantized=is_layer_quantized(i))
            for i in range(self.config.n_layers)
        ])
        self.norm = RMSNorm(self.config.d_model, eps=self.config.eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    @staticmethod
    def build_prefix_lm_mask(seq_len: int, prefix_len: int, device: torch.device) -> torch.Tensor:
        """
        Build Prefix-LM attention mask (Boolean):
        Prefix tokens (0..prefix_len-1): bidirectional attention.
        Target tokens (prefix_len..seq_len-1): causal autoregressive attention.
        """
        if not 0 <= prefix_len <= seq_len:
            raise ValueError("prefix_len must be in [0, seq_len]")
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        # 1. Prefix tokens can attend to all prefix tokens
        mask[:prefix_len, :prefix_len] = True
        # 2. Target tokens can attend to all prefix tokens
        mask[prefix_len:, :prefix_len] = True
        # 3. Target tokens attend to preceding target tokens (causal)
        target_len = seq_len - prefix_len
        if target_len > 0:
            causal = torch.tril(torch.ones((target_len, target_len), dtype=torch.bool, device=device))
            mask[prefix_len:, prefix_len:] = causal
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        prefix_len: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        alpha_distill: float = 0.5,
        temperature: float = 2.0,
    ) -> dict:
        B, S = input_ids.shape
        if B == 0 or not 0 < S <= self.config.max_seq_len:
            raise ValueError("input_ids must be nonempty and within max_seq_len")
        x = self.embed_tokens(input_ids)

        if mask is None:
            if prefix_len is not None and prefix_len > 0:
                mask = self.build_prefix_lm_mask(S, prefix_len, x.device).unsqueeze(0).unsqueeze(1)
            else:
                mask = torch.tril(torch.ones((S, S), dtype=torch.bool, device=x.device)).unsqueeze(0).unsqueeze(1)

        for layer in self.layers:
            x, _ = layer(x, self.rope, mask=mask)

        x = self.norm(x)
        # Tied embeddings projection
        logits = F.linear(x, self.embed_tokens.weight)
        if self.config.final_logit_softcapping is not None and self.config.final_logit_softcapping > 0:
            logits = self.config.final_logit_softcapping * torch.tanh(logits / self.config.final_logit_softcapping)

        loss = None
        target_tokens = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid = shift_labels != -100
            target_tokens = valid.sum()
            if not valid.any():
                raise ValueError("labels must contain at least one target token")
            ce_loss = F.cross_entropy(
                shift_logits[valid].float(), shift_labels[valid],
                label_smoothing=self.config.label_smoothing if self.training else 0.0,
            )
            loss = ce_loss

            if teacher_logits is not None:
                if teacher_logits.shape != logits.shape or temperature <= 0 or not 0 <= alpha_distill <= 1:
                    raise ValueError("KD needs matching logits, positive temperature and alpha_distill in [0, 1]")
                shift_teacher = teacher_logits[..., :-1, :][valid].float()
                kd_loss = F.kl_div(
                    F.log_softmax(shift_logits[valid].float() / temperature, dim=-1),
                    F.softmax(shift_teacher / temperature, dim=-1),
                    reduction="batchmean",
                ) * temperature ** 2
                loss = (1.0 - alpha_distill) * ce_loss + alpha_distill * kd_loss

        return {"logits": logits, "loss": loss, "target_tokens": target_tokens}

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        subvocab_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            return prompt_ids
        eos_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        B, P = prompt_ids.shape
        if B == 0 or P == 0 or P + max_new_tokens > self.config.max_seq_len:
            raise ValueError("prompt plus max_new_tokens must fit max_seq_len")
        x = self.embed_tokens(prompt_ids)
        prefix_mask = self.build_prefix_lm_mask(P, P, x.device).unsqueeze(0).unsqueeze(1)

        kv_caches = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, kv_caches[i] = layer(x, self.rope, mask=prefix_mask, offset=0)

        x = self.norm(x)
        logits = F.linear(x[:, -1:], self.embed_tokens.weight)
        if self.config.final_logit_softcapping is not None and self.config.final_logit_softcapping > 0:
            logits = self.config.final_logit_softcapping * torch.tanh(logits / self.config.final_logit_softcapping)
        if subvocab_mask is not None:
            logits = logits.masked_fill(~subvocab_mask, -1e9)
        next_token = torch.argmax(logits, dim=-1)

        finished = next_token.squeeze(-1) == eos_id
        generated = [next_token]
        curr_offset = P

        for step in range(max_new_tokens - 1):
            if finished.all():
                break
            tok_emb = self.embed_tokens(next_token)
            for i, layer in enumerate(self.layers):
                tok_emb, kv_caches[i] = layer(
                    tok_emb, self.rope, mask=None, kv_cache=kv_caches[i], offset=curr_offset
                )
            tok_emb = self.norm(tok_emb)
            logits = F.linear(tok_emb, self.embed_tokens.weight)
            if self.config.final_logit_softcapping is not None and self.config.final_logit_softcapping > 0:
                logits = self.config.final_logit_softcapping * torch.tanh(logits / self.config.final_logit_softcapping)
            if subvocab_mask is not None:
                logits = logits.masked_fill(~subvocab_mask, -1e9)

            if repetition_penalty != 1.0 or no_repeat_ngram_size > 0:
                logits = logits.clone()
                curr_gen = torch.cat(generated, dim=1)
                for b in range(B):
                    if finished[b]:
                        continue
                    gen_b = curr_gen[b].tolist()
                    if repetition_penalty != 1.0:
                        for tid in set(gen_b):
                            if logits[b, 0, tid] > 0:
                                logits[b, 0, tid] /= repetition_penalty
                            else:
                                logits[b, 0, tid] *= repetition_penalty
                    if no_repeat_ngram_size > 0 and len(gen_b) >= no_repeat_ngram_size - 1:
                        k = no_repeat_ngram_size - 1
                        suffix = tuple(gen_b[-k:])
                        for idx in range(len(gen_b) - k):
                            if tuple(gen_b[idx : idx + k]) == suffix:
                                logits[b, 0, gen_b[idx + k]] = -1e9

            next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.masked_fill(finished.unsqueeze(-1), self.config.pad_token_id)
            finished |= next_token.squeeze(-1) == eos_id
            generated.append(next_token)
            curr_offset += 1

        return torch.cat([prompt_ids, torch.cat(generated, dim=1)], dim=1)


class RuViInt8_10M(RuViBitNet10M):
    """
    RuVi-10M architecture with full INT8 weights and activations across all layers.
    """
    def __init__(self, config: Optional[RuViConfig] = None):
        if config is None:
            config = RuViConfig(
                weight_quant="int8",
                quantize_first_layer=True,
                quantize_last_layer=True,
                quantize_hidden_layers=True,
            )
        else:
            config.weight_quant = "int8"
            config.quantize_first_layer = getattr(config, "quantize_first_layer", True)
            config.quantize_last_layer = getattr(config, "quantize_last_layer", True)
            config.quantize_hidden_layers = True
        super().__init__(config)


if __name__ == "__main__":
    cfg = RuViConfig()
    model = RuViBitNet10M(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"RuVi-BitNet-10M initialized.")
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Test Prefix-LM Forward Pass
    prefix_len = 10
    target_len = 12
    total_len = prefix_len + target_len
    input_ids = torch.randint(0, cfg.vocab_size, (2, total_len))
    labels = input_ids.clone()
    labels[:, :prefix_len] = -100  # Mask out prefix from loss

    out = model(input_ids, prefix_len=prefix_len, labels=labels)
    print(f"Forward output logits shape: {out['logits'].shape}")
    print(f"Prefix-LM Cross-Entropy loss: {out['loss'].item():.4f}")

    # Test Generation with Prefix Prefill & KV Cache
    prompt = torch.tensor([[cfg.tag_ru_id, 100, 200, 300, cfg.sep_token_id]])
    gen_tokens = model.generate(prompt, max_new_tokens=8)
    print(f"Generated sequence shape: {gen_tokens.shape}")

    # Test INT8 Model
    int8_model = RuViInt8_10M()
    int8_layers = [m for m in int8_model.modules() if isinstance(m, Int8Linear)]
    print(f"RuViInt8_10M initialized with {len(int8_layers)} Int8Linear layers.")
    int8_out = int8_model(input_ids, prefix_len=prefix_len, labels=labels)
    print(f"INT8 Forward Cross-Entropy loss: {int8_out['loss'].item():.4f}")
    int8_gen = int8_model.generate(prompt, max_new_tokens=8)
    print(f"INT8 Generated sequence shape: {int8_gen.shape}")

    print("Self-test passed cleanly!")
