import math
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from einops import rearrange


@dataclass
class LocalArgs:
    codebook_size: int = 2048
    num_codebooks: int = 4

# Modified from https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/llama.py#L105
class KVCache(nn.Module):
    def __init__(
        self, n_layer, batch_size, max_seq_len, n_heads, head_dim, dtype, device
    ):
        super().__init__()
        cache_shape = (n_layer, batch_size, n_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, layer_idx, input_pos, k_val, v_val):
        # k_val: [B, H, S, D]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[layer_idx, :, :, input_pos:input_pos+1] = k_val
        v_out[layer_idx, :, :, input_pos:input_pos+1] = v_val

        return k_out[layer_idx], v_out[layer_idx]

# Modified from https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/llama.py#L756
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache

# Copied from https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/llama.py#L767
def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

# Copied from https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/llama.py#L742
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Copied from https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/llama.py#L731
class FeedForward(nn.Module):
    def __init__(self, config: LocalArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# Modified from https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/llama.py#L615
class Attention(nn.Module):
    def __init__(self, config: LocalArgs, layer_idx: int, use_sdpa: bool = True):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.layer_idx = layer_idx

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(
            config.dim, total_head_dim, bias=config.attention_qkv_bias
        )
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

        self.dropout = config.dropout
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.use_sdpa = use_sdpa
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[int] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if self.use_sdpa:
            if mask is None:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    y = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=True,
                        # No third party attn_mask here to use flash_attention
                    )
            else:
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        else:
            y = self.eq_scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        return self.wo(y)

    def eq_scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
    ) -> torch.Tensor:
        # This is a standard scaled dot product attention
        # It's low efficient, but it doesn't raise cuda error

        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(1, 1, L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        return attn_weight @ value

# Copied from https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/llama.py#L599
class TransformerBlock(nn.Module):
    def __init__(self, config: LocalArgs, layer_idx: int, use_sdpa: bool = True) -> None:
        super().__init__()
        self.attention = Attention(config, layer_idx, use_sdpa=use_sdpa)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: int = None, kv_cache: KVCache = None
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos, kv_cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

# Modified from https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/llama.py#L470
class AudioTransformer(nn.Module):
    def __init__(self, config, use_sdpa: bool = False):
        super().__init__()
        self.config = LocalArgs()
        self.config.codebook_size = config.codebook_size
        self.config.num_codebooks = config.num_codebooks
        if hasattr(config, "min_audio_token_id"):
            self.config.min_audio_token_id = config.min_audio_token_id
            self.config.max_audio_token_id = config.max_audio_token_id
        self.config.n_layer = 4
        self.config.dim = 1024
        self.config.n_head = 32
        self.config.n_local_heads = 32
        self.config.intermediate_size = 2816
        self.config.head_dim = self.config.dim // self.config.n_head
        self.config.norm_eps = 1e-5
        self.config.attention_qkv_bias = False
        self.config.dropout = 0.0

        self.embeddings = nn.Embedding(self.config.codebook_size, self.config.dim)
        if self.config.dim != config.hidden_size:
            self.input_proj = nn.Linear(config.hidden_size, self.config.dim, bias=False)
        else:
            self.input_proj = nn.Identity()
        self.layers = nn.ModuleList(
                TransformerBlock(self.config, layer_idx, use_sdpa=use_sdpa) for layer_idx in range(self.config.n_layer)
        )
        self.norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)
        self.token_head = nn.Linear(self.config.dim, self.config.codebook_size, bias=False)
        self.gradient_checkpointing = False

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.config.num_codebooks, self.config.dim // self.config.n_head, 10000),
            persistent=False,
        )
        self.register_buffer(
            "attention_mask",
            torch.tril(torch.ones(self.config.num_codebooks, self.config.num_codebooks, dtype=torch.bool)),
            persistent=False,
        )

    def run_model(self, hidden_states, freqs_cis, attention_mask, input_pos: int = None, kv_cache: KVCache = None):
        for layer in self.layers:
            # TODO: gradient_checkpointing is disabled because of bug
            if False: # self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    freqs_cis,
                    attention_mask,
                    use_reentrant=True,
                )
            else:
                hidden_states = layer(hidden_states, freqs_cis, attention_mask, input_pos, kv_cache)
        hidden_states = self.norm(hidden_states)
        logits = self.token_head(hidden_states)
        return logits.float()

    # inp: [bs, hidden_size]
    # labels: [bs, num_codebooks]
    # logits: [bs, num_codebooks, codebook_size]
    def forward(self, inp, labels):
        bs = inp.shape[0]

        hidden_states = self.input_proj(inp)
        if self.freqs_cis.dtype != hidden_states.dtype:
            self.freqs_cis = self.freqs_cis.to(dtype=hidden_states.dtype)
        if labels is not None:
            # Training mode
            # Get embedding
            assert bs == labels.shape[0] and labels.shape[1] == self.config.num_codebooks, f"Labels shape error: {labels.shape}"
            hidden_states = [hidden_states[:, None, :], self.embeddings(labels[..., :-1]).to(hidden_states.dtype)]
            hidden_states = torch.cat(hidden_states, dim=1) # [bs, num_codebooks, hidden_size]
            # Run attention layers
            logits = self.run_model(hidden_states, self.freqs_cis, self.attention_mask)
        else:
            # Inference mode
            raise RuntimeError(f"Please call function \"inference\" in inference mode")
        return logits

    # inp: [bs, seq_len, hidden_size]
    # out_tokens: [bs, 1, num_codebooks]
    @torch.inference_mode()
    def inference(self, inp, temperature=0, top_k=0):
        # Only use the last hidden states for token computation
        inp = inp[:, -1:, :]

        bs = inp.shape[0]
        if self.freqs_cis.dtype != inp.dtype:
            self.freqs_cis = self.freqs_cis.to(dtype=inp.dtype)

        inp = self.input_proj(inp)

        # Inference mode
        kv_cache = KVCache(
                self.config.n_layer,
                bs,
                self.config.num_codebooks,
                self.config.n_head,
                self.config.head_dim,
                dtype=inp.dtype,
                device=inp.device,
        )
        # Generate one token per step
        out_tokens = []
        for input_pos in range(self.config.num_codebooks):
            inp = inp.reshape(bs, 1, self.config.dim)
            local_freqs_cis = self.freqs_cis[input_pos]
            local_mask = self.attention_mask[None, None, input_pos, :self.config.num_codebooks]

            logits = self.run_model(inp, local_freqs_cis, local_mask, input_pos, kv_cache)
            logits = logits.squeeze(dim=1)

            # Apply temperature and top-k
            if temperature > 0:
                logits = logits / temperature
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))  # Safety check
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, -float("Inf"))

            # Do sample
            probs = nn.functional.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            next_tokens = next_tokens.reshape(bs, 1, 1)
            inp = self.embeddings(next_tokens)
            out_tokens.append(next_tokens)

        return torch.cat(out_tokens, dim=-1)
