from transformers import GPT2LMHeadModel, GPT2Config
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import torch
import torch.nn as nn

from typing import Optional
from dataclasses import dataclass

from model_extensions import S4D, CustomMLP, set_identity_layernorms

# ================================
# Configuration classes
# ================================
@dataclass
class HybridConfig:
    vocab_size: int
    n_positions: int
    n_embd: int = 256
    n_head: int = 4
    dropout: float = 0.0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    nope: bool = False
    # Repeat `layer_pattern` this many times (each char is one block: "a" = GPT-2, "s" = SSM).
    n_pattern_repeats: int = 1
    layer_pattern: str = "sa"
    between_block_mlp_layers: int = 1
    layer_norm: bool = True
    # between_block_mlp_norm: bool = False
    ssm_kernel: str = "s4"

@dataclass
class S4Config:
    vocab_size: int
    n_embd: int = 256
    n_layers: int = 4
    dropout: float = 0.2
    ssm_kernel: str = "s4"
    between_block_mlp_layers: int = 1
    layer_norm: bool = True
    # between_block_mlp_norm: bool = False


def make_ssm_module(d_model: int, dropout: float, transposed: bool, kernel: str, **kernel_kwargs):
    k = kernel.lower().strip()
    if k == "s4":
        return S4D(d_model, dropout=dropout, transposed=transposed, **kernel_kwargs)
    if k == "mamba":
        raise NotImplementedError(
            "ssm_kernel='mamba' is not implemented in this repo; use ssm_kernel='s4' or add a Mamba block in model_extensions.py."
        )
    raise ValueError(f"Unknown ssm_kernel {kernel!r} (expected 's4' or 'mamba').")


class NoPE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return 0

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    """Base class that applies CustomMLP to every block and optionally disables layer norms."""
    def __init__(self, config):
        super().__init__(config)
        for block in self.transformer.h:
            block.mlp = CustomMLP(config.n_embd, config.between_block_mlp_layers, config)
        if not config.layer_norm:
            set_identity_layernorms(self)


class NoPEGPT2LMHeadModel(CustomGPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer.wpe = NoPE()


class RegGPT2LMHeadModel(CustomGPT2LMHeadModel):
    def __init__(self, config, coef):
        super().__init__(config)
        self.coef = coef

    def forward(self, *args, labels: Optional[torch.LongTensor] = None, **kwargs):
        outputs = super().forward(*args, labels=labels, **kwargs)

        if labels is not None:
            loss2 = self.compute_regularizer()

            if isinstance(outputs, tuple):
                outputs = (outputs[0] + loss2 * self.coef,) + outputs[1:]
            else:
                outputs.loss = outputs.loss + loss2 * self.coef
        return outputs
    
    def compute_regularizer(self):
        pe = self.transformer.wpe.weight # (num_embeddings, embedding_dim)

        square_sum = 0
        for block in self.transformer.h:
            w_matrix = block.attn.c_attn.weight # W_qkv for this layer (including all heads), 
            # it can first be split (by columns) into 3 equal part, correspond to q, k, v. Each part then be spit into many parts for each head
            k_offset = block.attn.embed_dim
            head_dim = block.attn.head_dim
            for i in range(block.attn.num_heads):
                w_query = w_matrix[:, i*head_dim : (i+1)*head_dim]  # W_q for head i
                w_key = w_matrix[:, k_offset+i*head_dim : k_offset+(i+1)*head_dim]  # W_k for head i

                product = (pe @ w_query) @ ((pe @ w_key).T)
                product = (torch.tril(product)**2).sum(dim=0).mean()
                square_sum = square_sum + product

        return square_sum
    


# Dropout broke in PyTorch 1.11

if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



class S4ForSequenceModeling(nn.Module):
    """Code taken from https://github.com/state-spaces/s4."""

    def __init__(self, config: S4Config):
        super().__init__()

        self.config = config
        self.vocab_size = self.config.vocab_size
        self.embed_dim = config.n_embd

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)

        self.s4_layers = nn.ModuleList([
            make_ssm_module(
                config.n_embd,
                dropout=config.dropout,
                transposed=False,
                kernel=config.ssm_kernel,
            )
            for _ in range(config.n_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(config.n_embd)
            for _ in range(config.n_layers)
        ])

        self.dropouts = nn.ModuleList([
            nn.Dropout(config.dropout)
            for _ in range(config.n_layers)
        ])

        self.mlps = nn.ModuleList()
        for _ in range(config.n_layers):
            self.mlps.append(
                CustomMLP(config.n_embd, config.between_block_mlp_layers, config)
            )

        self.decoder = nn.Linear(config.n_embd, self.vocab_size)
        self.decoder.weight = self.wte.weight # share embedding weights

        if config.layer_norm == False:
            set_identity_layernorms(self)

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
        """
        Computes the loss directly to be used in the Trainer. Otherwise adapted code from https://github.com/state-spaces/s4.
        """

        if input_ids is not None:
            x = self.wte(input_ids) # (B, L, n_embd)
        else:
            x = inputs_embeds

        for layer, mlp, norm, dropout in zip(self.s4_layers, self.mlps, self.norms, self.dropouts):
            z, _ = layer(x)   # (B, L, n_embd) in and out
            z = mlp(z)
            z = dropout(z)
            x = norm(x + z)

        logits = self.decoder(x)

        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(logits, labels, self.vocab_size)

        return CausalLMOutput(
            loss=loss,
            logits=logits
        )

def _expand_hybrid_pattern(pattern: str, n_repeats: int) -> tuple[list[str], str]:
    motif = pattern.strip().lower()
    if not motif or any(c not in "as" for c in motif):
        raise ValueError(f"layer_pattern must be non-empty and only contain 'a' and 's', got {pattern!r}")
    expanded = list(motif * n_repeats)
    return expanded, motif * n_repeats


class HybridGPT2S4LMHeadModel(nn.Module):
    """
    Stacks GPT-2 blocks and SSM blocks according to `config.layer_pattern` repeated
    `config.n_pattern_repeats` times (e.g. pattern \"sa\" x 2 → S,A,S,A).
    """

    def __init__(self, config: HybridConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.n_embd

        self.layer_kinds, self._flat_pattern = _expand_hybrid_pattern(
            config.layer_pattern, config.n_pattern_repeats
        )

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = NoPE() if config.nope else nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        gpt2_cfg = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.n_positions,
            n_embd=config.n_embd,
            n_layer=1,
            n_head=config.n_head,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            pad_token_id=config.pad_token_id,
            attn_pdrop=config.dropout,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
        )

        n_ssm = self.layer_kinds.count("s")

        self.blocks = nn.ModuleList()
        for i, kind in enumerate(self.layer_kinds):
            if kind == "a":
                gpt2_block = GPT2Block(gpt2_cfg, layer_idx=i)
                gpt2_block.mlp = CustomMLP(config.n_embd, config.between_block_mlp_layers, config)
                self.blocks.append(gpt2_block)
            else:
                self.blocks.append(
                    make_ssm_module(
                        config.n_embd,
                        dropout=config.dropout,
                        transposed=False,
                        kernel=config.ssm_kernel,
                    )
                )

        self.ssm_norms = nn.ModuleList(
            nn.LayerNorm(config.n_embd) for _ in range(n_ssm)
        )
        self.ssm_dropouts = nn.ModuleList(
            nn.Dropout(config.dropout) for _ in range(n_ssm)
        )
        self.ssm_mlps = nn.ModuleList(
            CustomMLP(config.n_embd, config.between_block_mlp_layers, config)
            for _ in range(n_ssm)
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        if not config.layer_norm:
            set_identity_layernorms(self)

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, labels=None, **kwargs):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            hidden_states = self.wte(input_ids)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            seq_len, d_embd = inputs_embeds.size(-2), inputs_embeds.size(-1)
            hidden_states = inputs_embeds.contiguous().view(-1, seq_len, d_embd)
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = hidden_states.device
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        else:
            seq_len = hidden_states.size(1)
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(hidden_states.size(0), -1)

        hidden_states = hidden_states + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)

        ssm_idx = 0
        for kind, block in zip(self.layer_kinds, self.blocks):
            if kind == "a":
                out = block(hidden_states, use_cache=False, output_attentions=False)
                hidden_states = out[0] if isinstance(out, (tuple, list)) else out
            else:
                z, _ = block(hidden_states)   # (B, L, D) in and out
                z = self.ssm_mlps[ssm_idx](z)
                z = self.ssm_dropouts[ssm_idx](z)
                hidden_states = self.ssm_norms[ssm_idx](hidden_states + z)
                ssm_idx += 1

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels_flat = labels.view(-1, input_shape[-1])
            loss = ForCausalLMLoss(logits, labels_flat, self.vocab_size)

        return CausalLMOutput(
            loss=loss,
            logits=logits
        )