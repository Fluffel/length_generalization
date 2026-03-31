from transformers import GPT2LMHeadModel, GPT2Config
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import torch
import torch.nn as nn

from typing import Optional
from dataclasses import dataclass

from model_extensions import S4D

# ================================
# Configuration classes
# ================================
@dataclass
class HybridConfig:
    vocab_size: int
    n_positions: int
    n_embd: int = 256
    n_layer: int = 4 # number of GPT-2 + S4 layers
    n_head: int = 4 # number of attention heads per GPT-2 layer
    dropout: float = 0.0 # dropout rate
    bos_token_id: Optional[int] = None # beginning of sequence token id
    eos_token_id: Optional[int] = None # end of sequence token id
    pad_token_id: Optional[int] = None # padding token id
    nope: bool = False # whether to use no positional encoding in the GPT-2 layers
    start_with_attention: bool = True # defines the order of layers: True = [GPT2, S4, GPT2, S4, ...], False = [S4, GPT2, S4, GPT2, ...]

@dataclass
class S4Config:
    vocab_size: int
    n_embd: int = 256
    n_layers: int = 4
    dropout: float = 0.2

class NoPE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return 0

class NoPEGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer.wpe = NoPE()

class RegGPT2LMHeadModel(GPT2LMHeadModel):
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

        # self.encoder = nn.Linear(config.d_input, config.d_model)

        self.s4_layers = nn.ModuleList([
            S4D(config.n_embd, dropout=config.dropout, transposed=True)
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

        self.decoder = nn.Linear(config.n_embd, self.vocab_size)

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
        """
        Computes the loss directly to be used in the Trainer. Otherwise adapted code from https://github.com/state-spaces/s4.
        """

        if input_ids is not None:
            x = self.wte(input_ids) # (B, L, n_embd)
        else:
            x = inputs_embeds

        x = x.transpose(-1, -2)  # (B, n_embd, L)

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x

            z, _ = layer(z)
            z = dropout(z)

            x = x + z
            x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)  # (B, L, n_embd)

        logits = self.decoder(x)

        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(logits, labels, self.vocab_size)

        return CausalLMOutput(
            loss=loss,
            logits=logits
        )

class HybridGPT2S4LMHeadModel(nn.Module):
    """
    Interleaves GPT-2 blocks and S4 blocks:
    [GPT2Block, S4D, GPT2Block, S4D, ...] (or the reverse).
    """

    def __init__(self, config: HybridConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.n_embd

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

        self.layers = nn.ModuleList()
        self.layer_kinds = []
        for i in range(2 * config.n_layer):
            use_attention = (i % 2 == 0) if config.start_with_attention else (i % 2 == 1)
            if use_attention:
                self.layers.append(GPT2Block(gpt2_cfg, layer_idx=i))
                self.layer_kinds.append("attn")
            else:
                self.layers.append(S4D(config.n_embd, dropout=config.dropout, transposed=True))
                self.layer_kinds.append("s4")

        self.s4_norms = nn.ModuleList(
            nn.LayerNorm(config.n_embd) for _ in range(self.layer_kinds.count("s4"))
        )
        self.s4_dropouts = nn.ModuleList(
            nn.Dropout(config.dropout) for _ in range(self.layer_kinds.count("s4"))
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight # TODO: should this be shared?

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, labels=None, **kwargs):
        # Match HuggingFace GPT2Model input flattening: merge all leading dims into one batch,
        # keep last dim as sequence length, so wte yields (batch, seq, hidden) for GPT2Block.
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

        s4_idx = 0
        for layer, layer_kind in zip(self.layers, self.layer_kinds):
            if layer_kind == "attn":
                # Newer transformers returns a tensor; older versions return (hidden_states, presents, ...).
                out = layer(hidden_states, use_cache=False, output_attentions=False)
                hidden_states = out[0] if isinstance(out, (tuple, list)) else out
            else:
                x = hidden_states.transpose(-1, -2)  # (B, D, L)
                z, _ = layer(x)
                z = self.s4_dropouts[s4_idx](z)
                x = x + z
                hidden_states = self.s4_norms[s4_idx](x.transpose(-1, -2)) # (B, L, D)
                s4_idx += 1

        hidden_states = self.ln_f(hidden_states) # last layer norm
        logits = self.lm_head(hidden_states)  # (flat_batch, seq, vocab); matches flattened ids/embeds

        loss = None
        if labels is not None:
            labels_flat = labels.view(-1, input_shape[-1])
            loss = ForCausalLMLoss(logits, labels_flat, self.vocab_size)

        # Restore leading dims like GPT2Model (optional for callers; loss used flat logits above).
        # For input_ids of shape (L,) only one dim exists; it is the sequence length, so treat as batch 1.
        # Otherwise (-1,) + input_shape[1:] + (V,) drops seq: (1,L,V) would incorrectly become (L,V).
        # if len(input_shape) == 1:
        #     output_shape = (1, input_shape[0], logits.size(-1))
        # else:
        # output_shape = (-1,) + input_shape[1:] + (logits.size(-1),)
        # logits = logits.view(output_shape)

        return CausalLMOutput(
            loss=loss,
            logits=logits
        )