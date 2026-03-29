from transformers import GPT2LMHeadModel
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_outputs import CausalLMOutput
import torch
import torch.nn as nn

from typing import Optional
from dataclasses import dataclass

from model_extensions import S4D

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

@dataclass
class S4Config:
    vocab_size: int
    n_embd: int = 256
    n_layers: int = 4
    dropout: float = 0.2

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