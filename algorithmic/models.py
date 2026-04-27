import copy
import importlib
import logging
from typing import Optional

from transformers import GPT2LMHeadModel, GPT2Config
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_outputs import CausalLMOutput
import torch
import torch.nn as nn

from mambapy.mamba import ResidualBlock as MambaResidualBlock
from olmo_core.nn.transformer.config import TransformerConfig, TransformerBlockConfig
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from utils import ArchSlot, HybridConfig, RunConfig, SSMConfig, create_hybrid_config, create_ssm_config, create_transformer_config, mamba_config_from_ssm_config
from utils import run_length_encode
from model_extensions import S4D, CustomMLP, set_identity_layernorms

# AttentionConfig = None
# GatedDeltaNetConfig = None
# TransformerConfig = None
# TransformerBlockConfig = None

LOGGER = logging.getLogger(__name__)


class NoPE(nn.Module):
    """Drop-in replacement for a positional-embedding module that contributes nothing.

    Returns a 0-dim zero *tensor* rather than the Python scalar 0 so that downstream
    code which does things like ``self.wpe(position_ids).to(inputs_embeds.device)``
    (see ``transformers.models.gpt2.modeling_gpt2.GPT2Model.forward``) works
    unchanged. The 0-dim tensor broadcasts against any embedding tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_zero", torch.zeros(()), persistent=False)

    def forward(self, x):
        return self._zero

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    """Base class that applies CustomMLP to every block and optionally disables layer norms.
    Args:

    as_module (bool): If set to true, this will remove all embedding and head layers, such that the model can be used within layers.
    """
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

def make_ssm_module(config: SSMConfig):
    k = config.ssm_kernel.lower().strip()
    if k == "s4":
        return S4D(config.n_embd, dropout=config.dropout, transposed=False)
    if k == "mamba":
        return MambaResidualBlock(mamba_config_from_ssm_config(config))
    raise ValueError(f"Unknown ssm_kernel {config.ssm_kernel!r} (expected 's4' or 'mamba').")


class SSMModel(nn.Module):
    """Code taken from https://github.com/state-spaces/s4."""

    def __init__(self, config: SSMConfig):
        super().__init__()

        self.config = config
        self.vocab_size = self.config.vocab_size
        self.embed_dim = config.n_embd

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)

        self.ssm_layers = nn.ModuleList([
            make_ssm_module(config)
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

        for layer, mlp, norm, dropout in zip(self.ssm_layers, self.mlps, self.norms, self.dropouts):
            z = layer(x)   # (B, L, n_embd) in and out
            z = mlp(z) # moved here from the S4 code
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

def _expand_hybrid_pattern(pattern: str, n_repeats: int) -> tuple[list[str], list[tuple[str, int]]]:
    """
    Expand the hybrid pattern into a list of symbols and a list of (symbol, count) pairs.
    Consecutive 'a' layers are collapsed into a single block (counted) so they can be
    handled by one GPT-2 stack, whereas each 's' remains its own block.
    """
    motif = pattern.strip().lower()
    if not motif or any(c not in "as" for c in motif):
        raise ValueError(f"layer_pattern must be non-empty and only contain 'a' and 's', got {pattern!r}")

    # Repeat first so that adjacent 'a' runs across pattern repeats merge together
    # (e.g. pattern 'a' repeated 2x becomes a single block of count 2, not two of count 1).
    expanded = motif * n_repeats

    run_length_encoding: list[tuple[str, int]] = []
    i = 0
    while i < len(expanded):
        if expanded[i] == 'a':
            j = i
            while j < len(expanded) and expanded[j] == 'a':
                j += 1
            run_length_encoding.append(('a', j - i))
            i = j
        else:  # 's'
            run_length_encoding.append(('s', 1))
            i += 1

    layer_kinds = [kind for kind, _ in run_length_encoding]
    return layer_kinds, run_length_encoding



class HybridSSMTransformerModel(nn.Module):
    """
    Stacks GPT-2 blocks and SSM blocks according to `config.layer_pattern` repeated
    `config.n_pattern_repeats` times (e.g. pattern \"sa\" x 2 → S,A,S,A).
    """

    def __init__(self, config: HybridConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.n_embd

        self.layer_kinds, self.layer_kinds_rle = _expand_hybrid_pattern(
            config.layer_pattern, config.n_pattern_repeats
        )

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = NoPE() if config.nope else nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Match GPT2PreTrainedModel._init_weights for the outer embeddings, otherwise
        # nn.Embedding's default std=1 init (vs GPT-2's std=initializer_range≈0.02)
        # blows up the first attention's QK products by ~(1/0.02)^2 and the softmax
        # saturates on step 0 -> training never gets off the ground.
        init_std = getattr(config, "initializer_range", 0.02)
        nn.init.normal_(self.wte.weight, mean=0.0, std=init_std)
        if isinstance(self.wpe, nn.Embedding):
            nn.init.normal_(self.wpe.weight, mean=0.0, std=init_std)

        gpt2_cfg = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.n_positions,
            n_embd=config.n_embd,
            n_layer=1,
            n_head=config.n_head,
            between_block_mlp_layers=config.between_block_mlp_layers,
            layer_norm=config.layer_norm,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            pad_token_id=config.pad_token_id,
            attn_pdrop=config.dropout,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
        )
        ssm_config = SSMConfig(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            dropout=config.dropout,
            ssm_kernel=config.ssm_kernel,
        )
        # gpt2_cfg._attn_implementation = "eager"
        # self.gpt2_cfg = gpt2_cfg

        n_ssm = self.layer_kinds.count("s")
    
        self.blocks = nn.ModuleList()
        for kind, hidden_layer_count in self.layer_kinds_rle:
            if kind == "a":
                cfg = copy.deepcopy(gpt2_cfg)
                cfg.n_layer = hidden_layer_count
                gpt_2_head_model = CustomGPT2LMHeadModel(cfg)
                # Neutralize the inner model's own embedding-side and final pieces so
                # that only the outer wpe / drop / ln_f / lm_head are applied. This
                # makes the hybrid with pattern "a"*k equivalent to a standalone
                # CustomGPT2LMHeadModel with n_layer=k (up to the unused inner wte).
                gpt_2_head_model.transformer.wpe = NoPE()
                gpt_2_head_model.lm_head = nn.Identity()
                gpt_2_head_model.transformer.drop = nn.Identity()
                gpt_2_head_model.transformer.ln_f = nn.Identity()

                self.blocks.append(gpt_2_head_model)
            else:
                self.blocks.append(
                    make_ssm_module(ssm_config)
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

        inputs_embeds = hidden_states
        device = hidden_states.device
        cache_position = torch.arange(inputs_embeds.shape[1], device=device, dtype=torch.long)
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        else:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = hidden_states + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)

        ssm_idx = 0
        for kind, block in zip(self.layer_kinds, self.blocks):
            if kind == "a":
                # Pass the already-embedded hidden states as `inputs_embeds` so the
                # inner GPT-2 does not try to look them up in its own wte. The inner
                # wpe / drop / ln_f have been replaced with no-ops in __init__, so
                # this is just the stack of attention+MLP blocks plus an Identity head.
                out = block(inputs_embeds=hidden_states)
                hidden_states = out.logits if hasattr(out, "logits") else (
                    out[0] if isinstance(out, (tuple, list)) else out
                )
            else:
                z = block(hidden_states)   # (B, L, D) in and out
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


class OLMoCoreCausalLMAdapter(nn.Module):
    """Adapter to make an OLMo-core model compatible with HuggingFace Trainer output expectations."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Keep compatibility with logging code that probes `model.wte.weight.std()`.
        if hasattr(model, "embeddings"):
            self.wte = model.embeddings

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
        position_ids=None,
        **kwargs,
    ):
        del position_ids # OLMo-core does not use explicit absolute position IDs.

        if inputs_embeds is not None:
            raise ValueError("OLMo-core adapter does not support inputs_embeds; pass input_ids.")
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        # HF Trainer can pass this, but OLMo-core forward does not use it.
        kwargs.pop("attention_mask", None)

        logits = self.model(
            input_ids=input_ids,
            return_logits=True,
            **kwargs,
        )
        if labels is not None:
            # Keep loss semantics identical to the local GPT-2/SSM paths:
            # ForCausalLMLoss applies the autoregressive shift used by compute_metrics.
            loss = ForCausalLMLoss(logits, labels, logits.size(-1))
            return CausalLMOutput(loss=loss, logits=logits)

        return CausalLMOutput(loss=None, logits=logits)


def _require_olmo_core():
    global TransformerConfig, AttentionConfig, GatedDeltaNetConfig, TransformerBlockConfig
    if all(x is not None for x in (TransformerConfig, AttentionConfig, GatedDeltaNetConfig, TransformerBlockConfig)):
        return

    try:
        TransformerConfig = importlib.import_module("olmo_core.nn.transformer").TransformerConfig
        AttentionConfig = importlib.import_module("olmo_core.nn.attention").AttentionConfig
        GatedDeltaNetConfig = importlib.import_module("olmo_core.nn.attention.recurrent").GatedDeltaNetConfig
        TransformerBlockConfig = importlib.import_module("olmo_core.nn.transformer.config").TransformerBlockConfig
    except ImportError as exc:
        raise ImportError(
            "OLMo-core is not available. Install `olmo-core` to use run_config.use_olmo_core=True. See this official repo for installation https://github.com/allenai/OLMo-core/blob/main/README.md. Installation needs to be done by cloning the official git repository and not the ai2-olmo-core package."
        ) from exc


def _olmo_head_dim(arch: ArchSlot, run_config: RunConfig) -> int:
    raw_dim = run_config.olmo_gdn_head_dim_multiplier * (arch.d_model / arch.n_head)
    head_dim = max(1, int(raw_dim))
    if head_dim * arch.n_head > arch.d_model:
        raise ValueError(
            f"Invalid OLMo GDN head dim={head_dim} for d_model={arch.d_model}, n_head={arch.n_head}. "
            f"Reduce olmo_gdn_head_dim_multiplier (currently {run_config.olmo_gdn_head_dim_multiplier})."
        )
    return head_dim


def _build_olmo_base_transformer_config(
    *,
    vocab_size: int,
    n_layers: int,
    arch: ArchSlot,
):
    _require_olmo_core()
    cfg = TransformerConfig.llama_like(
        d_model=arch.d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=arch.n_head,
        n_kv_heads=arch.n_head,
        use_flash=False,
    )
    assert isinstance(cfg.block, TransformerBlockConfig)
    assert isinstance(cfg.block.sequence_mixer, AttentionConfig)
    if arch.dropout:
        cfg.block.dropout = arch.dropout
    return cfg


def _build_olmo_transformer_model(run_config: RunConfig, arch: ArchSlot, tokenizer):
    cfg = _build_olmo_base_transformer_config(
        vocab_size=len(tokenizer),
        n_layers=arch.n_layer,
        arch=arch,
    )
    if run_config.use_nope:
        cfg.block = cfg.block.replace(
            sequence_mixer=cfg.block.sequence_mixer.replace(rope=None)
        )
    model = cfg.build()
    return OLMoCoreCausalLMAdapter(model)


def _build_olmo_gdn_model(run_config: RunConfig, arch: ArchSlot, tokenizer):
    cfg = _build_olmo_base_transformer_config(
        vocab_size=len(tokenizer),
        n_layers=arch.n_layer,
        arch=arch,
    )
    gdn_cfg = GatedDeltaNetConfig(
        n_heads=arch.n_head,
        head_dim=_olmo_head_dim(arch, run_config),
        expand_v=run_config.olmo_gdn_expand_v,
        allow_neg_eigval=run_config.olmo_gdn_allow_neg_eigval,
    )
    cfg.block = cfg.block.replace(sequence_mixer=gdn_cfg)
    model = cfg.build()
    return OLMoCoreCausalLMAdapter(model)


def _build_olmo_hybrid_model(run_config: RunConfig, arch: ArchSlot, tokenizer):
    motif = run_config.hybrid_layer_pattern.strip().lower()
    if not motif or any(c not in "as" for c in motif):
        raise ValueError(
            f"hybrid_layer_pattern must be non-empty and contain only 'a' and 's', got {run_config.hybrid_layer_pattern!r}"
        )

    # Keep the same semantics as existing hybrid: repeat the pattern `arch.n_layer` times.
    pattern = list(motif) * arch.n_layer
    cfg = _build_olmo_base_transformer_config(
        vocab_size=len(tokenizer),
        n_layers=len(pattern),
        arch=arch,
    )
    if run_config.use_nope:
        attn_block = cfg.block.replace(
            sequence_mixer=cfg.block.sequence_mixer.replace(rope=None)
        )
    else:
        attn_block = cfg.block
    gdn_block = attn_block.replace(
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=arch.n_head,
            head_dim=_olmo_head_dim(arch, run_config),
            expand_v=run_config.olmo_gdn_expand_v,
            allow_neg_eigval=run_config.olmo_gdn_allow_neg_eigval,
        )
    )
    cfg.block = {"attn": attn_block, "gdn": gdn_block}
    cfg.block_pattern = ["attn" if c == "a" else "gdn" for c in pattern]

    model = cfg.build()
    return OLMoCoreCausalLMAdapter(model)

def build_model(run_config: RunConfig, arch: ArchSlot, tokenizer, n_positions: int):
    if run_config.use_olmo_core:
        if arch.layer_norm is False:
            raise ValueError("OLMo-core builders currently require layer_norm=True.")
        if arch.between_block_mlp_layers != 1:
            LOGGER.warning(
                "OLMo-core feed-forward uses one block FFN; between_block_mlp_layers=%s is ignored.",
                arch.between_block_mlp_layers,
            )
        if run_config.regularize != 0:
            LOGGER.warning("regularize=%s is ignored for OLMo-core models.", run_config.regularize)

        match run_config.model_family:
            case "transformer":
                LOGGER.info("Building OLMo-core transformer model")
                return _build_olmo_transformer_model(run_config, arch, tokenizer)
            case "ssm":
                LOGGER.info(
                    "Building OLMo-core standalone GDN model allow_neg_eigval=%s",
                    run_config.olmo_gdn_allow_neg_eigval,
                )
                return _build_olmo_gdn_model(run_config, arch, tokenizer)
            case "hybrid":
                LOGGER.info(
                    "Building OLMo-core hybrid model pattern=%s allow_neg_eigval=%s",
                    run_config.hybrid_layer_pattern,
                    run_config.olmo_gdn_allow_neg_eigval,
                )
                return _build_olmo_hybrid_model(run_config, arch, tokenizer)
            case _:
                raise ValueError(run_config.model_family)

    match run_config.model_family:
        case "transformer":
            cfg = create_transformer_config(tokenizer, n_positions, arch)

            if run_config.use_nope:
                LOGGER.info("Building transformer model variant=nope")
                return NoPEGPT2LMHeadModel(cfg)
            if run_config.regularize != 0:
                LOGGER.info("Building transformer model variant=regularized coef=%s", run_config.regularize)
                return RegGPT2LMHeadModel(cfg, run_config.regularize)

            LOGGER.info("Building transformer model variant=custom")
       
            return CustomGPT2LMHeadModel(cfg)
        case "ssm":
            cfg = create_ssm_config(tokenizer, run_config, arch)
            LOGGER.info("Building ssm model kernel=%s", run_config.ssm_kernel)
            return SSMModel(cfg)
        case "hybrid":
            cfg = create_hybrid_config(tokenizer, n_positions, run_config, arch)
            LOGGER.info(
                "Building hybrid model pattern=%s kernel=%s nope=%s",
                run_config.hybrid_layer_pattern,
                run_config.ssm_kernel,
                run_config.use_nope,
            )
            return HybridSSMTransformerModel(cfg)
        case _:
            raise ValueError(run_config.model_family)


if __name__ == "__main__":
    pattern = "saass"
    n_repeats = 2
    layer_kinds, run_length_encoded = _expand_hybrid_pattern(pattern, n_repeats)
    print("layer_kinds:", layer_kinds)
    for i, (kind, hidden_layer_count) in enumerate(run_length_encoded):
        print(f"Block {i}: {kind} with {hidden_layer_count} sub-layers")