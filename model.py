import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import ModelOutput, logging
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel

from audio_transformer import AudioTransformer

logger = logging.get_logger(__name__)


# Copied from https://github.com/pytorch/audio/blob/main/src/torchaudio/models/wav2vec2/components.py#L43
class LayerNorm(torch.nn.LayerNorm):
    """Layer norm with transpose"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.transpose(-2, -1)
        x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(-2, -1)
        return x

# Copied from https://github.com/pytorch/audio/blob/main/src/torchaudio/models/wav2vec2/components.py#L53
class ConvLayerBlock(torch.nn.Module):
    """Convolution unit of FeatureExtractor"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        layer_norm: Optional[torch.nn.Module],
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x (Tensor): Shape: ``[batch, in_channels, in_frame]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
            Optional[Tensor]: Shape ``[batch, ]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = torch.nn.functional.gelu(x)

        return x

# Copied from https://github.com/pytorch/audio/blob/main/src/torchaudio/models/wav2vec2/components.py#L146
class FeatureProjection(torch.nn.Module):
    """Layer that connects FeatureExtractor and Encoder

    Projects features to encoder dimension.

    Args:
        in_features (int): Input feature dim.
        out_features (int): Output feature dim.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout=0.1,
    ):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(in_features)
        self.projection = torch.nn.Linear(
            in_features,
            out_features,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor):
                Feature Tensor. shape: ``[batch, frame, in_feature]``
        Returns:
            Tensor: Projected features. ``[batch, frame, out_feature]``.
        """
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x

# Modified from https://github.com/pytorch/audio/blob/main/src/torchaudio/models/wav2vec2/components.py#L102
class FeatureExtractor(torch.nn.Module):
    """Extract features from audio

    Args:
        conv_layers (nn.ModuleList):
            convolution layers
    """

    def __init__(
        self,
        shapes=[(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)],
        bias=False,
        norm_mode="group_norm",
    ):
        super().__init__()
        if norm_mode not in ["group_norm", "layer_norm"]:
            raise ValueError("Invalid norm mode")
        blocks = []
        in_channels = 1
        for i, (out_channels, kernel_size, stride) in enumerate(shapes):
            normalization = None
            if norm_mode == "group_norm" and i == 0:
                normalization = torch.nn.GroupNorm(
                    num_groups=out_channels,
                    num_channels=out_channels,
                    affine=True,
                )
            elif norm_mode == "layer_norm":
                normalization = LayerNorm(
                    normalized_shape=out_channels,
                    elementwise_affine=True,
                )
            blocks.append(
                ConvLayerBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    layer_norm=normalization,
                )
            )
            in_channels = out_channels
        self.conv_layers = torch.nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x (Tensor):
                Input Tensor representing a batch of audio,
                shape: ``[batch, time]``.

        Returns:
            Tensor:
                The resulting feature, shape: ``[batch, frame, feature]``
            Optional[Tensor]:
                Valid length of each output sample. shape: ``[batch, ]``.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected the input Tensor to be 2D (batch, time). Found: {list(x.shape)}")

        x = x.unsqueeze(1)  # (batch, channel==1, frame)
        for layer in self.conv_layers:
            x = layer(x)  # (batch, feature, frame)
        x = x.transpose(1, 2)  # (batch, frame, feature)
        return x

# Modified from https://github.com/pytorch/audio/blob/main/src/torchaudio/models/wav2vec2/components.py#L102
class FeatureExtractorAdapter(torch.nn.Module):
    """Extract features from audio

    Args:
        conv_layers (nn.ModuleList):
            convolution layers
    """

    def __init__(
        self,
        shapes=(512, 512, 2, 2),
        hidden_size=2048,
        bias=False,
        norm_mode="group_norm",
    ):
        super().__init__()
        if norm_mode not in ["group_norm", "layer_norm"]:
            raise ValueError("Invalid norm mode")
        in_channels, out_channels, kernel_size, stride = shapes
        normalization = LayerNorm(
            normalized_shape=out_channels,
            elementwise_affine=True,
        )
        self.conv_layers = ConvLayerBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=False,
                    layer_norm=normalization,
                )
        self.feat_proj = FeatureProjection(out_channels, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x (Tensor):
                Input Tensor representing a batch of audio,
                shape: ``[batch, time]``.

        Returns:
            Tensor:
                The resulting feature, shape: ``[batch, frame, feature]``
            Optional[Tensor]:
                Valid length of each output sample. shape: ``[batch, ]``.
        """
        x = x.transpose(1, 2)  # (batch, feature, frame)
        x = self.conv_layers(x)  # (batch, feature, frame)
        x = x.transpose(1, 2)  # (batch, frame, feature)
        x = self.feat_proj(x)
        return x

@dataclass
class VoilaOutput(ModelOutput):
    """
    Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L678

    Base class for Voila outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The hidden state of the last attention layer.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    voila_pred: Optional[torch.FloatTensor] = None


# Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1103
class VoilaModel(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pad_vocab_size_multiple = 64

        self.ref_emb_linear = nn.Linear(256, config.hidden_size, bias=True)
        self.audio_transformer = AudioTransformer(config, use_sdpa=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        audio_labels: Optional[torch.LongTensor] = None,
        ref_embs: Optional[List[torch.Tensor]] = None,
        ref_embs_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, VoilaOutput]:
        r"""
        Args:
            input_ids: [bs, seq_len, num_codebooks]
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        assert len(inputs_embeds.shape) == 4
        if len(inputs_embeds.shape) == 4:
            inputs_embeds = inputs_embeds.mean(dim=2)

        if self.training or \
                (past_key_values is None and ref_embs is not None) or \
                (past_key_values is not None and past_key_values.get_seq_length() < 4 and ref_embs is not None):
            ref_embs = self.ref_emb_linear(ref_embs.to(self.ref_emb_linear.weight.dtype))
            ref_embs = ref_embs * ref_embs_mask.unsqueeze(-1).unsqueeze(-1)
            # (padding_left,padding_right,padding_top,padding_bottom,padding_front,padding_back)
            padding = (0, 0, 4, inputs_embeds.shape[1] - 5, 0, 0)
            ref_embs = torch.nn.functional.pad(ref_embs, padding, mode='constant', value=0.0)
            inputs_embeds = inputs_embeds + ref_embs

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VoilaOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _prepare_inputs_for_generation(
        self, input_ids, ref_embs=None, ref_embs_mask=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_cache_shape()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is None and \
                (past_key_values is None or past_key_values.get_seq_length() <= 0):
            inputs_embeds = self.model.embed_tokens(input_ids)
        if inputs_embeds is not None and \
                (past_key_values is None or past_key_values.get_seq_length() <= 0):
            model_inputs = {"inputs_embeds": inputs_embeds, "ref_embs": ref_embs, "ref_embs_mask": ref_embs_mask}
        else:
            model_inputs = {"input_ids": input_ids, "ref_embs": None}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        num_new_token: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], num_new_token))], dim=-1
            )

        return model_kwargs

    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[Union[int, List[int]]],
    ) -> torch.LongTensor:
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id not in eos_token_id)

        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return inputs.ne(pad_token_id).long()
        else:
            return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)

    @torch.inference_mode()
    def run_generate(
        self,
        input_ids: torch.LongTensor,
        ref_embs: Optional[List[torch.Tensor]] = None,
        ref_embs_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = 128,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        streamer: Optional["BaseStreamer"] = None,
        llm_audio_token_id: Optional[int] = None,
        min_audio_token_id: Optional[int] = None,
        temperature=0.2,
        top_k=50,
        audio_temperature=0.2,
        audio_top_k=50,
    ):
        assert eos_token_id is not None and pad_token_id is not None, "eos_token_id and pad_token_id are required for inference"
        assert llm_audio_token_id is not None and min_audio_token_id is not None, "llm_audio_token_id and min_audio_token_id are required for inference"
        assert len(input_ids.shape) == 2 or len(input_ids.shape) == 3, f"input_ids is supposed to be [batch, seq_len] or [batch, seq_len, num_codebooks], and got {input_ids.shape}"

        eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # Extend input_ids with additional num_codebooks dim
        if len(input_ids.shape) == 2:
            input_ids = input_ids[:, :, None].expand(1, 1, self.config.num_codebooks)

        this_peer_finished = False  # used by synced_gpus only
        max_length = input_ids.shape[1] + max_new_tokens

        model_kwargs = {
            "use_cache": True,
            "past_key_values": DynamicCache(),
            "attention_mask": self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            ),
        }
        # auto-regressive generation
        while True:
            # prepare model inputs
            model_inputs = self._prepare_inputs_for_generation(
                input_ids,
                ref_embs=ref_embs,
                ref_embs_mask=ref_embs_mask,
                **model_kwargs
            )

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
            )
            audio_tokens = self.audio_transformer.inference(
                outputs.last_hidden_state,
                temperature=audio_temperature,
                top_k=audio_top_k,
            )
            audio_tokens = torch.stack(
                [
                    audio_tokens[:, :, ci] + min_audio_token_id + ci*self.config.codebook_size
                    for ci in range(self.config.num_codebooks)
                ],
                dim=2,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            # Apply temperature and top-k
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))  # Safety check
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

            # sample
            probs = nn.functional.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Append NUM_CODEBOOK text tokens or audio_tokens
            if len(next_tokens.shape) == 1:
                next_tokens = next_tokens[:, None, None].expand(-1, 1, self.config.num_codebooks)
            next_tokens = torch.where(next_tokens==llm_audio_token_id, audio_tokens, next_tokens)

            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens[:, :, 0].ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=1)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if input_ids.shape[1] >= max_length:
                this_peer_finished = True

            if this_peer_finished:
                break

        if streamer is not None:
            streamer.end()

        return input_ids


# Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1103
class VoilaAudioAlphaModel(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pad_vocab_size_multiple = 64


        self.ref_emb_linear = nn.Linear(256, config.hidden_size, bias=True)
        self.audio_transformer = AudioTransformer(config, use_sdpa=False)

        self.feature_extractor = FeatureExtractor()
        self.audio_feature_extractor_adapter = FeatureExtractorAdapter(hidden_size=config.hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        audio_labels: Optional[torch.LongTensor] = None,
        ref_embs: Optional[List[torch.Tensor]] = None,
        ref_embs_mask: Optional[torch.LongTensor] = None,
        audio_datas: Optional[torch.FloatTensor] = None,
        audio_data_masks: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, VoilaOutput]:
        r"""
        Args:
            input_ids: [bs, seq_len, num_codebooks]
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        assert len(inputs_embeds.shape) == 4
        if len(inputs_embeds.shape) == 4:
            inputs_embeds = inputs_embeds.mean(dim=2)

        if self.training or \
                (past_key_values is None and ref_embs is not None) or \
                (past_key_values is not None and past_key_values.get_seq_length() < 4 and ref_embs is not None):
            ref_embs = self.ref_emb_linear(ref_embs.to(self.ref_emb_linear.weight.dtype))
            ref_embs = ref_embs * ref_embs_mask.unsqueeze(-1).unsqueeze(-1)
            # (padding_left,padding_right,padding_top,padding_bottom,padding_front,padding_back)
            padding = (0, 0, 4, inputs_embeds.shape[1] - 5, 0, 0)
            ref_embs = torch.nn.functional.pad(ref_embs, padding, mode='constant', value=0.0)
            inputs_embeds = inputs_embeds + ref_embs

        if self.training or audio_datas is not None:
            audio_embeds = self.feature_extractor(audio_datas)
            audio_embeds = self.audio_feature_extractor_adapter(audio_embeds)
            audio_embeds = audio_embeds * audio_data_masks[..., None]
            inputs_embeds = inputs_embeds + audio_embeds

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # We shift tokens and labels in dataloader
            shift_logits = logits.contiguous()
            shift_labels = labels.contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if audio_labels is not None:
            au_mask = (audio_labels >= 0).all(dim=-1)
            au_hidden_states = hidden_states[au_mask]
            au_audio_labels = audio_labels[au_mask]
            if len(au_hidden_states) <= 0:
                au_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
                au_audio_labels = torch.zeros_like(audio_labels).reshape(-1, self.config.num_codebooks)
                loss_weight = 0.0
            else:
                loss_weight = 1.0
            au_logits = self.audio_transformer(au_hidden_states, au_audio_labels)
            # We shift tokens and labels in dataloader
            shift_au_logits = au_logits.contiguous()
            shift_audio_labels = au_audio_labels.contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_au_logits = shift_au_logits.view(-1, self.config.codebook_size)
            shift_audio_labels = shift_audio_labels.view(-1)
            # Enable model parallelism
            shift_audio_labels = shift_audio_labels.to(shift_au_logits.device)
            au_loss = loss_fct(shift_au_logits, shift_audio_labels)

            loss += au_loss * loss_weight
        else:
            # au_tokens = self.audio_transformer.inference(hidden_states)
            pass

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VoilaOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _prepare_inputs_for_generation(
        self, input_ids, ref_embs=None, ref_embs_mask=None, audio_datas=None, audio_data_masks=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_cache_shape()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is None and \
                (past_key_values is None or past_key_values.get_seq_length() <= 0):
            inputs_embeds = self.model.embed_tokens(input_ids)
        if inputs_embeds is not None and \
                (past_key_values is None or past_key_values.get_seq_length() <= 0):
            model_inputs = {"inputs_embeds": inputs_embeds, "ref_embs": ref_embs, "ref_embs_mask": ref_embs_mask, "audio_datas": audio_datas, "audio_data_masks": audio_data_masks}
        else:
            model_inputs = {"input_ids": input_ids, "ref_embs": None, "audio_datas": None, "audio_data_masks": None}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        num_new_token: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], num_new_token))], dim=-1
            )

        return model_kwargs

    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[Union[int, List[int]]],
    ) -> torch.LongTensor:
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id not in eos_token_id)

        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return inputs.ne(pad_token_id).long()
        else:
            return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)

    @torch.inference_mode()
    def run_generate(
        self,
        input_ids: torch.LongTensor,
        ref_embs: Optional[List[torch.Tensor]] = None,
        ref_embs_mask: Optional[torch.LongTensor] = None,
        audio_datas: Optional[torch.FloatTensor] = None,
        audio_data_masks: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = 128,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        streamer: Optional["BaseStreamer"] = None,
        llm_audio_token_id: Optional[int] = None,
        min_audio_token_id: Optional[int] = None,
        temperature=0.2,
        top_k=50,
        audio_temperature=0.2,
        audio_top_k=50,
    ):
        assert eos_token_id is not None and pad_token_id is not None, "eos_token_id and pad_token_id are required for inference"
        assert llm_audio_token_id is not None and min_audio_token_id is not None, "llm_audio_token_id and min_audio_token_id are required for inference"
        assert len(input_ids.shape) == 2 or len(input_ids.shape) == 3, f"input_ids is supposed to be [batch, seq_len] or [batch, seq_len, num_codebooks], and got {input_ids.shape}"

        eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # Extend input_ids with additional num_codebooks dim
        if len(input_ids.shape) == 2:
            input_ids = input_ids[:, :, None].expand(1, 1, self.config.num_codebooks)

        this_peer_finished = False  # used by synced_gpus only
        max_length = input_ids.shape[1] + max_new_tokens

        model_kwargs = {
            "use_cache": True,
            "past_key_values": DynamicCache(),
            "attention_mask": self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            ),
        }
        # auto-regressive generation
        while True:
            # prepare model inputs
            model_inputs = self._prepare_inputs_for_generation(
                input_ids,
                ref_embs=ref_embs,
                ref_embs_mask=ref_embs_mask,
                audio_datas=audio_datas,
                audio_data_masks=audio_data_masks,
                **model_kwargs
            )

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
            )
            audio_tokens = self.audio_transformer.inference(
                outputs.last_hidden_state,
                temperature=audio_temperature,
                top_k=audio_top_k,
            )
            audio_tokens = torch.stack(
                [
                    audio_tokens[:, :, ci] + min_audio_token_id + ci*self.config.codebook_size
                    for ci in range(self.config.num_codebooks)
                ],
                dim=2,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            # Apply temperature and top-k
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))  # Safety check
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

            # sample
            probs = nn.functional.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Append NUM_CODEBOOK text tokens or audio_tokens
            if len(next_tokens.shape) == 1:
                next_tokens = next_tokens[:, None, None].expand(-1, 1, self.config.num_codebooks)
            next_tokens = torch.where(next_tokens==llm_audio_token_id, audio_tokens, next_tokens)

            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens[:, :, 0].ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=1)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if input_ids.shape[1] >= max_length:
                this_peer_finished = True

            if this_peer_finished:
                break

        if streamer is not None:
            streamer.end()

        return input_ids


# Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1103
class VoilaAutonomousModel(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pad_vocab_size_multiple = 64

        self.ref_emb_linear = nn.Linear(256, config.hidden_size, bias=True)
        self.audio_transformer = AudioTransformer(config, use_sdpa=False)
        self.voila_predictor = nn.Sequential(nn.Linear(config.hidden_size, 2, bias=True),)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        audio_labels: Optional[torch.LongTensor] = None,
        voila_labels: Optional[torch.LongTensor] = None,
        ref_embs: Optional[List[torch.Tensor]] = None,
        ref_embs_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, VoilaOutput]:
        r"""
        Args:
            input_ids: [bs, seq_len, num_codebooks]
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        assert len(inputs_embeds.shape) == 4
        if len(inputs_embeds.shape) == 4:
            inputs_embeds = inputs_embeds.mean(dim=2)

        if self.training or \
                (past_key_values is None and ref_embs is not None) or \
                (past_key_values is not None and past_key_values.get_seq_length() < 4 and ref_embs is not None):
            ref_embs = self.ref_emb_linear(ref_embs.to(self.ref_emb_linear.weight.dtype))
            ref_embs = ref_embs * ref_embs_mask.unsqueeze(-1).unsqueeze(-1)
            # (padding_left,padding_right,padding_top,padding_bottom,padding_front,padding_back)
            padding = (0, 0, 4, inputs_embeds.shape[1] - 5, 0, 0)
            ref_embs = torch.nn.functional.pad(ref_embs, padding, mode='constant', value=0.0)
            inputs_embeds = inputs_embeds + ref_embs

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        # calc voila_predict_loss
        voila_pred = self.voila_predictor(hidden_states)
        voila_pred = voila_pred.float()

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VoilaOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            voila_pred=voila_pred,
        )

    def _prepare_inputs_for_generation(
        self, input_ids, ref_embs=None, ref_embs_mask=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_cache_shape()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is None and \
                (past_key_values is None or past_key_values.get_seq_length() <= 0):
            inputs_embeds = self.model.embed_tokens(input_ids)
        if inputs_embeds is not None and \
                (past_key_values is None or past_key_values.get_seq_length() <= 0):
            model_inputs = {"inputs_embeds": inputs_embeds, "ref_embs": ref_embs, "ref_embs_mask": ref_embs_mask}
        else:
            model_inputs = {"input_ids": input_ids, "ref_embs": None}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        num_new_token: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], num_new_token))], dim=-1
            )

        return model_kwargs

    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[Union[int, List[int]]],
    ) -> torch.LongTensor:
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id not in eos_token_id)

        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return inputs.ne(pad_token_id).long()
        else:
            return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)

    @torch.inference_mode()
    def run_generate(
        self,
        input_ids: torch.LongTensor,
        input_generator,
        ref_embs: Optional[List[torch.Tensor]] = None,
        ref_embs_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: Optional[int] = 128,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        streamer: Optional["BaseStreamer"] = None,
        llm_audio_token_id: Optional[int] = None,
        min_audio_token_id: Optional[int] = None,
        llm_assistant_token_id: Optional[int] = None,
        temperature=0.2,
        top_k=50,
        audio_temperature=0.8,
        audio_top_k=50,
    ):
        assert eos_token_id is not None and pad_token_id is not None, "eos_token_id and pad_token_id are required for inference"
        assert llm_audio_token_id is not None and min_audio_token_id is not None, "llm_audio_token_id and min_audio_token_id are required for inference"
        assert len(input_ids.shape) == 2 or len(input_ids.shape) == 3, f"input_ids is supposed to be [batch, seq_len] or [batch, seq_len, num_codebooks], and got {input_ids.shape}"

        eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # Extend input_ids with additional num_codebooks dim
        input_ids = input_ids.clone()
        if len(input_ids.shape) == 2:
            input_ids = input_ids[:, :, None].expand(1, 1, self.config.num_codebooks)

        this_peer_finished = False  # used by synced_gpus only
        max_length = input_ids.shape[1] + max_new_tokens

        model_kwargs = {
            "use_cache": True,
            "past_key_values": DynamicCache(),
            "attention_mask": self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            ),
        }
        speaking = False
        # auto-regressive generation
        while True:
            # prepare model inputs
            model_inputs = self._prepare_inputs_for_generation(
                input_ids,
                ref_embs=ref_embs,
                ref_embs_mask=ref_embs_mask,
                **model_kwargs
            )

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
            )
            audio_tokens = self.audio_transformer.inference(
                outputs.last_hidden_state,
                temperature=audio_temperature,
                top_k=audio_top_k,
            )
            audio_tokens = torch.stack(
                [
                    audio_tokens[:, :, ci] + min_audio_token_id + ci*self.config.codebook_size
                    for ci in range(self.config.num_codebooks)
                ],
                dim=2,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # voila head output
            voila_head_pred = outputs.voila_pred[:, -1, :]
            voila_head_pred = torch.argmax(voila_head_pred, dim=-1)
            voila_head_pred = voila_head_pred.cpu()[0].item()

            # pre-process distribution
            # Apply temperature and top-k
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))  # Safety check
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

            # sample
            probs = nn.functional.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # voila head pred == 1, use assistant token
            if voila_head_pred == 1 and not speaking:
                next_tokens[0] = llm_assistant_token_id
                speaking = True
            elif next_tokens[0] == eos_token_id:
                speaking = False

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Append NUM_CODEBOOK text tokens or audio_tokens
            if len(next_tokens.shape) == 1:
                next_tokens = next_tokens[:, None, None].expand(-1, 1, self.config.num_codebooks)
            audio_token_mask = next_tokens == llm_audio_token_id
            next_tokens = next_tokens * torch.logical_not(audio_token_mask) + audio_tokens * audio_token_mask

            if audio_token_mask[0, 0, 0].item():
                try:
                    new_input_tokens = next(input_generator)
                except:
                    this_peer_finished = True
                    break
                new_input_tokens = new_input_tokens[None,None,:]
            else:
                new_input_tokens = next_tokens
            new_input_tokens = torch.cat([new_input_tokens, next_tokens], dim=2)

            input_ids = torch.cat([input_ids, new_input_tokens], dim=1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs
            )

            # # if eos_token was found in one sentence, set sentence to finished
            # if eos_token_id_tensor is not None:
            #     unfinished_sequences = unfinished_sequences.mul(
            #         next_tokens[:, :, 0].ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=1)
            #     )

            #     # stop when each sentence is finished
            #     if unfinished_sequences.max() == 0:
            #         this_peer_finished = True

            # stop if we exceed the maximum length
            if input_ids.shape[1] >= max_length:
                this_peer_finished = True

            if this_peer_finished:
                break

        if streamer is not None:
            streamer.end()

        return input_ids
