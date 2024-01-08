import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import warnings
from typing import Optional, Tuple, Union, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, TextStreamer
from transformers.models.llama import modeling_llama

import json
from transformers import logging
logging.set_verbosity_error()

import sys
sys.path.append('../')
sys.path.append('../evals')

from evals.aug import LLaMAReRoPE


try:
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
except Exception:
    raise ModuleNotFoundError(
        "Please install FlashAttention first, e.g., with pip install flash-attn --no-build-isolation, Learn more at https://github.com/Dao-AILab/flash-attention#installation-and-features"
    )

try:
    from einops import rearrange
except Exception:
    raise ModuleNotFoundError("Please install einops first, e.g., with pip install einops")


try:
    from flash_attn.flash_attn_interface import (
        flash_attn_kvpacked_func, 
        flash_attn_varlen_kvpacked_func, 
    )
    from flash_attn.bert_padding import unpad_input, pad_input
    flash_attn_v2_installed = True
    print('>>>> Flash Attention installed')
except ImportError:
    flash_attn_v2_installed = False
    raise ImportError('Please install Flash Attention: `pip install flash-attn --no-build-isolation`')

try:
    from flash_attn.layers.rotary import apply_rotary_emb_func
    flash_rope_installed = True
    print('>>>> Flash RoPE installed')
except ImportError:
    flash_rope_installed = False
    raise ImportError('Please install RoPE kernels: `pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary`')



## original repeate kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

## flash repeate kv
@torch.jit.script
def repeat_kv_flash(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, _, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, :, None, :].expand(batch, slen, 2, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, 2, num_key_value_heads * n_rep, head_dim)


## original rope apply
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

## flash rope apply
def apply_rotary_pos_emb_flash(q, k, cos, sin, position_ids):
    q_embed = apply_rotary_emb_func(q, cos, sin, False, True) # inplace=True
    k_embed = apply_rotary_emb_func(k, cos, sin, False, True) # inplace=True
    return q_embed, k_embed 
    

## monkey patch cahce rope apply
def apply_rotary_pos_emb_cache(q, k, cos_sin, position_ids):
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    gather_indices = gather_indices.repeat(1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3])
    bsz = gather_indices.shape[0]
    cos, sin = (torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices) for x in cos_sin)
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    return q, k
    
    
## original llama attention forward
def forward_original(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
   
    
## flash llama attention forward
def forward_flash(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        ### get shape and cache config
        bsz, q_len, h_size = hidden_states.size()

        has_layer_past = past_key_value is not None

        if has_layer_past:
            past_kv = past_key_value[0]
            past_len = past_key_value[1]
        else:
            past_len = 0

        ### projection for qkv
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            q = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            q = torch.cat(q, dim=-1)

            k = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            k = torch.cat(k, dim=-1)

            v = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            v = torch.cat(v, dim=-1)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states) 

        # NOTE: do not transpose (seq_len, nh) here, 
        # cuz the shape requirement for flash_attn 
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        ### apply q, k with flash rope
        kv_seq_len = k.shape[1] + past_len
        cos, sin = self.rotary_emb(v.transpose(1,2), seq_len=kv_seq_len) # shape: [1, 1, seq_len, dim]
        cos, sin = cos.squeeze(1).squeeze(0), sin.squeeze(1).squeeze(0)  # shape: [seq_len, dim] => squeeze as shape requirement for flash_attn
        cos, sin = cos[:, :cos.shape[-1]//2], sin[:, :sin.shape[-1]//2] # shape: [seq_len, rotary_dim] => rotary_dim = head_dim / 2, as shape requirement for flash_attn
        q, k = apply_rotary_pos_emb_flash(q, k, cos, sin, position_ids)
        
        # q, k = apply_rotary_pos_emb(q.transpose(1,2), k.transpose(1,2), cos, sin, position_ids)
        # q, k = q.transpose(1,2), k.transpose(1,2)
        
        ### stack kv and repeat the heads
        kv = torch.stack([k, v], 2)
        kv = repeat_kv_flash(kv, self.num_key_value_groups)

        ### update kv cache 
        # and concatenate the past kvs with the current kv, for shape requirement 
        if has_layer_past:
            new_len = past_len+q.size(1)
            if new_len > past_kv.size(1):
                past_kv = torch.cat([
                    past_kv, 
                    torch.empty(bsz, 256, 2, kv.size(3), kv.size(4), dtype=kv.dtype, device=kv.device)
                ], 1)
            past_kv[:, past_len:new_len] = kv
            kv = past_kv[:, :new_len]
        else:
            past_kv = kv

        past_key_value = (past_kv, past_len+q.size(1)) if use_cache else None

        ### do flash-attention
        attention_mask = None
        if attention_mask is not None: # varlen, ignore padding tokens, efficient for large batch with many paddings
            unpadded_kv, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(kv, attention_mask)
            unpadded_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask[:, -q.size(1):])
            attn_outputs = flash_attn_varlen_kvpacked_func(
                unpadded_q, unpadded_kv, 
                cu_seqlens_q, cu_seqlens_k, 
                max_seqlen_q, max_seqlen_k,
                dropout_p=0.0, 
                causal=(not has_layer_past), 
                return_attn_probs=output_attentions
            )

            attn_output = attn_outputs[0] if output_attentions else attn_outputs
            attn_output = pad_input(
                attn_output, indices_q, bsz, max_seqlen_q
            ).reshape(bsz, q_len, h_size)
            attn_weights = attn_outputs[2] if output_attentions else None
        else: # no padding tokens, more efficient
            attn_outputs = flash_attn_kvpacked_func(
                q, kv, 
                dropout_p=0.0,
                causal=(not has_layer_past), 
                return_attn_probs=output_attentions
            )

            attn_output = attn_outputs[0] if output_attentions else attn_outputs
            attn_output = attn_output.reshape(bsz, q_len, h_size)
            attn_weights = attn_outputs[2] if output_attentions else None

        ### projection for output
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


## flash llama patch decoder mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # [bsz, seq_len]
    return attention_mask

## flash llama patch cached decoder mask
def _prepare_decoder_attention_mask_cache(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat((
            torch.full((input_shape[0], past_key_values_length), True, dtype=attention_mask.dtype, device=attention_mask.device),
            attention_mask
        ), dim=-1)

    if attention_mask is not None and torch.all(attention_mask):
        return None

    return attention_mask


def forward_flash_patch(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()
    if output_attentions:
        warnings.warn("Output attentions is not supported for patched `LlamaAttention`, returning `None` instead.")


    if self.pretraining_tp > 1: # added for latest(transformer-4.31.0) llama 2
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # Past Key value support
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None
    
    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states], dim=2)  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    key_padding_mask = attention_mask

    if key_padding_mask is None:
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        max_s = q_len
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device)
        attn_output = flash_attn_varlen_qkvpacked_func(qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True)
        attn_output = rearrange(attn_output, "(b s) ... -> b s ...", b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
        attn_output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        attn_output = rearrange(
            pad_input(rearrange(attn_output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len),
            "b s (h d) -> b s h d",
            h=nheads,
        )
    
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.pretraining_tp > 1: # added for latest(transformer-4.31.0) llama 2
        attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def forward_flash_monkey_patch(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    assert past_key_value is None, "past_key_value is not supported"

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]
    assert not output_attentions, "output_attentions is not supported"
    assert not use_cache, "use_cache is not supported"

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    key_padding_mask = attention_mask

    if key_padding_mask is None:
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        max_s = q_len
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        output = flash_attn_varlen_qkvpacked_func(
            qkv, cu_q_lens, max_s, 
            dropout_p=0.0, softmax_scale=None, causal=True
        )
        output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(
            x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
        )
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
            ),
            "b s (h d) -> b s h d",
            h=nheads,
        )
    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, None


def forward_flash_monkey_patch_cache(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "output_attentions is only for testing only when using flash attention, \
            because they are not guaranteed to be correct (wrong scaling), according to flash_atnn/flash_attn_interface.py"
        )

    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, 'num_key_value_heads', self.num_heads)
    
    ## projection on qkv 
    # and transpose to make the seq_len and head_dim to be the last two dims 
    # for applying RoPE and repeat_kv operations
    q, k, v = ( # q,k,v shape: (bs, nh, q_len, hd)
        torch.cat( # added for latest(transformer-4.31.0) llama 2
            [
                F.linear(hidden_states, weight_slice) 
                for weight_slice in op.weight.split(nh * self.head_dim // self.pretraining_tp, dim=0)
            ], dim=-1
            ).view(bsz, q_len, nh, self.head_dim).transpose(1,2) if self.pretraining_tp > 1 \
        else op(hidden_states).view(bsz, q_len, nh, self.head_dim).transpose(1,2) 
        for op, nh in (
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads)
        ) 
    ) 
    
    ## concat the kv seq_len with the kv cache
    kv_seq_len = k.shape[-2] + (
        past_key_value[0].shape[-2] if past_key_value is not None else 0
    )

    ## apply RoPE to q, k
    cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

    ## concat the kv with the kv cache if exists, with shape = (bs, nh, kv_seq_len, hd)
    k, v = [torch.cat([past_key_value[i], x], dim=-2) if past_key_value is not None else x 
            for i, x in enumerate((k, v))]
    
    ## update kv cache if use_cache=True
    past_key_value = (k, v) if use_cache else None
    
    ## repeat k/v heads if n_kv_heads < n_heads
    k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)

    ## transpose the qkv shape to be compatible with flash-attn
    # and stack kv together to apply faster flash-attention, 
    # with shape: q.shape = (bs, q_len, nh, hd), kv.shape = (bs, kv_seq_len, 2, nh, hd)
    q, kv = [x.transpose(1, i+2) for i, x in enumerate((q, torch.stack((k, v), dim=2)))]

    ## do flash-attention
    key_padding_mask = attention_mask # here the flexible attention mask is shut down, and the attention mask is the same with key_padding_mask
    if key_padding_mask is None: # without the padding mask
        outputs = flash_attn_kvpacked_func(
                q, kv, 
                dropout_p=0.0, softmax_scale=None, causal=True,
                return_attn_probs=output_attentions
            )
        attn_output = outputs[0] if output_attentions else outputs
        attn_weights = outputs[2] if output_attentions else None
        attn_output = attn_output.view(bsz, q_len, -1) # shape = (bs, q_len, nh*hd) 
    else: # with padding mask
        q, indices, cu_q_lens, max_q = unpad_input(q, key_padding_mask[:, -q_len:])
        kv, _, cu_k_lens, max_k = unpad_input(kv, key_padding_mask)
        outputs = flash_attn_varlen_kvpacked_func(
                q, kv,
                cu_q_lens, cu_k_lens, 
                max_q, max_k,
                dropout_p=0.0, softmax_scale=None, causal=True
            )
        output_unpad = outputs[0] if output_attentions else outputs
        attn_weights = outputs[2] if output_attentions else None
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim) # shape = (total_len, nh*hd)
        attn_output = pad_input(output_unpad, indices, bsz, q_len) # shape = (bs, q_len, nh*hd)
        
    ## projection on output
    attn_output = sum([ # added for latest(transformer-4.31.0) llama 2
        F.linear(output_slice, o_proj_slice) 
        for output_slice, o_proj_slice in zip(
            attn_output.split(self.hidden_size // self.pretraining_tp, dim=2),
            self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
        )
    ]) if self.pretraining_tp > 1 else self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value


model_dir = '/mnt/cfs_bj/big_model'
# model_name = 'models/openbuddy-llama2-13b-v8.1-fp16'
model_name = "finetune-checkpoint/ecllama2-7B-gemini-5e-5-84-fp16-87cpk-0809-400w-epoch1"

device = "cuda:0"


aug = "flash-rerope"
if aug == "flash": # FIXME: failed with attention_mask is not None, and even attention_mask is None, the output is unreasonable
    print("#"*25, " using flash llama ", "#"*25)
    modeling_llama.LlamaAttention.forward = forward_flash
elif aug == "flash-patch": # FIXME: failed with nonsenece came out, and cannot use_cache=True
    print("#"*25, " using flash llama with patch", "#"*25)
    modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    modeling_llama.LlamaAttention.forward = forward_flash_patch
elif aug == 'flash-monkey-patch': # FIXME: succeeded with reasonble output(the same as the original), but cannot use_cache=True
    print("#"*25, " using flash llama with monkey patch", "#"*25)
    modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    modeling_llama.LlamaAttention.forward = forward_flash_monkey_patch
elif aug == "flash-monkey-patch-cache": # NOTE: when updating flash-attn to 2.2.1, this one suceeded perfectly, even with use_cache=True!
                                        # And I enhanced it with the pretraining_tp mode and supoorting output_attentions 
    print("#"*25, " using flash llama with monkey patch plus with cache", "#"*25) 
    modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_cache
    modeling_llama.LlamaAttention.forward = forward_flash_monkey_patch_cache
elif aug == "rerope":
    print("#"*25, " using rerope llama", "#"*25) 
    LLaMAReRoPE.window = 512
    LLaMAReRoPE.training_length = 4096
    LLaMAReRoPE.logn_scaling = True
    modeling_llama.LlamaAttention = LLaMAReRoPE.LlamaReRoPEAttention
elif aug == 'flash-rerope':
    print("#"*25, " using flash rerope llama", "#"*25) 
    LLaMAReRoPE.window = 512
    LLaMAReRoPE.training_length = 4096
    LLaMAReRoPE.flash_rerope_inner_apply = False
    LLaMAReRoPE.logn_scaling = True
    modeling_llama.LlamaAttention = LLaMAReRoPE.LlamaFlashReRoPEAttention
elif aug == "": # original llama with no aug
    print("#"*25, " using original llama ", "#"*25)



tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name), 
                                          trust_remote_code=True, 
                                          use_fast=True)
model = AutoModelForCausalLM.from_pretrained(os.path.join(model_dir, model_name), 
                                             trust_remote_code=True, 
                                             device_map='auto',
                                             torch_dtype=torch.bfloat16)

# set something
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.unk_token
streamer = TextStreamer(tokenizer, skip_prompt=True)

print(f"\n => tokenizer: \n{tokenizer}\n")

print(f"\n => model: \n{model}\n")

input_text = input("Please input your prompt below: \n")
default_prompt = '我最近很累,请问有什么办法缓解呀'
if input_text == "": 
    input_text = default_prompt
elif input_text == "context_mfqa":
    sample_path = './data/contexts/context_mfqa.json'
    sample = json.load(open(sample_path))
    prompt = f"阅读以下文字并用中文简短回答：\n\n{sample['context']}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{sample['input']}"
    input_text = f"User: {prompt}\n\n Assistant: "
elif input_text == "context_vcsum":
    sample_path = './data/contexts/context_vcsum.json'
    sample = json.load(open(sample_path))
    prompt = "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：".format(**sample)
    input_text = f"User: {prompt}\n\n Assistant: "

inputs = tokenizer([input_text], return_tensors='pt').to(device)
context_length = inputs.input_ids.shape[-1]
print("context_length: ", context_length)

max_new_tokens = 100
output_ids = model.generate(
                        inputs.input_ids, 
                        do_sample=False,
                        num_beams=1,
                        streamer=streamer,
                        max_new_tokens=max_new_tokens, 
                        use_cache=False if aug in ["flash-patch", 
                                                   "flash-monkey-patch",
                                                   ] else True,
                        )
output_text = tokenizer.batch_decode(
    [output_ids[0][context_length:]], 
    skip_special_tokens=True,
    )[0]

print(f"\n => output text: \n{output_text}\n")
