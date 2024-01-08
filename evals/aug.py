############################       base libs import     ############################
import warnings
import importlib
import math
from functools import partial
from contextlib import contextmanager
from typing import Tuple, List, Dict, Union, Optional, Any

############################       own libs import     ############################
from utils import check_package_version, check_str_bool

############################       dev libs import     ############################

import numpy as np
import torch # pytorch >= 2.0.0
check_package_version(pacakge=torch, min_version="2.0.0")

import torch.nn as nn
import torch.nn.functional as F

import transformers
import transformers.models.llama.modeling_llama as modeling_llama
    
try:
    import flash_attn
    from flash_attn.flash_attn_interface import ( # flash-attn v.2.2.1
        flash_attn_func, 
        flash_attn_kvpacked_func,
        flash_attn_varlen_kvpacked_func
    )
    from flash_attn.bert_padding import unpad_input, pad_input
    
    check_package_version(pacakge=flash_attn, min_version="2.2.1")
except Exception:
    raise ModuleNotFoundError(
        "Please install FlashAttention first, " + 
        "e.g., with pip install flash-attn --no-build-isolation, " + 
        "Learn more at https://github.com/Dao-AILab/flash-attention#installation-and-features"
    )

try:
    from flash_attn.layers.rotary import apply_rotary_emb_torch
except ImportError:
    raise ImportError("Please install RoPE kernels: " + 
                      "`pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary`"
                      )


try:
    from einops import rearrange
except Exception:
    raise ModuleNotFoundError("Please install einops first, e.g., with pip install einops")

import flash_rerope

############################       Base aug module / manager class     ############################


class AugModule(object):
    """BaseAugModule to define the common interfaces for all aug modules."""
    
    def __init__(self, aug_method: str = '') -> None:
        self.aug_method = aug_method
    
    def aug(self):
        """Implement the aug steps here"""
        raise NotImplementedError("The self.aug method must be implemented.")


class AugManger(object):
    """Base Aug ContextManager Class""" 
    def __init__(self, aug_method: str = '', aug_params: dict = {}):
        self.aug_method = aug_method
        self.aug_params = aug_params
        
    def __enter__(self): 
        """Code to run when entering the context"""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Code to run when exiting the context"""
        pass
    
    
############################       LLaMA / Llama aug methods     ############################


class LLaMAFlashAttn(AugModule):
    """Aug Llama with FlashAttention to make it much more time/memory efficient"""
    
    SUPPORTED_IMPLS = ['vanilla-v2', # the implementation version of flash-attention, 'vanilla-v2'(default) means the vanilla implementation from DaoAILab/flash-attention v2.0.0, 'pytorch-v2' means the implementation of scaled-dot-product-attention from pytorch2.0
                       'pytorch-v2'
                    ]
    DEFAULT_IMPL = 'vanilla-v2'
    IMPL_KEY = 'flash_impl'
    
    class LlamaFlashAttention(modeling_llama.LlamaAttention): 
        def forward(
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
            q, k = modeling_llama.apply_rotary_pos_emb(q, k, cos, sin, position_ids)

            ## concat the kv with the kv cache if exists, with shape = (bs, nh, kv_seq_len, hd)
            k, v = [torch.cat([past_key_value[i], x], dim=-2) if past_key_value is not None else x 
                    for i, x in enumerate((k, v))]
            
            ## update kv cache if use_cache=True
            past_key_value = (k, v) if use_cache else None
            
            ## repeat k/v heads if n_kv_heads < n_heads
            k, v = [modeling_llama.repeat_kv(x, self.num_key_value_groups) for x in (k,v)]
            
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

    class LlamaTorchV2Attention(modeling_llama.LlamaAttention):
        def forward(
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
                    "output_attentions cannot be returned since the pytorchv2.0.0 implementation of scaled_dot_product_attention does not support it."
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
            q, k = modeling_llama.apply_rotary_pos_emb(q, k, cos, sin, position_ids)

            ## concat the kv with the kv cache if exists, with shape = (bs, nh, kv_seq_len, hd)
            k, v = [torch.cat([past_key_value[i], x], dim=-2) if past_key_value is not None else x 
                    for i, x in enumerate((k, v))]
            
            ## update kv cache if use_cache=True
            past_key_value = (k, v) if use_cache else None
            
            ## repeat k/v heads if n_kv_heads < n_heads
            k, v = [modeling_llama.repeat_kv(x, self.num_key_value_groups) for x in (k,v)]

            ## do pytorch2.0 scaled_dot_product_attetion ( flash-attention + xpos-memory-efficient attention )
            if attention_mask is None: # without attention mask
                attn_output = F.scaled_dot_product_attention(q, k, v, # shape = (bs, nh, q_len, hd)
                                                             is_causal=True, dropout_p=0.0)
            else: # with attention mask
                attn_output = F.scaled_dot_product_attention(q, k, v, # shape = (bs, nh, q_len, hd)
                                                             attn_mask=attention_mask, dropout_p=0.0)
                
            ## reshape the output to (bs, q_len, hidden_size)
            attn_output = attn_output.permute(0, 2, 1, 3) # shape = (bs, q_len, nh, hd)
            if not attn_output.is_contiguous(): attn_output = attn_output.contiguous()
            attn_output = attn_output.view(bsz, q_len, -1) # shape = (bs, q_len, nh*hd)
                
            ## projection on output
            attn_output = sum([ # added for latest(transformer-4.31.0) llama 2
                F.linear(output_slice, o_proj_slice) 
                for output_slice, o_proj_slice in zip(
                    attn_output.split(self.hidden_size // self.pretraining_tp, dim=2),
                    self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
                )
            ]) if self.pretraining_tp > 1 else self.o_proj(attn_output)

            return attn_output, None, past_key_value

    @staticmethod
    def _prepare_decoder_attention_mask_naive(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # [bsz, seq_len]
        return attention_mask

    @staticmethod
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # [bsz, seq_len]
        if past_key_values_length > 0 and attention_mask is not None:
            attention_mask = torch.cat((
                torch.full((input_shape[0], past_key_values_length), 
                           True, dtype=attention_mask.dtype, device=attention_mask.device),
                attention_mask
            ), dim=-1)

        if attention_mask is not None and torch.all(attention_mask):
            return None
        
        print(f"attention mask: \n{attention_mask}\n")

        return attention_mask 
        
    def __init__(self, aug_method: str = 'flash', **kwargs) -> None:
        super().__init__(aug_method)
        
        # check cuda capability
        cuda_major, _ = torch.cuda.get_device_capability()
        if cuda_major < 8:
            raise EnvironmentError(
                "Flash attention is only supported on Ampere or Hopper GPU during training due to head dim > 64 backward."
                "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
            )
            
        # choose one of the supported implementation
        if LLaMAFlashAttn.IMPL_KEY in kwargs and kwargs[LLaMAFlashAttn.IMPL_KEY] not in LLaMAFlashAttn.SUPPORTED_IMPLS: 
            raise KeyError(f"The impl {kwargs[LLaMAFlashAttn.IMPL_KEY]} is not supported")
        elif LLaMAFlashAttn.IMPL_KEY not in kwargs: impl = LLaMAFlashAttn.DEFAULT_IMPL
        else: impl = kwargs[LLaMAFlashAttn.IMPL_KEY]
            
        # set aug patches
        if self.aug_method == 'flash': # original llama flash attention patch
            if impl == 'vanilla-v2':
                self.llama_attention_class = LLaMAFlashAttn.LlamaFlashAttention
                self.decoder_attention_mask = LLaMAFlashAttn._prepare_decoder_attention_mask  
            elif impl == 'pytorch-v2':
                self.llama_attention_class = LLaMAFlashAttn.LlamaTorchV2Attention
                # NOTE: here the original llama decoder attention mask can be used, so no replacement is needed
                self.decoder_attention_mask = modeling_llama.LlamaModel._prepare_decoder_attention_mask
        else:
            raise KeyError(f"The aug method {self.aug_method} is not supported")

    def aug(self):
        modeling_llama.LlamaAttention = self.llama_attention_class
        modeling_llama.LlamaModel._prepare_decoder_attention_mask = self.decoder_attention_mask


class LLaMALogNScaleRoPE(AugModule):
    
    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        """original RoPE apply func"""
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (LLaMALogNScaleRoPE.rotate_half(q) * sin)
        k_embed = (k * cos) + (LLaMALogNScaleRoPE.rotate_half(k) * sin)
        return q_embed, k_embed
    
    @staticmethod
    def apply_rotary_pos_emb_and_logn_scale(q, k, cos, sin, position_ids, training_length):
        q_embed, k_embed = LLaMALogNScaleRoPE.apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        scale = ((position_ids + 1)[:, None, :, None].log() / np.log(training_length)).clip(1)
        return q_embed * scale.to(q_embed.dtype), k_embed
    
    def __init__(self, aug_method: str = 'logn', 
                training_length=2048,
                **kwargs) -> None:
        super().__init__(aug_method)
        
        self.training_length = training_length
        
        if aug_method == 'logn':
            self.rope_apply_func = partial(
                LLaMALogNScaleRoPE.apply_rotary_pos_emb_and_logn_scale,
                training_length=training_length
            )
        else:
            raise KeyError("The aug method {} is not supported".format(aug_method))
    
    def aug(self):
        modeling_llama.apply_rotary_pos_emb = self.rope_apply_func


class LLaMANTKRoPE(AugModule):
    """Aug Llama with MTK-based RoPE to enlarge the context window during inference"""

    class LlamaNTKRotaryEmbedding(nn.Module):
        """Llama NTK-RoPE Implementation"""
        def __init__(self, 
                     dim, 
                     aug_method, ratio, b, 
                     max_position_embeddings=2048, 
                     base=10000, 
                     device=None,
                     **kwargs
                     ):
            super().__init__()
            
            # max context window enlarged by ratio times
            if kwargs['dynamic']: ratio = 1 # dynamically set ratio, firtsly use the original one 
            
            self.ntk_ratio = ratio
            self.max_seq_len_cached = max_position_embeddings * ratio 
            self.kwargs = kwargs
            
            # keep the aug args to be reaug during forward
            self.aug_kwargs = dict( 
                aug_method=aug_method,
                d=dim, k=ratio, b=b, base=base, device=device
            )

            # ntk aug inv freq
            inv_freq = LLaMANTKRoPE._ntk_aug(**self.aug_kwargs)
            # register inv freq
            self._register_inv_freq(inv_freq=inv_freq)
        
            # get cos/sin cached
            cos_cached, sin_cached = LLaMANTKRoPE._get_cos_sin_cached(
                                            seq_len=self.max_seq_len_cached, 
                                            inv_freq=inv_freq,
                                            device=self.inv_freq.device,
                                    )
            # register cos/sin cache
            self._register_cos_sin_cache(cos_cached=cos_cached, sin_cached=sin_cached)
            
        def forward(self, x, seq_len=None):
            # x: [bs, num_attention_heads, seq_len, head_size]
            if seq_len > self.max_seq_len_cached: 
                # give the too long seq a temperary embedding
                # and if in dynamic mode, register the new embedding cache and ntk ratio
                 return self._get_new_cos_sin_cache(x, seq_len)
        
            return (
                self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            )
            
        def _get_new_cos_sin_cache(self, x, seq_len):
            """
            Deal with the input of too long seq_len.
            """
            # recompute the reasonable ntk ratio
            new_ntk_ratio, new_max_seq_len_cached = self._recompute_ntk_ratio(seq_len) 
            
            # reaug inv freq
            new_aug_kwargs = dict(self.aug_kwargs)
            new_aug_kwargs['k'] = new_ntk_ratio
            new_aug_kwargs['device'] = x.device
            new_inv_freq = LLaMANTKRoPE._ntk_aug(**new_aug_kwargs)
            
            # reget cos/sin cache 
            new_cos_cached, new_sin_cached = LLaMANTKRoPE._get_cos_sin_cached(
                                                    seq_len=new_max_seq_len_cached, 
                                                    inv_freq=new_inv_freq,
                                                    device=x.device
                                            )  
            
            # if in dynamic mode, dynamically set the new properties
            if self.kwargs['dynamic']: 
                self.ntk_ratio = new_ntk_ratio
                self.max_seq_len_cached = new_max_seq_len_cached
                self.aug_kwargs = new_aug_kwargs
                # also register new inv freq
                self._register_inv_freq(inv_freq=new_inv_freq)
                # also register new cos/sin cache
                self._register_cos_sin_cache(cos_cached=new_cos_cached, sin_cached=new_sin_cached)
            
            return (
                new_cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                new_sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            )

        def _recompute_ntk_ratio(self, new_seq_len):
            max_position_embeddings = self.max_seq_len_cached // self.ntk_ratio
            new_ntk_ratio = int(np.ceil(new_seq_len / max_position_embeddings))
            if new_ntk_ratio % 2 == 1: # to keep ntk ratio even
                new_ntk_ratio += 1
            new_max_seq_len_cached = max_position_embeddings * new_ntk_ratio 
            
            return new_ntk_ratio, new_max_seq_len_cached

        def _register_inv_freq(self, inv_freq):
            assert inv_freq.dtype in (torch.float16, torch.float32), "inv_freq should be float16 or float32"
            self.register_buffer("inv_freq", inv_freq)
            
        def _register_cos_sin_cache(self, cos_cached, sin_cached, dtype=None):
            if dtype is None: dtype = torch.get_default_dtype()
                
            self.register_buffer("cos_cached", cos_cached.to(dtype=dtype), persistent=False)
            self.register_buffer("sin_cached", sin_cached.to(dtype=dtype), persistent=False)

    @staticmethod
    def _get_cos_sin_cached(seq_len, inv_freq, device, dtype=torch.float32):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached, sin_cached = emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
        
        return cos_cached, sin_cached
    
    
    @staticmethod
    def _ntk_aug(aug_method, d, k, b, base, device, dtype=torch.float32):
        # aug inv freq
        if aug_method == 'ntk':
            base *= k ** (d / (d-2)) # base change formula
            inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).to(device=device, dtype=dtype) / d))
        elif aug_method == 'ntk-fix':
            base *= k ** (d / (d-2)) # base change formula 
            lamda = k ** (2 / (d-2)) # the solution to ntk
            inv_freq = 1.0 / ((base ** (torch.arange(0, d, 2).to(device=device, dtype=dtype) / d)) * lamda) # NOTE: the core of NTK-RoPE-Fix
        elif aug_method == 'ntk-mix':
            a = np.log(k) / (d / 2)**b
            inv_freq = base**(-torch.arange(0, d, 2).float().to(device=device, dtype=dtype) / d)
            inv_freq *= (-a * torch.arange(1, d // 2 + 1).float().to(device=device, dtype=dtype)**b).exp() # NOTE: the core of NTK-RoPE-mix
        else:
            raise NotImplementedError(f"The aug method {aug_method} is not supported yet")
        
        return inv_freq


    def __init__(self, aug_method='ntk', 
                 ntk_ratio=8, 
                 ntk_mix_b=0.625, 
                 **kwargs) -> None:
        super().__init__(aug_method)
        
        self.ratio = ntk_ratio
        self.b = ntk_mix_b
        self.kwargs = kwargs
        
        # get the bool params
        self.kwargs['logn'] = check_str_bool(kwargs['logn'])
        self.kwargs['dynamic'] = check_str_bool(kwargs['dynamic'])
        self.kwargs['flash'] = check_str_bool(kwargs['flash'])
        
        self.llama_embedding_class = partial( # freeze the ratio argument
            LLaMANTKRoPE.LlamaNTKRotaryEmbedding, 
            aug_method=self.aug_method,
            ratio=self.ratio,
            b=self.b, # NTK-mix's arg
            **self.kwargs,
        )    
        
        
    def aug(self):
        modeling_llama.LlamaRotaryEmbedding = self.llama_embedding_class
        # use logn scale rope apply func
        if self.kwargs['logn']: LLaMALogNScaleRoPE(**self.kwargs).aug()
        # use flash attention func
        if self.kwargs['flash']: LLaMAFlashAttn(**self.kwargs).aug()


class LLaMAReRoPE(AugModule):
    """Aug Llama with ReRoPE to enlarge the context window during inference
    """
    
    window = 512
    training_length = 4096
    scaling_factor = 16
    logn_scaling = False
    
    class LlamaReRoPEAttention(modeling_llama.LlamaAttention):
        """Llama RePoPE Implementation
        Copied from Jianlin Su
            Citation:
            @misc{rerope2023,
                title={Rectified Rotary Position Embeddings},
                author={Jianlin Su},
                year={2023},
                howpublished={url{https://github.com/bojone/rerope}},
            }
        """
        def forward(
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
            
            if LLaMAReRoPE.logn_scaling:
                query_states *= ((position_ids + 1)[:, None, :, None].log() / np.log(LLaMAReRoPE.training_length)).clip(1).to(query_states.dtype)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
                position_ids = torch.cat([past_key_value[2], position_ids], dim=1)

            past_key_value = (key_states, value_states, position_ids) if use_cache else None
            
            if q_len == 1:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                position_ids = (position_ids[:, -1] - position_ids).clip(max=LLaMAReRoPE.window)
                _, key_states = LLaMAReRoPE.apply_rotary_pos_emb(None, key_states, cos, -sin, position_ids)
                key_states = modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
                value_states = modeling_llama.repeat_kv(value_states, self.num_key_value_groups)
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            else:
                cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, LLaMAReRoPE.window + 1))
                query_states1, key_states1 = LLaMAReRoPE.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
                query_states2, _ = LLaMAReRoPE.apply_rotary_pos_emb(query_states, None, cos, sin, position_ids * 0 + LLaMAReRoPE.window)

                # repeat k/v heads if n_kv_heads < n_heads
                key_states1 = modeling_llama.repeat_kv(key_states1, self.num_key_value_groups)
                key_states2 = modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
                value_states = modeling_llama.repeat_kv(value_states, self.num_key_value_groups)

                attn_weights1 = torch.matmul(query_states1, key_states1.transpose(2, 3)) / math.sqrt(self.head_dim)
                attn_weights2 = torch.matmul(query_states2, key_states2.transpose(2, 3)) / math.sqrt(self.head_dim)
                rectified_mask = (position_ids[:, -q_len:, None] - position_ids[:, None]).abs() < LLaMAReRoPE.window
                attn_weights = torch.where(rectified_mask, attn_weights1, attn_weights2)

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
    
    class LlamaLeakyReRoPEAttention(modeling_llama.LlamaAttention):
        """Llama Leaky-ReRoPE Implementation
        Copied from Jianlin Su
            Citation:
            @misc{rerope2023,
                title={Rectified Rotary Position Embeddings},
                author={Jianlin Su},
                year={2023},
                howpublished={url{https://github.com/bojone/rerope}},
            }
        """
        def _init_rope(self):
            self.rotary_emb = modeling_llama.LlamaRotaryEmbedding(
                self.head_dim, 
                max_position_embeddings=self.max_position_embeddings)
            self.rotary_emb2 = modeling_llama.LlamaLinearScalingRotaryEmbedding(
                self.head_dim, 
                max_position_embeddings=self.max_position_embeddings, 
                scaling_factor=LLaMAReRoPE.scaling_factor)
        
        def forward(
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
            
            if LLaMAReRoPE.logn_scaling:
                query_states *= ((position_ids + 1)[:, None, :, None].log() / np.log(LLaMAReRoPE.training_length)).clip(1).to(query_states.dtype)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
                position_ids = torch.cat([past_key_value[2], position_ids], dim=1)

            past_key_value = (key_states, value_states, position_ids) if use_cache else None
            
            offset = LLaMAReRoPE.window * (LLaMAReRoPE.scaling_factor - 1)
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            cos2, sin2 = self.rotary_emb2(value_states, seq_len=kv_seq_len + offset)
            if q_len == 1:
                position_ids = position_ids[:, -1:] - position_ids
                cos = torch.cat([cos[:, :, :LLaMAReRoPE.window], cos2[:, :, LLaMAReRoPE.window + offset:]], axis=2)
                sin = torch.cat([sin[:, :, :LLaMAReRoPE.window], sin2[:, :, LLaMAReRoPE.window + offset:]], axis=2)
                _, key_states = LLaMAReRoPE.apply_rotary_pos_emb(None, key_states, cos, -sin, position_ids)
                key_states = modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
                value_states = modeling_llama.repeat_kv(value_states, self.num_key_value_groups)
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            else:
                query_states1, key_states1 = LLaMAReRoPE.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
                query_states2, _ = LLaMAReRoPE.apply_rotary_pos_emb(query_states, None, cos2, sin2, position_ids + offset)
                _, key_states2 = LLaMAReRoPE.apply_rotary_pos_emb(None, key_states, cos2, sin2, position_ids)

                # repeat k/v heads if n_kv_heads < n_heads
                key_states1 = modeling_llama.repeat_kv(key_states1, self.num_key_value_groups)
                key_states2 = modeling_llama.repeat_kv(key_states2, self.num_key_value_groups)
                value_states = modeling_llama.repeat_kv(value_states, self.num_key_value_groups)

                attn_weights1 = torch.matmul(query_states1, key_states1.transpose(2, 3)) / math.sqrt(self.head_dim)
                attn_weights2 = torch.matmul(query_states2, key_states2.transpose(2, 3)) / math.sqrt(self.head_dim)
                rectified_mask = (position_ids[:, -q_len:, None] - position_ids[:, None]).abs() < LLaMAReRoPE.window
                attn_weights = torch.where(rectified_mask, attn_weights1, attn_weights2)

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
    
    class LlamaFlashReRoPEAttention(modeling_llama.LlamaAttention):
        """Llama Flash RePoPE Implementation
        For ReRoPE algorithm:
            Copied from Jianlin Su
                Citation:
                @misc{rerope2023,
                    title={Rectified Rotary Position Embeddings},
                    author={Jianlin Su},
                    year={2023},
                    howpublished={url{https://github.com/bojone/rerope}},
                }
        For Flash fused algorithm:
            Utilizing Triton library
        """
        def forward(
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
            
            if LLaMAReRoPE.logn_scaling:
                query_states *= ((position_ids + 1)[:, None, :, None].log() / np.log(LLaMAReRoPE.training_length)).clip(1).to(query_states.dtype)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
                position_ids = torch.cat([past_key_value[2], position_ids], dim=1)

            past_key_value = (key_states, value_states, position_ids) if use_cache else None
            
            if q_len == 1:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                position_ids = (position_ids[:, -1] - position_ids).clip(max=LLaMAReRoPE.window)
                _, key_states = LLaMAReRoPE.apply_rotary_pos_emb(None, key_states, cos, -sin, position_ids)
                key_states = modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
                value_states = modeling_llama.repeat_kv(value_states, self.num_key_value_groups)
                
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    attn_mask = attention_mask, is_causal = attention_mask is None,
                ).transpose(1, 2).contiguous()
            else:
                cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, LLaMAReRoPE.window + 1))
                key_states = modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
                value_states = modeling_llama.repeat_kv(value_states, self.num_key_value_groups)
                
                attn_output = flash_rerope.flash_attn_func_with_fused_rerope(
                    query_states.transpose(1,2), key_states.transpose(1,2), value_states.transpose(1,2), # [bs, seq_len, nh, hd]
                    cos.squeeze(1).squeeze(0), sin.squeeze(1).squeeze(0),  # [seq_len, dim]
                    position_ids, LLaMAReRoPE.window, bias=None, 
                    causal=True, softmax_scale=1/math.sqrt(self.head_dim),
                    inner=LLaMAReRoPE.flash_rerope_inner_apply,
                )

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
    
        
    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos[:, :, -q.shape[2]:]) + (modeling_llama.rotate_half(q) * sin[:, :, -q.shape[2]:]) if q is not None else None
        k_embed = (k * cos) + (modeling_llama.rotate_half(k) * sin) if k is not None else None
        return q_embed, k_embed
    
    def __init__(self, aug_method='rerope', 
                 rerope_window=512, 
                 rerope_length=4096, 
                 rerope_scaling_factor=16, 
                 **kwargs)-> None:
        """
        window: the position window size of the relu linear part
        training_length: the training max context length
        scaling_factor: the scaling factor for context (the same as ntk ratio)
        """
        super().__init__(aug_method)
        
        LLaMAReRoPE.window = self.window = rerope_window
        LLaMAReRoPE.training_length = self.training_length = rerope_length
        LLaMAReRoPE.scaling_factor = self.scaling_factor = rerope_scaling_factor
        LLaMAReRoPE.logn_scaling = check_str_bool(kwargs['logn'])
        LLaMAReRoPE.flash = check_str_bool(kwargs['flash'])
        LLaMAReRoPE.flash_rerope_inner_apply = check_str_bool(kwargs['flash_rerope_inner_apply'])
        
        self.kwargs = kwargs
        
        if self.aug_method == 'rerope':
            if LLaMAReRoPE.flash:
                self.llama_attention_class = LLaMAReRoPE.LlamaFlashReRoPEAttention
            else:
                self.llama_attention_class = LLaMAReRoPE.LlamaReRoPEAttention
        elif self.aug_method == 'rerope-leaky':
            if LLaMAReRoPE.flash:
                raise NotImplementedError(f"The flash attention for aug method {self.aug_method} is not suppoerted")
            else:
                self.llama_attention_class = LLaMAReRoPE.LlamaLeakyReRoPEAttention
        else:
            raise NotImplementedError(f"The aug method {self.aug_method} in ReRoPE is not suppoerted")
        
    def aug(self):
        modeling_llama.LlamaAttention = self.llama_attention_class
    
    
class LLaMAAugManager(AugManger):
    """Llama Aug ContextManager Class"""
    
    SUPPORT_METHODS = {
            'ntk': dict(
                name='NTK-RoPE',
                args=dict(
                    logn=False, # whether to apply the logn scaling 
                    training_length=2048, # the logn scaling param
                    flash=False, # whether to use flash attn to speed up and save memory, if used, the flash aug args can be also specified in the config
                    flash_impl = 'vanilla-v2', # the implementation version of flash-attention, 'vanilla-v2'(default) means the vanilla implementation from DaoAILab/flash-attention v2.0.0, 'pytorch-v2' means the implementation of scaled-dot-product-attention from pytorch2.0
                    
                    dynamic=False, # when dynamic is set to 'true', the ntk ratio won't be applied
                    ntk_ratio=8, # the ntk ratio to make the context incremented by ratio times, default 8
                )
            ),
            'ntk-fix': dict(
                name='NTK-RoPE-fix',
                args=dict(
                    logn=False, # whether to apply the logn scaling 
                    training_length=2048, # the logn scaling param
                    flash=False, # whether to use flash attn to speed up and save memory
                    flash_impl = 'vanilla-v2', # the implementation version of flash-attention, 'vanilla-v2'(default) means the vanilla implementation from DaoAILab/flash-attention v2.0.0, 'pytorch-v2' means the implementation of scaled-dot-product-attention from pytorch2.0
                    
                    dynamic=False, # when dynamic is set to 'true', the ntk ratio won't be applied
                    ntk_ratio=8, # the ntk ratio to make the context incremented by ratio times, default 8
                ) 
            ),
            'ntk-mix': dict(
                name='NTK-RoPE-mix',
                args=dict(
                    logn=False, # whether to apply the logn scaling 
                    training_length=2048, # the logn scaling param
                    flash=False, # whether to use flash attn to speed up and save memory
                    flash_impl = 'vanilla-v2', # the implementation version of flash-attention, 'vanilla-v2'(default) means the vanilla implementation from DaoAILab/flash-attention v2.0.0, 'pytorch-v2' means the implementation of scaled-dot-product-attention from pytorch2.0
                    
                    dynamic=False, # when dynamic is set to 'true', the ntk ratio won't be applied
                    ntk_ratio=8, # the ntk ratio to make the context incremented by ratio times, default 8
                    ntk_mix_b=0.625,  # the ntk-rope-mix hyper-param for the exp solution to the lambda function
                )
            ),
            'rerope': dict(
                name='ReRoPE',
                args=dict(
                    logn=False, # the training_length param is the same as the rerope_length
                    flash=False, # whether to use flash attn to speed up and save memory
                    flash_rerope_inner_apply=False, # whether to apply the rerope to q,k inside the kernel, default False.
                    
                    rerope_window=512, # rerope_window: the window size for the rerope design, default 512
                    rerope_length=4096, # rerope_length: the training length of the LLaMA model, e.g. 2048 for LLaMA, and 4096 for LLaMA2
                )
            ),
            'rerope-leaky': dict(
                name='Leaky-ReRoPE',
                args=dict(
                    logn=False, # the training_length param is the same as the rerope_length
                    
                    rerope_window=512, # rerope_window: the window size for the rerope design, default 512
                    rerope_length=4096, # rerope_length: the training length of the LLaMA model, e.g. 2048 for LLaMA, and 4096 for LLaMA2
                    rerope_scaling_factor=8, # rerope_scaling_factor: the scaling times for the context window, the same as ntk_ratio in ntk methods, used only rerope-leaky aug method, default 8
                )
            ),
            'flash': dict( # only use flash attn to speed up and save memory, without any lcw-aug method plugged in
                name='Flash-Attn',
                args=dict(
                    flash_impl = 'vanilla-v2', # the implementation version of flash-attention, 'vanilla-v2'(default) means the vanilla implementation from DaoAILab/flash-attention v2.0.0, 'pytorch-v2' means the implementation of scaled-dot-product-attention from pytorch2.0
                )
                    
            ),
            'logn': dict(
                name='LogN-Scaling',
                args=dict(
                    training_length=2048, # the training length of the model, to scale the q value by max( 1, log_{training_length}n ), where n is the seq length
                )
            )
        }
    
    def __init__(self, aug_method: str = '', aug_params: dict = {}):
        super().__init__(aug_method, aug_params)
        
    def __enter__(self): 
        """Modify the original llama implementation to aug for long context window
        including: RotaryEmbedding, Attention, etc
        """
        if self.aug_method == '':  # no aug method used
            pass
        else: # select the aug method
            if 'flash' in self.aug_method: # aug only with flash attention
                aug_module = LLaMAFlashAttn(aug_method=self.aug_method, **self.aug_params)
            elif 'logn' in self.aug_method: # aug only with logn scaling
                aug_module = LLaMALogNScaleRoPE(aug_method=self.aug_method, **self.aug_params)
            elif 'ntk' in self.aug_method:
                # replace the LLaMA RoPE class with NTK-RoPE
                aug_module = LLaMANTKRoPE(aug_method=self.aug_method, **self.aug_params)
            elif 'rerope' in self.aug_method:
                # replace the LLaMA RoPE class with ReRoPE
                aug_module = LLaMAReRoPE(aug_method=self.aug_method, **self.aug_params)
            else:
                raise KeyError(f"The aug method {self.aug_method} is not supported")  
            
            aug_module.aug()

        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Reload original llama implementation to avoid keep the aug methods for next model"""
        if self.aug_method != '':
            importlib.reload(transformers.models.llama.modeling_llama)
    
    
############################       ChatGLM aug methods     ############################