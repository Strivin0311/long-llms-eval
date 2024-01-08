import sys
import os
import importlib
from typing import Optional
import time

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, TextStreamer
from torchsummary import summary
from accelerate import init_empty_weights, infer_auto_device_map

# modeling_chatglm path: /mnt/cfs_bj/big_model/models/chatglm2-6b-32k/modeling_chatglm.py


def auto_configure_device_map(num_gpus: int) -> dict:
    """Copied from official ChatGLM(2)-6B repo"""
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    
    # 本文件来源于https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    # 仅此处做少许修改以支持ChatGLM2
    
    ## chatglm2
    device_map = {
        'transformer.embedding.word_embeddings': 0,
        'transformer.word_embeddings.weight': 0, 
        'transformer.encoder.final_layernorm': 0,
        'transformer.final_layernorm.weight': 0, 
        'transformer.final_layernorm.bias': 0,
        'transformer.output_layer': 0,
        'transformer.rotary_pos_emb': 0,
        'lm_head': 0
    }
    
    ## chatglm 1
    # device_map = {
    #     'transformer.word_embeddings': 0,
    #     'transformer.final_layernorm': 0, 
    #     'lm_head': 0,
    # }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.encoder.layers.{i}'] = gpu_target
        used += 1

    return device_map

def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None
        if self.training:
            use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                # token by token decoding, use tuple format
                if kv_caches[0] is not None:
                    presents = presents + (kv_cache,)
                # prefilling in decoding, use tensor format to save cuda memory
                else:
                    if len(presents) == 0:
                        presents = kv_cache
                    else:
                        # NOTE: change here to support multi-gpus for chatglm2-32k 
                        # by moving kv_cache(smaller tensor) onto presents(larger tensor)' device
                        if presents.device != kv_cache.device: 
                            kv_cache = kv_cache.to(presents.device)

                        presents = torch.cat((presents, kv_cache), dim=0)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions

model_dir = '/mnt/cfs_bj/big_model/models/'
# model_name = 'chatglm2-6b'
model_name = 'chatglm2-6b-32k'
device = "cuda:0"

config = AutoConfig.from_pretrained(os.path.join(model_dir, model_name), trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name), trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True)

method = 0
load_chatglm2_32k_method = [
    'single_gpu',
    'multi_gpus_auto_device_map',
    'multi_gpus_manual_device_map',
    'multi_gpus_semi_auto_device_map',
][method]


if load_chatglm2_32k_method == 'single_gpu':
    model = AutoModel.from_pretrained(os.path.join(model_dir, model_name), trust_remote_code=True).half().to(device)
else:
    modeling_chatglm = importlib.import_module('transformers_modules.chatglm2-6b-32k.modeling_chatglm')
    modeling_chatglm.GLMTransformer.forward = forward # change forward to support multi-gpus
    
    if load_chatglm2_32k_method == 'mult_gpus_auto_device_map':
        model = AutoModel.from_pretrained(os.path.join(model_dir, model_name), 
                                    trust_remote_code=True, 
                                    device_map='auto').half()
    elif load_chatglm2_32k_method == 'mult_gpus_manual_device_map':
        num_gpus = 8
        device_map = auto_configure_device_map(num_gpus)
        model = AutoModel.from_pretrained(os.path.join(model_dir, model_name), 
                                                trust_remote_code=True, 
                                                device_map=device_map
                                                ).half()
    elif load_chatglm2_32k_method =='mult_gpus_semi_auto_device_map':
        num_gpus = 8
        max_memory = {i: '16GiB' for i in range(num_gpus)}
        max_memory[0] = '10GiB'
        with init_empty_weights():
            model = AutoModel.from_config(config)
            device_map = infer_auto_device_map(model, no_split_module_classes=["GLMBlock"], 
                                               dtype=torch.float16, 
                                               max_memory=max_memory
                                               )
        model = AutoModel.from_pretrained(os.path.join(model_dir, model_name), 
                                                trust_remote_code=True, 
                                                device_map=device_map
                                                ).half()
    


mems = {
        f"cuda{i}": f"{torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB"
        for i in range(torch.cuda.device_count())
}

print(f"\n => memory usage: \n{mems}\n")

print(f"\n => tokenizer: \n{tokenizer}\n")

print(f"\n => model: \n{model}\n")

if vars(model).get('hf_device_map', None):
    print(f"\n => model's device map: \n{model.hf_device_map}\n")
else:
    print(f"\n => model's device map: \n{device}\n")

print(f"\n => The representation of the model: \n{repr(model)}\n")

input_text = input("Please input your prompt below: \n")
default_prompt = '你好,能给我讲讲中国的历史吗'

start_time = time.time()
output_text, _ = model.chat(tokenizer, input_text, history=[], streamer=streamer, do_sample=False, num_beams=1)
elapsed_time = time.time() - start_time

print(f"\n => output text: \n{output_text}\n")
print(f"\n => costed time for method: {load_chatglm2_32k_method}: {elapsed_time:.2f} seconds\n")
