import os
import warnings
import json
from typing import Optional, Tuple, Union, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, TextStreamer
from transformers.models.llama import modeling_llama



model_dir = '/mnt/cfs_bj/big_model/models/'
model_name = 'openbuddy-llama2-13b-v8.1-fp16'
device = "cuda:0"

## prompt length: 9547
## negative log-likelyhood: 4.985832691192627
## perplexity:  146.32537841796875

# sample_path = './data/context_mfqa.json'
# sample = json.load(open(sample_path))
# prompt = f"阅读以下文字并用中文简短回答：\n\n{sample['context']}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{sample['input']}"
# context = f'User: {prompt}\n\nAssistant:' 

## prompt length: 2361
## negative log-likelyhood: 1.900581955909729
## perplexity:  6.689786911010742
sample_path = './data/context_plotpatcher.json'
sample = json.load(open(sample_path))
prompt = "{prompt}\n\n{question}".format(**sample)
context = f'User: {prompt}\n\nAssistant:'


tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name), 
                                          trust_remote_code=True, 
                                          use_fast=True)
model = AutoModelForCausalLM.from_pretrained(os.path.join(model_dir, model_name), 
                                             trust_remote_code=True, 
                                             device_map='auto',
                                             torch_dtype=torch.bfloat16)


# set tokenizer
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.unk_token
streamer = TextStreamer(tokenizer, skip_prompt=True)

print(f"\n => tokenizer: \n{tokenizer}\n")

print(f"\n => model: \n{model}\n")


# get input ids and target ids
input_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
target_ids = input_ids.clone()
print(f"prompt length: {input_ids.shape[-1]}")

# get loss
with torch.no_grad():
    outputs = model(input_ids=input_ids, labels=target_ids)
    nll = outputs.loss # negative log-likelihood
    
print(f"negative log-likelyhood: {nll}")
pll = torch.exp(nll)
print(f"perplexity:  {pll}")




