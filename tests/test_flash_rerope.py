import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import argparse
import json
import sys
sys.path.append('../')
sys.path.append('../evals')


from evals.config import ModelConfig
from evals.model import load_model
from evals.utils import time_manager


parser = argparse.ArgumentParser(description="test flash attention with fused rerope, implemented on triton")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--model_name", type=str, default="rerope", choices=["rerope", 
                                                                        "flash_rerope_inner",
                                                                        "flash_rerope_outter",
                                                                        "ntk",
                                                                        "flash_ntk",
                                                                        ])
args = parser.parse_args()
args.verbose = True


## load the model configs
rerope_config = ModelConfig.load({
    "register_class": "base_llama",
    "model_name": "obllama2(rerope-logn)",
    "model_path": "/mnt/cfs_bj/big_model/models/openbuddy-llama2-13b-v8.1-fp16",
    "tokenizer_path": "",
    "use_cache": "true",
    "max_prompt_length": -1,
    "use_fast_tokenizer": "false",
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "tensor_parallel": "true",
    "decode_strategy": "greedy",
    "aug_method": "rerope",
    "logn": "true",
    "training_length": 4096,
    "flash": "false",
    "rerope_window": 512,
    "rerope_length": 4096
})

flash_rerope_inner_config = ModelConfig.load({
    "register_class": "base_llama",
    "model_name": "obllama2(flash-rerope-inner)",
    "model_path": "/mnt/cfs_bj/big_model/models/openbuddy-llama2-13b-v8.1-fp16",
    "tokenizer_path": "",
    "use_cache": "true",
    "max_prompt_length": -1,
    "use_fast_tokenizer": "false",
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "tensor_parallel": "true",
    "decode_strategy": "greedy",
    "aug_method": "rerope",
    "logn": "true",
    "training_length": 4096,
    "flash": "true",
    "flash_rerope_inner_apply": "true",
    "rerope_window": 512,
    "rerope_length": 4096
})

flash_rerope_outter_config = ModelConfig.load({
    "register_class": "base_llama",
    "model_name": "obllama2(flash-rerope-outter)",
    "model_path": "/mnt/cfs_bj/big_model/models/openbuddy-llama2-13b-v8.1-fp16",
    "tokenizer_path": "",
    "use_cache": "true",
    "max_prompt_length": -1,
    "use_fast_tokenizer": "false",
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "tensor_parallel": "true",
    "decode_strategy": "greedy",
    "aug_method": "rerope",
    "logn": "true",
    "training_length": 4096,
    "flash": "true",
    "flash_rerope_inner_apply": "false",
    "rerope_window": 512,
    "rerope_length": 4096
})

ntk_config = ModelConfig.load({
    "register_class": "base_llama",
    "model_name": "obllama2(ntk-logn)",
    "model_path": "/mnt/cfs_bj/big_model/models/openbuddy-llama2-13b-v8.1-fp16",
    "tokenizer_path": "",
    "use_cache": "true",
    "max_prompt_length": -1,
    "use_fast_tokenizer": "false",
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "tensor_parallel": "false",
    "decode_strategy": "greedy",
    "aug_method": "ntk",
    "logn": "true",
    "training_length": 4096,
    "dynamic": "true",
    "flash": "false",
    "ntk_ratio": 16
})

flash_ntk_config = ModelConfig.load({
    "register_class": "base_llama",
    "model_name": "obllama2(flash-ntk-logn)",
    "model_path": "/mnt/cfs_bj/big_model/models/openbuddy-llama2-13b-v8.1-fp16",
    "tokenizer_path": "",
    "use_cache": "true",
    "max_prompt_length": -1,
    "use_fast_tokenizer": "false",
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "tensor_parallel": "true",
    "decode_strategy": "greedy",
    "aug_method": "ntk",
    "logn": "true",
    "training_length": 4096,
    "dynamic": "true",
    "flash": "true",
    "ntk_ratio": 16
})


## gen / load the test data
min_len = 30_000
max_len = 33_000
num = 5
prompt_template = "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}"
max_new_tokens = 64

src_path = "../datasets/LongBench-repo/datasets/zh/lsht.jsonl"
data_root = f"./data/test_{min_len//1_000}k_{max_len//1_000}k_{num}/"
data_path = os.path.join(data_root, 'samples.jsonl')

if not os.path.exists(data_root):
    os.makedirs(data_root)
    # gen data from the src path
    data = []
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(data) >= num: break
            sample = json.loads(line)
            if min_len <= sample['length'] <= max_len:
                data.append(sample)
    with open(data_path, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"\nThe data has been generated at {os.path.join(data_root, 'samples.jsonl')}\n")
data = []
with open(data_path, encoding='utf-8') as f:
    for line in f: data.append(json.loads(line))
    print(f"\nThe data {data_path} has been loaded\n")
    
## evaluate the model on the data
@time_manager("Evaluation")
def evaluate_simple(model_name, config, args):
    print(f"\n Evaluting {model_name} on {data_path}\n")
    lcw_model = load_model(config, args)
    preds = []
    for idx, sample in enumerate(data):
        input_text = prompt_template.format(**sample)
        try:
            preds.append(lcw_model(
                inputs=[input_text], refs=[], 
                lengths=[sample['length']], infos=[],
                max_new_tokens=max_new_tokens, start_idx=idx
            ))
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA OOM when evaluating {model_name} on sample {idx} from data {data_path} with length {sample['length']}")
                continue
            else: raise
    with open(os.path.join(data_root, f"result_{model_name}.json"), 'w', encoding='utf-8') as f:
        json.dump(preds, f, ensure_ascii=False)

model_configs = {
    'rerope': rerope_config,
    'flash_rerope_inner': flash_rerope_inner_config,
    'flash_rerope_outter': flash_rerope_outter_config,
    'ntk': ntk_config,
    'flash_ntk': flash_ntk_config,
}

evaluate_simple(args.model_name, model_configs[args.model_name], args)




