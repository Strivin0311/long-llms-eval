import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
import argparse
import json
from copy import deepcopy

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


def set_gpus(gpu_num):
    if gpu_num == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    elif gpu_num == 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    elif gpu_num == 4:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    elif args.gpu_num == 8:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

def main(args):
    ## set gpu nums: NOTE: the visible gpus should be set before the 'import torch' or any code which initiializes cuda
    set_gpus(args.gpu_num)
    
    import sys
    sys.path.append('../')
    sys.path.append('../evals')
    
    from transformers import logging
    logging.set_verbosity_error()
    
    from evals.config import ModelConfig
    from evals.model import load_model
    from evals.utils import time_manager, info_str, memory_used, GPUMemoryMonitor
    
    import torch
    print(f"The number of available gpus: {os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count()}\n")

    ## load the model configs
    obllama2_13B_rerope_config = ModelConfig.load({
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

    obllama2_13B_flash_rerope_inner_config = ModelConfig.load({
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

    obllama2_13B_flash_rerope_outter_config = ModelConfig.load({
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

    obllama2_13B_ntk_config = ModelConfig.load({
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

    obllama2_13B_flash_ntk_config = ModelConfig.load({
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


    len_splits = [8_000, 10_000, 12_000, 15_000, 18_000, 20_000, 22_000, 24_000, 26_000, 28_000, 30_000, 32_000, 34_000]
    num = 5
    src_data_root = "../datasets/LongBench-repo/datasets/zh"
    maxlen_config_path = "../datasets/LongBench-repo/config/dataset2maxlen.json"
    prompt_config_path = "../datasets/LongBench-repo/config/dataset2prompt.json"

    model_configs = {
        "obllama2-13B" + "-" + 'rerope': obllama2_13B_rerope_config,
        "obllama2-13B" + "-" + 'flash_rerope_inner': obllama2_13B_flash_rerope_inner_config,
        "obllama2-13B" + "-" + 'flash_rerope_outter': obllama2_13B_flash_rerope_outter_config,
        "obllama2-13B" + "-" + 'ntk': obllama2_13B_ntk_config,
        "obllama2-13B" + "-" + 'flash_ntk': obllama2_13B_flash_ntk_config,
    }

    boundary_template = {
            'length': [],
            'pass': [],
            'range': [],
            'aug_method': [],
            'gpu_usage': [],
        }

    def report_inference_boundary(args, boundaries, model_names: list, output_dir):
        # report output as json
        with open(os.path.join(output_dir, f"inference_boundary_output.json"), 'w', encoding='utf-8') as f:
            json.dump(boundaries, f)
        
        # report as markdown table
        df = pd.DataFrame(boundaries).sort_values(by='length', key=lambda x: x.astype(int))
        table_str = ""
        for model_name in model_names:
            df_model = df[df['aug_method'] == model_name]
            table_str += f"## Inference Boundary Report on model {args.model_base} aug by {model_name} with {args.gpu_num} {args.gpu_type}\n"
            table_str += tabulate(df_model, headers='keys', tablefmt="pipe")+ "\n\n"
            
        with open(os.path.join(output_dir, f"inference_boundary_report.md"), 'w', encoding='utf-8') as f:
            f.write(table_str)
        
        # report as curve
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        
        sns.lineplot(data=df, x='length', y='gpu_usage', 
                     hue='aug_method', style='pass', markers=True, ax=ax)
        plt.xticks(rotation=90)
        plt.xlabel('sample length')
        plt.ylabel('GPU Memory Usage (GB)')
        plt.title(f'Inference Boundary for {args.model_base} aug by \n{model_names}\n with {args.gpu_num} gpus', fontdict=dict(size=8))
        plt.savefig(os.path.join(output_dir, f"inference_boundary_curve.png"))

    ## gen / load the test data
    if args.mode == "gen_data":
        args.verbose = True
        print(info_str("Generating testing data"))
        with open(maxlen_config_path, 'r', encoding='utf-8') as f:
            maxlen_config = json.load(f)
        with open(prompt_config_path, 'r', encoding='utf-8') as f:
            prompt_config = json.load(f)
            
        for mil, mal in tqdm(zip(len_splits[:-1], len_splits[1:]), total=len(len_splits)-1):
            samples, max_lens, prompts = [], [], []
            output_dir = f"./data/test_{mil//1_000}k_{mal//1_000}k_{num}/"
            for filename in os.listdir(src_data_root):
                filepath = os.path.join(src_data_root, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        sample = json.loads(line)
                        if mil <= sample['length'] < mal:
                            samples.append(sample)
                            max_lens.append(maxlen_config[filename.split('.')[0]])
                            prompts.append(prompt_config[filename.split('.')[0]])
                        if len(samples) == num: break
                if len(samples) == num: break
            
            with open(os.path.join(output_dir, 'samples.jsonl'), 'w', encoding='utf-8') as f:
                for sample in samples: f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            with open(os.path.join(output_dir, 'max_new_tokens.json'), 'w', encoding='utf-8') as f:
                json.dump(max_lens, f)
            with open(os.path.join(output_dir, 'prompt_templates.json'), 'w', encoding='utf-8') as f:
                json.dump(prompts, f, ensure_ascii=False)
        
    elif args.mode == "report_all":
        report_dir = os.path.join(args.output_root, args.model_base, f"{args.gpu_num}-gpus")
        boundaries = deepcopy(boundary_template) 
        for model_name in os.listdir(report_dir):
            with open(os.path.join(report_dir, model_name, 'inference_boundary_output.json'), 'r', encoding='utf-8') as f:
                boundary = json.load(f)
            for k in boundaries:
                boundaries[k] += boundary[k]
                
        result_dir = os.path.join(report_dir, 'all')
        if not os.path.exists(result_dir): os.makedirs(result_dir) 
        
        report_inference_boundary(args, boundaries, os.listdir(report_dir), result_dir)
        
        print(info_str("Done!")) 
            
    else: # test_inference_boundary
        result_dir = os.path.join(args.output_root, args.model_base, f"{args.gpu_num}-gpus", args.model_name)
        config_name = args.model_base + "-" + args.model_name
        if not os.path.exists(result_dir): os.makedirs(result_dir) 
        
        ## evaluate the model on the data
        @time_manager("Evaluation")
        def evaluate_simple(config, args, mil, mal, max_lens, prompts, already_fail):
            print(info_str(f"Testing Inference Bounadry on model {args.model_base} aug by {args.model_name} with {args.gpu_num} {args.gpu_type} in length range: [{mil}, {mal})"))
            
            preds, boundary = [], deepcopy(boundary_template)
            if already_fail:
                for idx, sample in enumerate(samples):
                    boundary['length'].append(str(sample['length']))
                    boundary['pass'].append('no')
                    boundary['range'].append(f"({mil}, {mal})")
                    boundary['aug_method'].append(args.model_name)
                    boundary['gpu_usage'].append(args.gpu_limit)
            else:
                lcw_model = load_model(config, args)
                for idx, sample in enumerate(samples):
                    input_text = prompts[idx].format(**sample)
                    try:
                        with GPUMemoryMonitor(reduce='max') as monitor:
                            preds.append(lcw_model(
                                inputs=[input_text], refs=[], 
                                lengths=[sample['length']], infos=[],
                                max_new_tokens=max_lens[idx], start_idx=idx
                            ))
                            
                            boundary['length'].append(str(sample['length']))
                            boundary['pass'].append('yes')
                            boundary['range'].append(f"({mil}, {mal})")
                            boundary['aug_method'].append(args.model_name)
                            boundary['gpu_usage'].append(monitor.get_memory(reduce='max'))
                            torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            print(f"CUDA OOM when evaluating on sample {idx+1} with length {sample['length']}")
                            boundary['length'].append(str(sample['length']))
                            boundary['pass'].append('no')
                            boundary['range'].append(f"({mil}, {mal})")
                            boundary['aug_method'].append(args.model_name)
                            boundary['gpu_usage'].append(args.gpu_limit)
                            torch.cuda.empty_cache()
                            continue
                        else: raise
                
            return boundary, len(preds) == num
                    
        already_fail = False
        boundaries = deepcopy(boundary_template)     
        for mil, mal in tqdm(zip(len_splits[:-1], len_splits[1:])):
            input_dir = f"./data/test_{mil//1_000}k_{mal//1_000}k_{num}/"
            samples, max_lens, prompts = [], [], []
            with open(os.path.join(input_dir, 'samples.jsonl'), 'r', encoding='utf-8') as f:
                for line in f: samples.append(json.loads(line))
            with open(os.path.join(input_dir, 'max_new_tokens.json'), 'r', encoding='utf-8') as f:
                max_lens = json.load(f)
            with open(os.path.join(input_dir, 'prompt_templates.json'), 'r', encoding='utf-8') as f:
                prompts = json.load(f)
                
            boundary, all_pass = evaluate_simple(
                model_configs[config_name], 
                args, mil, mal, max_lens, prompts,
                already_fail
            )
            
            for k in boundaries: boundaries[k] += boundary[k]
            if not all_pass: already_fail = True
            
            torch.cuda.empty_cache()
            
        report_inference_boundary(args, boundaries, [args.model_name], result_dir)     
        
        print(info_str("Done!"))


parser = argparse.ArgumentParser(description="test the inference boundary for each aug method in single-gpu or multi-gpus env")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--mode", type=str, default="test_inference_boundary", choices=["gen_data", 
                                                                           "test_inference_boundary",
                                                                           "report_all"
                                                                           ])
parser.add_argument("--model_base", type=str, default="obllama2-13B")
parser.add_argument("--model_name", type=str, default="rerope", choices=["rerope", 
                                                                        "flash_rerope_inner",
                                                                        "flash_rerope_outter",
                                                                        "ntk",
                                                                        "flash_ntk",
                                                                        ])
parser.add_argument("--gpu_num", type=int, default=1, choices=[1, 2, 4, 8])
parser.add_argument("--output_root", type=str, default="./data/inference_boundary_results")
parser.add_argument("--gpu_type", type=str, default="A800")
parser.add_argument("--gpu_limit", type=float, default=80.0)

args = parser.parse_args()

print(f"The arguments are: \n{args}\n")

main(args)