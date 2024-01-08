############################       base libs import     ############################

import os
import argparse
import json
import time
from typing  import List, Dict, Union, Optional, Any

############################     config class    ############################


class Config(object):
    """The base class of the config"""
    config_template = {}
    
    def __init__(self, register_class: str) -> None:
        self.register_class = register_class
        self.config_dict = {}
    
    def _get_config_dict(self) -> None:
        """get the self.config dict to be used in self.save"""
        self.config_dict = {}
    
    def get_config_dict(self) -> None:
        self._get_config_dict()
        return self.config_dict
    
    def save(self, save_path: str) -> None:
        """to save the config params to a config.json"""
        self.get_config_dict()
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config_dict, ensure_ascii=False)
    
    @staticmethod 
    def load(config_path: Union[str, dict]) -> dict:
        """load from the config.json to instantiate a config object"""
        if isinstance(config_path, dict):
            return config_path
        # check the file format
        if not config_path.endswith(".json"):
            raise ValueError(f"the config path {config_path} should end with '.json'")
        # check if the file path exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"the config file {config_path} does not exist")
        # load json object from the config file
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            
        config_dict['config_path'] = config_path
        return config_dict
    
    @staticmethod
    def template() -> dict:
        """generate a template config dict to help the user config"""
        config_template = {}
       
        return config_template
        
   
class ModelConfig(Config):
    """The config class for the model"""
    def __init__(self,
                register_class: str, # the name of one of the registered LCWAugModel class to be instantiated
                model_name: str, # the model unique name to show on the report
                model_path: str, # the model path to load the model
                tokenizer_path: str = '', # the tokenizer path to load the tokenizer, default using the corresponding model's path 
                use_cache: str = 'true', # whether to use the cache in model.generate call or not
                max_prompt_length: int = 32_000, # the maximum length of prompt words, default 32k, and -1 means no limit
                use_fast_tokenizer: str = "true", # whether to use the fast rust-based tokenizer, default 'true'
                torch_dtype: str = 'float16',  # The type of torch dtype in use, default using float16
                device_map: str = 'auto', # The device map distribution, default using 'auto' to automatically dispatch from all the available devices
                tensor_parallel: str = 'false', # whether to use tensor parallel, deault 'false' for not using
                decode_strategy: str = 'greedy', # how to decode the output text from the model's logits, default using greedy, the other choices are beam, topk_sampling, topp_sampling, etc, depending on the model 
                aug_method: str = '', # the name of lcw aug method for this model, if None, it means not use aug method or the model to be load has already been auged during modeling
                **kwargs # additional config params specific by the register class
                ) -> None:
        super().__init__(register_class) # check register class

        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.use_cache = use_cache
        self.max_prompt_length = max_prompt_length
        self.use_fast_tokenizer = use_fast_tokenizer
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.tensor_parallel = tensor_parallel
        self.decode_strategy = decode_strategy
        self.aug_method = aug_method
        
    
        for k, v in kwargs.items(): setattr(self, k, v)
    
    def _get_config_dict(self) -> None:
        self.config_dict = {
            'register_class': self.register_class,
            'model_name': self.model_name,
            'model_path':self.model_path,
            'tokenizer_path': self.tokenizer_path,
            'use_cache': self.use_cache,
            'max_prompt_length': self.max_prompt_length,
            'use_fast_tokenizer': self.use_fast_tokenizer,
            'torch_dtype': self.torch_dtype,
            'device_map': self.device_map,
            'tensor_parallel': self.tensor_parallel,
            'decode_strategy': self.decode_strategy,
            'aug_method': self.aug_method,
        }
        
        for k, v in vars(self).items():
            if k == 'config_dict' or k in self.config_dict:
                continue
            self.config_dict[k] = v
        
    @staticmethod
    def load(config_path: Union[str, dict]) -> Any:
        config_dict = Config.load(config_path)
        model_config = ModelConfig(**config_dict)
        return model_config

    @staticmethod
    def template() -> dict:
        config_template = {
            'register_class': 'required a registered LCWAugModel (sub)class key here',
            'model_name': "required a specific model's name you like here",
            'model_path': 'required a model loading path here',
            'tokenizer_path': "the tokenizer loading path here, you can keep it empty to use the corresponding model's path",
            'use_cache':"'true' / 'false' to specify whether to use the cache in model.generate call or not, default 'true'",
            'use_fast_tokenizer': "whether to use the fast rust-based tokenizer, default 'true', setting if to 'false' will use python-based tokenizer",
            'max_prompt_length': 'the maximum length of the prompt words, if over, truncate it, default as 32k, and -1 means no limit',
            'torch_dtype': "the torch dtype string of the model you want, default using float16, you can also use bfloat16, etc",
            'device_map': "the device map distribution, default using 'auto', or you can pass 'single' to use the only first gpu, or more generally, pass the dict-like device map to specify",
            'tensor_parallel': "whether to use tensor parallel, default 'false' for not using",
            'decode_strategy': "the string to set how to decode the output text from the model's logits, default using 'greedy', the other choices are 'beam', 'topk_sampling', 'topp_sampling', etc, depending on the model",
            'aug_method': 'you can set the aug method here, or keep it empty for no aug',
            '[additional config]': 'any basic python type',
        }
        
        return config_template

       
class DatasetConfig(Config):
    """The config class for the dataset"""
    def __init__(self, 
                register_class: str, # the name of one of the registered LCWDataset class to be instantiated
                dataset_name: List[str], # the dataset name (may be with subdataset name) to be loaded,
                dataset_path: str = '', # the dataset loading path here, default loading from the hugging-face.co if empty
                split: str = 'test', # the split of the dataset to be actually used, default as test
                batch_size: int = 1, # the batched sample number to be used during iteration
                prompt_template: str = '',  # this is required to generate the actual inputs for the model, but may be set somewhere else, depending on both the dataset and the model
                max_new_tokens: int = 0, # the maximum number of new tokens to be generated by the model for this dataset， -1 meanse no limit, 0 means to be automatically set by the dataset itself
                length_splits: List[int] = [1000, 2000, 4000, 8000, 16000],  # evalute at different length split range
                metrics: List[str] = [],  # metrics used for evaluating the model, by default used the metrics defined in the dataset
                main_metric: str = "", # the main metric if there's multi ones, to be used in the radarchart, default using the first metric in the score output
                ppl_metric: str = "false", # whether to use the ppl metric, default as false
                few_shot: int = 0, # the few-shot examples for each sample
                few_shot_split: str = 'dev', # the split of the dataset to choose the few-shot examples from, if few_shot > 0
                few_shot_prefix: str = "", # the prefix prompt of the few-shot examples, which may be specific to the dataset
                few_shot_suffix: str = "", # the suffix prompt of the few-shot examples, which may be specific to the dataset
                **kwargs # additional config params specific by the register class 
                ) -> None:
        super().__init__(register_class)
        
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.split = split
        self.batch_size = batch_size
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.length_splits = length_splits
        self.metrics = metrics
        self.main_metric = main_metric
        self.ppl_metric = ppl_metric
        self.few_shot = few_shot
        self.few_shot_split = few_shot_split
        self.few_shot_prefix= few_shot_prefix
        self.few_shot_suffix = few_shot_suffix
        
        for k, v in kwargs.items(): setattr(self, k, v)
        
    def _get_config_dict(self) -> None:
        self.config_dict = {
            'register_class': self.register_class,
            'dataset_name': self.dataset_name,
            'dataset_path': self.dataset_path,
            'batch_size': self.batch_size,
            'prompt_template': self.prompt_template,
            'max_new_tokens': self.max_new_tokens,
            'length_splits': self.length_splits,
            'mmetric': self.metrics,
            'main_metric': self.main_metric,
            'ppl_metric': self.ppl_metric,
            'few_shot': self.few_shot,
            'few_shot_split': self.few_shot_split,
            'few_shot_prefix': self.few_shot_prefix,
            'few_shot_suffix': self.few_shot_suffix
        }
        
        for k, v in vars(self).items():
            if k == 'config_dict' or k in self.config_dict:
                continue
            self.config_dict[k] = v
         
    @staticmethod
    def load(config_path: Union[str, dict]) -> Any:
        config_dict = Config.load(config_path)    
        dataset_config = DatasetConfig(**config_dict)
        return dataset_config

    @staticmethod
    def template() -> dict:
        config_template = {
            'register_class': 'required a registered LCWDataset (sub)class key here',
            'dataset_name': ["required the specific dataset's loading name(s) here, following the hugging-face style",],
            'dataset_path': "required the specific dataset's loading path here, default loading from the huggingface.co if empty",
            'split': "the split of the dataset to be actually used, default as 'test' ",
            'batch_size': 'the batched sample number to be used during one iteration, default as 1(int)',
            'prompt_template': 'you can set the specific prompt template you need here, or keep it empty to let it be generated automatically probably',
            'max_new_tokens': 'the maximum number of new tokens to be generated by the model for this dataset, -1 means no limit, default using 0 to let dataset handle it',
            'length_splits': ["the lengths of the different length split ranges, default as [1000, 2000, 4000, 8000, 16000](List[int])",],
            'metrics': ["the metrics used for evaluating the model, you can keep it empty list to use the metrics defined in the dataset itself",],
            'main_metric': "the main metric for evaluating the model, keep it empty to default use the first metric in the score output",
            'ppl_metric': "the metric used for evaluating the model's perplexity, default 'false' to not evaluate it, since it may cost twice forward",
            'few_shot': "the number of few-shot examples, default set 0, i.e. zero-shot setting",
            'few_shot_split': "the split of the dataset for few-shot setting, default as 'dev' ",
            'few_shot_prefix': "the few-shot example prefix prompt, default empty, but should be set when using few-shot setting, maybe specific to dataset",
            'few_shot_suffix': "the few-shot example suffix prompt, default empty, but should be set when using few-shot setting, maybe specific to dataset",
            '[additional config]': 'any basic python type',
        }
        
        return config_template


class EvaluatorConfig(Config):
    def __init__(self,
                 model_configs: List[Dict], # the list of config dicts of the model which will be evaluated.
                 dataset_configs: List[Dict], # the list of config dict of the dataset which will be evaluated.
                 register_class: str = 'base_evaluator', # the name of the Evaluator class to be instantiated
                 exp_name: str = '', # one evaluate config points to one exp, and this exp name will be the name of the output dir, default as '' to use the timestamp + config_file_name
                 **kwargs, # additional config params specific by the register class
                 ) -> None:
        super().__init__(register_class)
        
        self.model_configs = [ModelConfig(**mc) for mc in model_configs]
        self.dataset_configs = [DatasetConfig(**dc) for dc in dataset_configs]
        
        for k, v in kwargs.items(): setattr(self, k, v)
        
        self.exp_name = exp_name if exp_name != '' \
                        else f"exp-config[{os.path.basename(self.config_path)}]-time[{time.strftime('%Y-%m-%d %H:%M')}]"
        
    def _get_config_dict(self) -> None:
        self.config_dict = {
            'register_class': self.register_class,
            'exp_name': self.exp_name,
            'model_configs': [],
            'dataset_configs': [],
        }
        
        for model_config in self.model_configs:
            self.config_dict['model_configs'].append(model_config.get_config_dict())
        for dataset_config in self.dataset_configs:
            self.config_dict['dataset_configs'].append(dataset_config.get_config_dict())
            
        for k, v in vars(self).items():
            if k == 'config_dict' or k in self.config_dict:
                continue
            self.config_dict[k] = v
            
    @staticmethod
    def load(config_path: Union[str, dict]) -> Any:
        config_dict = Config.load(config_path)
        for k, v in {'model_configs': 'ModelConfig', 'dataset_configs': 'DatasetConfig'}.items():
            if k not in config_dict:
                raise KeyError(f"There should be a key '{k}' which points to a list of {v} objects")
        
        eval_config = EvaluatorConfig(**config_dict)
        return eval_config
        
    @staticmethod
    def template() -> dict:
         config_template = {
            'register_class': "required a registered Evaluator class key here, default as 'base_evaluator'", 
            'exp_name': 'you can speficify the name of the exp you want, or keep it empty to use the timestamp + config_file_name',
            'model_configs': [ModelConfig.template()],
            'dataset_configs': [DatasetConfig.template()],
         }
         
         return config_template
   
   
#####################     geneate config template func   ###################
  
    
def gen_config_template(args):
    if args.config_type == 'model':
        config_template = ModelConfig.template()
    elif args.config_type=='dataset':
        config_template = DatasetConfig.template()
    elif args.config_type=='evaluate':
        config_template = EvaluatorConfig.template()
        
    with open(args.config_path, 'w') as f:
        json.dump(config_template, f)
        
    print(f"Generated a config template file for you at {args.config_path}")
    
     
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        description="This is the script to define the config class and generate the config file template"
    )
    parser.add_argument("--config-path", type=str, required=True, 
                        default='The config file path to generate')
    parser.add_argument("--config-type", type = str, default = 'evaluate', 
                        help='The config template type to generate', choices=['evaluate','model', 'dataset'])
    
    args = parser.parse_args() 
    gen_config_template(args)
