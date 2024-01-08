############################       base libs import     ############################

import os
import argparse
import warnings
from typing import List, Tuple, Dict, Optional, Union

############################       dev libs import     ############################

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, TextStreamer
import tensor_parallel as tp
from accelerate import infer_auto_device_map
import requests

############################       own libs import     ############################

from base import BaseModule
from config import ModelConfig
from utils import info_str, info_output, info_ordinal, info_dict
from utils import memory_used, num_of_params, get_torch_dtype
from utils import time_manager, Timer
from aug import LLaMAAugManager
from decode import DecodingStrategy

############################     model class    ############################


class LCWAugModel(BaseModule):
    """The base model class for the LCW-Aug model, which implements the basic loading and predicting functions
    with some common aug methods used
    """
    
    def __init__(self, model_config, args) -> None:
        super().__init__()
        
        # init attributes
        self.args = args
        for k, v in model_config.get_config_dict().items(): setattr(self, k, v)
        
        # init aug methods
        self._init_aug_methods()
        self._init_aug_params()
        self._info_aug_method_and_params()
        
        # load model and tokenizer
        self._load()
        
        # init decoding strategies
        self._init_decode_strategies()
        self._init_decode_stategy_params()
        self._info_decode_strategy_and_params()
        
    def _init_decode_strategies(self):
        """Init the decode strategies, 
        Format:
            self.decode_strategies = {
                'strategy_name': {
                    'strategy': decode_config_func (which is called in _get_decode_strategy)
                    'args': the available arguments to be set in model config file for this strategy
                }
            }
        """
        self.decode_strategies = DecodingStrategy.SUPPORT_STRATEGIES
        
    def _init_decode_stategy_params(self):
        """Init the decode strategy params for decoding
        """
        if self.decode_strategy not in self.decode_strategies:
            raise KeyError(f"Invalid decoding strategy {self.decode_strategy}" + 
                           f"Available decoding strategies are: \n{self.decode_strategies.keys()}\n")
        self.decode_strategy_params = {}
        
        for arg_name, arg_default_value in self.decode_strategies[self.decode_strategy]['args'].items():
            self.decode_strategy_params[arg_name] = vars(self).get(arg_name, arg_default_value)
        
    def _init_aug_methods(self):
        """Init the aug methods here
        Format:
        self.aug_methods = {
            'aug_method_key': {
                'name': "...",
                'args': [...]
            }
        }
        """
        self.aug_methods = {"": {}}
          
    def _init_aug_params(self) -> None:
        """Init the parameters for aug method used
        """
        if self.aug_method not in self.aug_methods: # check if the aug method supported for llama
            raise ValueError(f"Invalid aug method {self.aug_method}\n" +
                             f"Available aug methods are: \n{self.aug_methods.keys()}\n")
        
        self.aug_params = {}
        if self.aug_method != '': # aug method used
            for arg_name, arg_default_value in self.aug_methods[self.aug_method]['args'].items():
                self.aug_params[arg_name] = vars(self).get(arg_name, arg_default_value)
         
    def _load(self) -> None:
        """Load the model and corresponding tokenizer here
        """
        ### load tokenizer
        # set tokenizer path
        if self.tokenizer_path == '':
            self.tokenizer_path = self.model_path
        # verbose info
        if self.args.verbose:
            print(info_str(f"Loading tokenizer from {self.tokenizer_path}"))
        # load
        self._load_tokenizer()
        if self.args.verbose:
            print(info_str(f"The tokenizer has been loaded, whose info is shown below: \n {self.tokenizer}\n"))
        
        ### load streamer
        self._load_streamer()
        if self.args.verbose:
            print(info_str(f"The streamer has been loaded, whose info is shown below: \n {self.streamer}\n"))
        
        ### load model
        # verbose info
        if self.args.verbose:
            initial_memory = memory_used(format=False)
            print(info_str(f"Loading model {self.model_name} from {self.model_path}"))
            print(info_str(f"Using {self.device_map} device map"))
            print(info_str(f"{'' if self.tensor_parallel.lower() == 'true' else 'NOT '}Using tensor parallel"))
        # load
        self._load_model()
        # verbose info
        if self.args.verbose:
            memory_delta = memory_used(format=False) - initial_memory
            print(info_str(f"The model {self.model_name} has been loaded, whose strcuture is shown below: \n {self.model}\n"))
            print(info_str(f"which has {num_of_params(self.model)} paraemters and costed {memory_used(mem=memory_delta)} memory"))
        
    def _load_tokenizer(self) -> None:
        """Load the tokenizer here, called in self._load during __init__
        """
        # load
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, 
                                       trust_remote_code=True,
                                       use_fast = self.use_fast_tokenizer.lower() == "true"
                                       )
     
    def _load_streamer(self) -> None:
        """Load the streamer to decode the generated token into stdout as soon as it's ready
        for one verbose info
        """ 
        if not self.args.verbose:
            self.streamer = None
        else:
            self.streamer = TextStreamer(self.tokenizer, skip_prompt=True)
     
    @time_manager("Loading the model")
    def _load_model(self) -> None:
        """Load the model here, called in self._load during __init__
        """
        device_map = self.device_map
        parallel = self.tensor_parallel.lower() == 'true'
        
        # load   
        if device_map == 'auto':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, 
                                        trust_remote_code=True, 
                                        device_map='auto', # use all gpus
                                        torch_dtype=get_torch_dtype(self.torch_dtype)
                                        )
        elif device_map == 'single':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, 
                                        trust_remote_code=True, 
                                        torch_dtype=get_torch_dtype(self.torch_dtype)
                                        ).cuda() 
        else:
            raise ValueError(f"Invalid device map type for _load_model: {device_map}")
        # to eval mode
        self.model.eval() 
        # parallelize on each available gpu
        if parallel: self.model = tp.tensor_parallel(self.model) 
         
    def _get_inputs_dict(self, inputs: List[str]) -> Dict:
        """Get the inputs dict with input_ids, attention mask, and context length from the tokenizer
        which is called in forward
        """
        inputs_dict = self.tokenizer(inputs, # no padding here
                                     return_tensors='pt').to(self.model.device)
        
        return inputs_dict

    def _truncate_inputs(self, 
                         inputs: List[str],
                         ) -> List[str]:
        """Truncate each input text to not exceed the max prompt length, called in forward
        Note that: the truncate strategy is fixed with truncate_prompt_from_middle,
        following the same strategy LongBench used, 
        since the start and the end substrings often contain important information than the middle one
        """
        if self.max_prompt_length == -1: # no limit
            return inputs
        
        def truncate_prompt_from_middle(prompt):
            # truncate from the middle
            if len(prompt) > self.max_prompt_length:
                half_max_length = self.max_prompt_length // 2
                prompt = prompt[:half_max_length] +  prompt[-half_max_length:]
                
            return prompt
        
        return list(map(truncate_prompt_from_middle, inputs))

    def _get_outputs(self, 
                     inputs: List[str], refs: List, 
                     lengths: List[int], infos: List[Dict],
                     max_new_tokens, **kwargs) -> List[Dict]:
        """Get the pred output dict list, which is called in forward
        """
         # get inputs dict
        inputs_dict = self._get_inputs_dict(inputs)
        
        # get decode strategy
        decode_strategy = self._get_decode_strategy(max_new_tokens)
        
        # generate
        outputs, inference_times = self._generate(inputs_dict, decode_strategy)
        
        # decode
        context_length = inputs_dict.input_ids.shape[-1]
        outputs = [output[context_length:] for output in outputs]
        pred_outputs = self._decode(inputs, outputs, refs, lengths, infos, inference_times)
        
        return pred_outputs
    
    def _get_decode_strategy(self, max_new_tokens) -> Dict:
        """Get the decode strategy, including generation config, logits processor, etc 
        which is called in _get_outputs
        """
        decode_strategy_func = self.decode_strategies[self.decode_strategy]['strategy']
        decode_strategy = decode_strategy_func(max_new_tokens, **self.decode_strategy_params)   
        return decode_strategy
       
    def _generate(self, inputs_dict: Dict, decode_strategy: Dict) -> Tuple[List, List]:
        """generate the output id lists and the inference time, which is called in _get_outputs
        """ 
        with Timer() as timer:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs_dict,
                    return_dict_in_generate=False,
                    output_scores=False,
                    streamer=self.streamer,
                    use_cache=self.use_cache.lower() == "true",  
                    **decode_strategy,
                )
            inference_time = timer.time
            return outputs, [inference_time for _ in outputs]

    def _decode(self, 
                inputs: List[str], outputs: List[Union[int, str]], refs: List, 
                lengths: List[int], infos: List[Dict],
                inference_times: List[float]
                ) -> List[Dict]:
        """decode the outputs to the predictions, and return the pred_output_dict list
        which is called in _get_ouputs
        Return:
            pred_outputs = [{
                prediction: "",
                reference: "" | [""],
                length: n,
                speed: (sec / word),
                {info}
            }]
        """
        
        if isinstance(outputs[0], str): outputs_text = outputs
        else:
            outputs_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        pred_outputs = [{
                self.input_key: info_output(input_text),
                self.prediction_key: prediction,
                self.reference_key: reference,
                self.length_key: length,
                self.speed_key: inf_time / (len(prediction) + length),
                **info
            } for input_text, prediction, reference, length, info, inf_time in zip(inputs, outputs_text, refs, lengths, infos, inference_times)
        ]
        
        return pred_outputs
    
    def _evaluate_ppl(self, inputs: List[str], pred_outputs: List[Dict]) -> List[Dict]:
        """Evaluate the perplexity for each input text in input and append the ppl score to each pred_output dict
        Note: only the white-box or gray-box language model, i.e. logits/loss-available can evaluate the accurate ppl metric
        """
        
        @torch.no_grad()
        def compute_ppl(input_text):
            inputs_dict = self._get_inputs_dict([input_text])
            target_ids = inputs_dict.input_ids.clone()
            nll = self.model(**inputs_dict, labels=target_ids).loss
            ppl = float(torch.exp(nll).cpu().numpy())
            return ppl
        
        for input_text, pred_output in zip(inputs, pred_outputs):
            pred_output[self.ppl_key] = compute_ppl(input_text)
        
        return pred_outputs
        
    def _info_inputs(self, inputs: List[str], **kwargs) -> None:
        """Print the key infos about the prediction outputs, often used in verbose"""
        start = kwargs.get('start_idx', 0)
        for i, input_text in enumerate(inputs):
            print(info_str(f"Input for the {info_ordinal(start + i + 1)} sample"))
            print(f"{info_output(input_text, return_info=True)}\n") 
             
    def _info_outputs(self, pred_outputs: List[Dict], **kwargs) -> None:
        """Print the key infos about the prediction outputs, often used in verbose"""
        start = kwargs.get('start_idx', 0)
        for i, pred_output in enumerate(pred_outputs):
            print(info_str(f"Prediction output for the {info_ordinal(start + i + 1)} sample"))
            # prediction
            print(f"=> {self.prediction_key}: \n{info_output(pred_output[self.prediction_key])}\n") 
            # references
            if isinstance(pred_output[self.reference_key], list): 
                for i, ref in enumerate(pred_output[self.reference_key]):
                    print(f"=> {self.reference_key} {i+1}: \n{info_output(ref)}") 
            else:
                print(f"=> {self.reference_key}: \n{info_output(pred_output[self.reference_key])}")
            # other infos    
            print("") 
            print(f"=> {self.length_key}: {pred_output[self.length_key]}\n" + 
                f"=> {self.speed_key}: {pred_output[self.speed_key]:.2e} (sec / word)"
            )
        
    def _info_aug_method_and_params(self) -> None:
        if self.args.verbose: # verbose info
            if self.aug_method == '': # no aug method
                print(info_str(f"No aug method used"))
            else:
                print(info_str(f"Using aug method {self.aug_methods[self.aug_method]['name']}"))
                print(info_str(
                    f"with aug params: \n" +
                    f"{info_dict(self.aug_params)}\n"
                ))
                
    def _info_decode_strategy_and_params(self) -> None:
        if self.args.verbose:
            print(info_str(f"Using decoding strategy {self.decode_strategy}"))
            print(info_str(
                    f"with params: \n" +
                    f"{info_dict(self.decode_strategy_params)}\n"
                ))
            
    @time_manager("Inference")
    def forward(self, 
                inputs: List[str], refs: List,
                lengths: List[int], infos: List[Dict],
                max_new_tokens: int = -1,
                ppl: bool = False,
                **kwargs
                ) -> List[Dict]:
        """Do prediction here
        :param: inputs: a list of long-text prompt to put into the model: shape = (batch_size x l), l in [1, max_seq_len]
        :param: refs: a list of text-based answers to be scored with the corresponding predictions: shape = (batch_size x ref_shape)
        :param: max_new_tokens: max number of tokens to predict
        :reutrn: a list of pred_output_dict, each of which contains the prediction with the correponding refererence, and some other infos
        """
       
        # truncate inputs
        inputs = self._truncate_inputs(inputs)
        if self.args.verbose:
            self._info_inputs(inputs, **kwargs)
        
        # get pred_output_dict list
        pred_outputs = self._get_outputs(inputs, refs, lengths, infos,
                                         max_new_tokens=None if max_new_tokens == -1 else max_new_tokens,
                                         **kwargs
                                         )
        
        # evalute additional ppl metric optionally 
        # NOTE: only the white-box/gray-box model can evaluate ppl
        if ppl and getattr(self, 'model', None) is not None:
            pred_outputs = self._evaluate_ppl(inputs, pred_outputs)
        elif ppl: warnings.warn("The model is not a white-box/gray-box model, thus cannot evaluate the PPL metric")
        
        if self.args.verbose:
            self._info_outputs(pred_outputs, **kwargs)
        
        return pred_outputs
        

class LLaMALCWAugModel(LCWAugModel):
    """The base model class for the LCW-Aug LLaMA model, which implements the RoPE-aug methods for LLaMA"""
        
    def _init_aug_methods(self):
        super()._init_aug_methods()
        self.aug_methods.update(LLaMAAugManager.SUPPORT_METHODS)
         
    def _load_tokenizer(self) -> None:
        # super()._load_tokenizer() 
        # load
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, 
                                       trust_remote_code=True,
                                       use_fast = False
                                       )
        if self.use_fast_tokenizer.lower() == "true":
            warnings.warn("The LLaMA based model is fixed with python-based tokenizer(use_fast is always set 'False')")
        
        # set something
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.unk_token
               
    def _load_model(self) -> None:
        """Load the LLaMA model with aug methods, called in self._load during __init__
        """
        with LLaMAAugManager( # use llama aug manager to wrap the load model func
            aug_method=self.aug_method, 
            aug_params=self.aug_params,
        ):
            super()._load_model()
          
     
class ChatGLMLCWAugModel(LCWAugModel):
    """The base model class for the LCW-Aug ChatGLM model"""
    
    @time_manager("Loading the model")
    def _load_model(self) -> None:
        # load
        if self.device_map != 'single' or self.tensor_parallel: 
            warnings.warn("The ChatGLM based model can be ONLY used in single gpu mode and no tensor parallel for now")
        
        # FIXME: how to use chatglm in multi-gpus and tensor parallelism mode?
        # load method 1
        # self._load_model_on_multi_gpus()
        # load method 2
        # self._load_model_through_ddp()
        # load method 3
        self._load_model_on_single_gpu()
        
    def _load_model_on_single_gpu(self) -> None:
        self.model = AutoModel.from_pretrained(
                        self.model_path, 
                        trust_remote_code=True,
                        torch_dtype=get_torch_dtype(self.torch_dtype)
                    ).half().cuda()
        self.model.eval() 
        
    def _load_model_through_ddp(self) -> None:
        ### single gpu
        self.model = AutoModel.from_pretrained(
                        self.model_path, 
                        trust_remote_code=True,
                        torch_dtype=get_torch_dtype(self.torch_dtype)
                    ).half()
        
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            if self.args.verbose:
                print(info_str(f"Using {torch.cuda.device_count()} GPUs to parallel the ChatGLM"))
            
        self.model.cuda().eval()
        
        self.model = self.model.module
        
    def _load_model_on_multi_gpus(self) -> None:
        ### multi gpu 
        # => FIXME: problem with RuntimeError: Expected all tensors to be on the same device, but found at least two devices
        self.model = ChatGLMLCWAugModel.load_model_on_gpus(
                    self.model_path, 
                    num_gpus=len(self.args.gpus), 
                )
        self.model.eval()
        
    def _get_decode_strategy(self, max_new_tokens) -> Dict:
        decode_strategy = super()._get_decode_strategy(max_new_tokens, **self.decode_strategy_params)
        
        # extract generation_config nad logits_processor if exists
        generation_config = decode_strategy['generation_config']
        logits_processor = decode_strategy.get('logits_processor', None)
        decode_strategy.pop('generation_config')
        decode_strategy.pop('logits_processor', None)
        
        return dict(
            # the required arguments matching the attributes in GenerationConfig
            # defined in chatglm.chat()
            do_sample=generation_config.do_sample,
            num_beams=generation_config.num_beams,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            logits_processor=logits_processor,
            # other own arguments for model.generate()
            max_new_tokens=max_new_tokens, # to override the max_length defined in chatglm.chat()
            generation_config=generation_config, # the arguments above will override the corresponding paramenters in GenerationConfig
            **decode_strategy, # other decode strategy config beyond generation_config nad logits_processor 
        )
        
    def _get_outputs(self, 
                     inputs: List[str], refs: List, 
                     lengths: List[int], infos: List[Dict],
                     max_new_tokens, **kwargs) -> List[Dict]:
        """ChatGLM has the convenient function chat to directly generate output texts from input texts"""
        
        decode_strategy = self._get_decode_strategy(max_new_tokens)
        
        outputs, inference_times = [], []
        for input_text, length in zip(inputs, lengths):
            with Timer() as timer:
                output_text, _ = self.model.chat(
                                    self.tokenizer, query=input_text, history=[], 
                                    streamer=self.streamer,
                                    use_cache=self.use_cache.lower() == "true",
                                    **decode_strategy
                                )
                outputs.append(output_text)
                inference_times.append(timer.time / (len(output_text) + length))
                
        pred_outputs = self._decode(inputs,outputs, refs, lengths, infos,  inference_times)
            
        return pred_outputs
 
    @staticmethod
    def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
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

    @staticmethod
    def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], 
                            num_gpus: int = 2,
                            device_map: Optional[Dict[str, int]] = None, 
                            auto_device_map: bool = True,
                            **kwargs) -> nn.Module:
        """Copied from official ChatGLM-6B repo 
        and enhance it with auto device map dispatch from:
        https://github.com/THUDM/ChatGLM-6B/pull/1381
        """
        
        if num_gpus < 2 and device_map is None:
            model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
        else:
            if device_map is None:
                if not auto_device_map: # original official manual device map
                    device_map = ChatGLMLCWAugModel.auto_configure_device_map(num_gpus)
                else: # use accelerate to auto infer device map
                    device_map = infer_auto_device_map(model, 
                                                       no_split_module_classes=["GLMBlock"])
            
            model = AutoModel.from_pretrained(checkpoint_path, 
                                              trust_remote_code=True, 
                                              device_map=device_map,
                                              **kwargs).half()
            
            # e.g. Use max_memory to set the upper limit memory size of each device.
            # Huggingface suggest to save some memory of gpu0 for some reasons.
            #device_map = infer_auto_device_map(model, max_memory={0: "4GiB", 1: "10GiB", "cpu": "30GiB"}, no_split_module_classes=["GLMBlock"])
            #print(device_map)
            # model = dispatch_model(model, device_map=device_map)

        return model


class ERNIEBotAugModel(LCWAugModel):
    """The base model class for the ERNIEBot Model (Only remote API can be accessed)"""
    
    _DEFAULT_URL = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token=24.859b2bf5be69c6282ecee040d3755298.2592000.1695817672.282335-35322565"
    
    _DEFAULT_TIMEOUT = 5000    
        
    def _load(self):
        """Only need to load the server url"""
        
        warnings.warn(f"The ERNIE Bot model has ONLY remote API access, thus some configurations you set may not be used at all, " + 
                      "including streamer, decode strategy, aug method and their arguments." + "\n" + 
                      "Meanwhile, the speed measurement makes no sense between EB and any other locally loaded models, " + 
                      "since the network traffic time and no max_new_tokens limit" 
                      )
        
        self.server_url = vars(self).get('server_url', ERNIEBotAugModel._DEFAULT_URL)
        self.timeout = vars(self).get('timeout', ERNIEBotAugModel._DEFAULT_TIMEOUT)
        
    def _get_outputs(self, 
                inputs: List[str], refs: List,
                lengths: List[int], infos: List[Dict],
                max_new_tokens: int = -1,
                **kwargs
                ) -> List[Dict]:
        """ERNIE Bot can only be accessed by remote API service"""
        
        outputs, inference_times = [], []
        for input_text, length in zip(inputs, lengths):
            with Timer() as timer:
                data = {
                    "messages": [{ "role": "user", "content": input_text}],
                    "stream": False,
                }
                resp = requests.post(url=self.server_url, json=data, timeout=self.timeout)
                if "result" in resp.json():
                    output_text = resp.json()["result"]
                else:
                    output_text = resp.json()["error_msg"] 
                if max_new_tokens > 0: output_text = output_text[:max_new_tokens]
                
                outputs.append(output_text)
                inference_times.append(timer.time / (len(output_text) + length))
                
        pred_outputs = self._decode(inputs, outputs, refs, lengths, infos, inference_times)
        
        return pred_outputs
        

############################    model registry    ############################


model_registry = {
    'base': LCWAugModel,
    # llama based
    'base_llama': LLaMALCWAugModel,
    # glm based
    'base_chatglm': ChatGLMLCWAugModel,
    # ernie based
    'base_ernie': ERNIEBotAugModel,
    
}


############################       load func     #################################


def load_model(model_config, args):
    model_class = model_config.register_class
    if model_class not in model_registry:
        raise KeyError(f'The class {model_class} is not registered yet')
    
    model = model_registry[model_class](model_config, args)
    return model


############################       main func     ############################### 


def main(args):
    args.verbose = True
    model_config = ModelConfig.load(args.config_path)
    lcw_model = load_model(model_config, args)
    
    user_input = input("\nPlease enter your prompt for the lcw-aug model: \n")
    pred_output = lcw_model([user_input], [""], max_new_tokens=-1)[0]
    
    print(info_str(f"Here's your prediction output: \n{pred_output}\n"))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This the script for taking a first quick look at some LCWAugModel subclass, \
                                     espeicially when a new model class is built and haven't been tested with the evaluator yet"
                                     )
    parser.add_argument("--config-path", type=str, default="", help="The model config json file path")
    args = parser.parse_args()
    main(args)