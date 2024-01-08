############################       base libs import     ############################

from typing import Any, Tuple, List, Union, Optional, Dict

############################       dev libs import     ############################

import torch
from transformers import GenerationConfig, LogitsProcessorList, LogitsProcessor

############################     decoding strategy class    ############################


class DecodingStrategyFuncs(object):
    def _get_greedy_decode_strategy(max_new_tokens, 
                                    **kwargs) -> Dict:
        """Gets generation config based on greedy-seach decoding strategy in self._get_decode_strategy
        """
        
        return dict(
                generation_config=GenerationConfig(
                    do_sample=False,
                    num_beams=1,
                    
                    early_stopping=True,
                    max_new_tokens=max_new_tokens,
                    **kwargs
                )
            )
 
    def _get_beam_decode_strategy(max_new_tokens, 
                                  num_beams=2, 
                                  **kwargs) -> Dict:
        """Gets generation config based on beam-seach decoding strategy in self._get_decode_strategy
        """
        
        return dict(
           generation_config=GenerationConfig(
                do_sample=False,
                
                num_beams=num_beams,
                
                early_stopping=True,
                max_new_tokens=max_new_tokens,
                **kwargs
            ) 
        ) 
 
    def _get_sampling_decode_strategy(max_new_tokens, 
                                      temperature=0.01,
                                      top_p=0.95, top_k=50, 
                                      **kwargs) -> Dict:
        """Gets generation config based on sampling decoding strategy in self._get_decode_strategy
        """
        
        return dict(
           generation_config=GenerationConfig(
                do_sample=True,
                
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                
                early_stopping=True,
                max_new_tokens=max_new_tokens,
                **kwargs
            ) 
        ) 

    def _get_contrastive_decode_strategy(max_new_tokens,
                                        top_k=4,
                                        penalty_alpha=0.6,
                                        **kwargs) -> Dict:
        """Gets generation config based on contrastive decoding strategy in self._get_decode_strategy
        """
        
        return dict(
           generation_config=GenerationConfig(
                do_sample=True,
                
                top_k=top_k,
                penalty_alpha=penalty_alpha,
                
                early_stopping=True,
                max_new_tokens=max_new_tokens,
                **kwargs
            ) 
        )
       
    def _get_abcd_decode_strategy(max_new_tokens,
                                  tokenizer=None,
                                  choices=['A', 'B', 'C', 'D'],
                                  **kwargs
                                  ) -> Dict:
        """Gets generation config based on ABCD decoding strategy in self._get_decode_strategy
        ABCD decoding strategy is specific to MCQA task, where the model only outputs the next one token(so the arg max_new_tokens is not used), 
        which is choosen from a bunch of option letters with the max logit
        """
        class ABCDLogitsProcessor(LogitsProcessor):
            def __init__(self, tokenizer, choices=None):
                self.choices = ['A', 'B', 'C', 'D'] if choices is None else choices
                self.token_ids = [tokenizer.convert_tokens_to_ids([t])[0] for t in self.choices]
                
            def __call__(self, input_ids, logits):
                _, vocab_size = logits.shape # logits.shape = (batch_size, vocab_size)
                filter_idxs = [i for i in range(vocab_size) if i not in self.token_ids]
                logits[:, filter_idxs] = -torch.inf
                    
                return logits
        
        if not tokenizer:
            raise ValueError("No tokenizer given! The ABCD decoding strategy requires one tokenizer to be provided.")    
        
        processors = LogitsProcessorList()
        abcd_logits_processor = ABCDLogitsProcessor(tokenizer=tokenizer, choices=choices)
        processors.append(abcd_logits_processor)
        
        return dict(
           generation_config=GenerationConfig(
                do_sample=False,
                num_beam=1,
    
                max_new_tokens=1,
                **kwargs
            ),
            logits_processor=processors,
        )
        
        
class DecodingStrategy(object):

    SUPPORT_STRATEGIES = {
            'greedy': {
                'strategy': DecodingStrategyFuncs._get_greedy_decode_strategy,
                'args': dict()
            },
            'beam': {
                'strategy': DecodingStrategyFuncs._get_beam_decode_strategy,
                'args': dict(
                    num_beams=5,
                )
            },
            'sampling': {
                'strategy': DecodingStrategyFuncs._get_sampling_decode_strategy,
                'args': dict(
                    top_p=0.95,
                    top_k=50,
                    temperature=0.01,
                )
            },
            'contrastive': {
                'strategy': DecodingStrategyFuncs._get_contrastive_decode_strategy,
                'args': dict(
                    top_k=4,
                    penalty_alpha=0.6,
                )
            },
            'abcd': {
                'strategy': DecodingStrategyFuncs._get_abcd_decode_strategy,
                'args': dict(
                    tokenizer=None,
                    choices=['A', 'B', 'C', 'D']
                )
            }
        }