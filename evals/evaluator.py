############################       base libs import     ############################

import os
import time
import json
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Any

############################       dev libs import     ############################

import torch
from transformers import logging
logging.set_verbosity_error()

############################       own libs import     ############################

from base import BaseModule
from utils import info_ordinal, info_list, info_str, memory_used
from utils import time_manager
from model import load_model
from dataset import load_dataset
from report import Reporter

############################     evaluator class    ############################


class Evaluator(BaseModule):
    def __init__(self, eval_config=None, args=None) -> None:
        super().__init__()
        
        self.args = args
        self.eval_config = eval_config
        if eval_config is not None:
            self.model_configs = eval_config.model_configs
            self.dataset_configs  = eval_config.dataset_configs
        else:
            print(info_str(f"No config has been set"))
        self.outputs = {}
        
        # init output dir
        if eval_config is not None and args is not None:
            self.output_dir = os.path.join(self.args.output_dir, self.eval_config.exp_name)
      
        # init reporter
        self.reporter = Reporter()

    @time_manager("Evalution")           
    def forward(self) -> None:
        """Evaluate each model with each dataset, and save the output with some visualized reports of the results to disk."""
        args = self.args
        
        for mi, model_config in enumerate(self.model_configs):
            model_name = model_config.model_name
            print(info_str(f"Evaluating the {info_ordinal(mi+1)} model: {model_name} on each dataset"))
            # init lcw-aug model (loading the pretrained model with the corresponding/self-defined tokenizer, maybe adding some aug methods to be evaluated)
            model = load_model(model_config, args)
            # evaluate the model on each dataset
            for di, dataset_config in enumerate(self.dataset_configs):
                dataset_name = info_list(dataset_config.dataset_name)
                print(info_str(f"Evaluating {model_config.model_name} on the {info_ordinal(di+1)} dataset {dataset_name}"))
                # init lcw dataset (loading the hugging-face style dataset with the corresponding/self-defined metric)
                dataset = load_dataset(dataset_config, args)
                # evaluate this dataset on the current model
                try:
                    self._evaluate_single(model, model_name, dataset, dataset_name)
                except RuntimeError as e:
                    self.reporter.log_error(e, model_name, dataset_name)
                    if "CUDA out of memory" in str(e):
                        print(info_str(f"ERROR: GPU OOM detected when evaluating: \n" + 
                                       f"model {model_name} on dataset {dataset_name}\n" + 
                                       f"Skip this evaluation and continue ...\n",
                                       side_str="!"))
                        self.reset_evaulate()
                        continue
                    else: # report the outputs before the error happened
                        self.report()
                        raise
            # reset the env for next model
            self.reset_evaulate()
            print("\n")
            
        self.report()
    
    def _evaluate_single(self,  model, model_name, dataset, dataset_name) -> None:
        """Evaluate a single dataset on the specified model, and save the output of the results to disk."""
        # evaluate the current model on this dataset
        output = {
            'model_name': model_name, 'dataset_name': dataset_name,
            self.pred_output_key: [], self.score_output_key: {},
        }
        max_new_tokens =  dataset.max_new_tokens
        # prediction
        with tqdm(enumerate(dataset), total=len(dataset)) as pbar:
            pred_sample_num = 0
            for bi, batch in pbar:
                # update model prediction output for current batch
                try:
                    pred_output = model(*batch, 
                                        max_new_tokens=max_new_tokens, 
                                        ppl=dataset.ppl_metric.lower() == 'true',
                                        start_idx=pred_sample_num,
                                        )
                except RuntimeError as e:
                    self.reporter.log_error(e, model_name, dataset_name, pred_sample_num+1)
                    if "CUDA out of memory" in str(e):
                        print(info_str(f"\nERROR: GPU OOM detected when evaluating: \n" + 
                                       f"model {model_name} on the {info_ordinal(pred_sample_num+1)} sample in dataset {dataset_name}\n", 
                                       f"Skip this sample and continue ...\n",
                                       side_str="!"))
                        self.reset_predict()
                        continue
                    else: raise
                    
                # evaluate the single prediction and put the result into pred output for each sample
                for po in pred_output: po[self.single_score_output_key] = dataset([po], split=False)
                # extend to the all samples output
                output[self.pred_output_key].extend(pred_output)
                
                # update the logging message for current batch
                pred_sample_num = min((bi+1) * dataset.batch_size, dataset.num_samples)
                pbar.set_description(f"[GPU Memory Used: total: {memory_used()} | max: {memory_used(reduce='max')}]")
                
                # pbar.update(1) # NOTE: do not use update here, cuz the parallelism will make the update faster than it should be
                pbar.n = bi+1
                pbar.refresh()
                
                # reset prediction
                self.reset_predict()
            
        # scoring
        print(info_str("Scoring the predictions with the references"))
        score_output = dataset(output[self.pred_output_key])
        output[self.score_output_key] = score_output
        # save the single output of the evalution results for the current model on this dataset
        self.save(output)
        print(info_str(f"Done Scoring! Saved the output into the {self.output_dir}"))
        # add this output to outputs
        for output_key in (self.pred_output_key, self.score_output_key):
            self.outputs.setdefault(dataset_name, {}).setdefault(model_name, {})[output_key] = output[output_key]   
         
    def save(self, output: dict):
        """Save the self.output for specific certain model and certain dataset"""
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        
        model_name, dataset_name = output['model_name'], output['dataset_name']
        output_path = os.path.join(self.output_dir, 
                                   f"output_model[{model_name}]_dataset[{dataset_name}].json")
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False)
        
    @time_manager("Report")
    def report(self, output_dir: Optional[str] = None):
        """
            Report some visualized files from the self.output 
            or all the output_xxx.json files from the argument 'output_dir',
            while generating the complete output.json, to the args.output_dir for this experiment
            :param: remove: if True, remove the based output_xxx.json in the 'output_dir' after reporting
            
            Args:
                outputs (dict): {
                    dataset_name: {
                        model_name: {
                                pred_output: [ {} ], 
                                score_output: {
                                    'all': {'acc': 0.5, ...},
                                    '<1k': {'acc': 0.5, ...},
                                    '1k~2k': {'acc': 0.5, ...},
                                    '2k+': {'acc': 0.5, ...},
                                }
                        } 
                    } 
                }
        """
        # init output dir
        if output_dir is not None:
            outputs = self.reporter.collect_report(output_dir)
            output_dir = os.path.join(output_dir, f"exp-time[{time.strftime('%Y-%m-%d %H:%M')}]")
        else:
            outputs = self.outputs
            output_dir = self.output_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir)
            
        # report 1: output.json
        print(info_str(f"Reporting output")) 
        output_path = os.path.join(output_dir, f"output.json")
        self.reporter.report_outputs(outputs, output_path)
            
        # report 2: leaderboard.md
        print(info_str(f"Reporting leaderboard")) 
        leaderboard_path = os.path.join(output_dir, f"leaderboard.md")
        self.reporter.report_leaderboard(outputs, leaderboard_path)
            
        # report 3: plot.png
        print(info_str(f"Reporting length plot")) 
        plot_path = os.path.join(output_dir, f"plot.png")
        self.reporter.report_lengthplot(outputs, plot_path)
        
        # report 4: optional radar.png
        print(info_str(f"Reporting radar chart"))
        radarchart_path = os.path.join(output_dir, f"radarchart.png") 
        self.reporter.report_radarchart(outputs, radarchart_path)
        
        # report 5: report error
        print(info_str(f"Reporting error"))
        error_path = os.path.join(output_dir, f"error.json")
        self.reporter.report_error(error_path)
                  
    def reset_evaulate(self):
        """reset the environment for next model"""
        torch.cuda.empty_cache() # empty the gpu cache to save space for next model
        
    def reset_predict(self):
        """reset the environment for next prediction"""
        torch.cuda.empty_cache() # empty the gpu cache to save space for next prediction
        

############################    evaluator registry    ############################


evaluator_registry = {
    'base_evaluator': Evaluator,
}


############################       load func     #################################


def load_evaluator(evaluator_config, args):
    evaluator_class = evaluator_config.register_class
    if evaluator_class not in evaluator_registry:
        raise KeyError(f'The class {evaluator_class} is not registered yet')
    
    evaluator = evaluator_registry[evaluator_class](evaluator_config, args)
    return evaluator 
    