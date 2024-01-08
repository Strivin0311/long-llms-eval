############################       base libs import     ############################

import os
import argparse
from typing import List, Union, Tuple, Dict, Optional, Any, Iterator

############################       dev libs import     ############################

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_dataset_builder, DownloadMode
from evaluate import load, combine

############################       own libs import     ############################

from base import BaseModule
from utils import info_str, info_list, info_dict, time_manager
from utils import get_split_len_strs_and_conds, get_cond_idxs
from config import DatasetConfig


############################    dataset class    ############################


class LCWDataset(BaseModule):
    """The base dataset class for LCW dataset, which load the dataset and corresponding metrics in huggingface-style,
    with batched iteration func to get the dataset samples 
    """
    def __init__(self,  dataset_config, args) -> None:
        super().__init__()
        
        # init attributes
        self.args = args
        for k, v in dataset_config.get_config_dict().items(): setattr(self, k, v)
        
        # load dataset and the corresponding metric
        self._load()
        
    @property
    def num_samples(self):
        """Get the number of samples in the dataset as an dynamic attribute
        """
        return len(self.dataset)
        
    def _load_dataset(self) -> None:
        """Load the dataset, which is called in self._load during __init__
        """
        if self.dataset_path == '':
            # load from the huggingface
            self.dataset = load_dataset(*self.dataset_name, split=self.split)
        else:
            # load from the local with the loading script
            builder = load_dataset_builder(self.dataset_path, *self.dataset_name[1:])
            builder.download_and_prepare(download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
            self.dataset = builder.as_dataset(split=self.split)
        
    def _load_few_shot_dataset(self) -> None:
        """Load the few-shot dataset, 
        ofen from a different split of the same datset
        if self.few_shot = 0, then run zero-shot mode and few shot dataset is None
        """
        if self.few_shot > 0:
            if self.dataset_path == '':
                # load from the huggingface
                self.fewshot_dataset = load_dataset(*self.dataset_name, split=self.few_shot_split)
            else:
                # load from the local with the loading script
                builder = load_dataset_builder(self.dataset_path, *self.dataset_name[1:])
                builder.download_and_prepare(download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
                self.fewshot_dataset = builder.as_dataset(split=self.few_shot_split) 
                
            if len(self.fewshot_dataset) < self.few_shot or (
                self.split == self.few_shot_split and len(self.fewshot_dataset) <= self.few_shot
            ):
                raise ValueError(f"The few-shot dataset has only {len(self.fewshot_dataset)} samples, " + 
                                 f"not fit to {self.few_shot}-shot setting " + 
                                 f"with the same split {self.split}" if self.split == self.few_shot_split else ""
                                 )
                                 
        else: self.fewshot_dataset = None
        
    def _load_metric(self) -> None:
        """Load the metric, which is called in self._load during __init__
        """
        if self.metrics != []:
            # load from the huggingface
            if len(self.metrics) == 1:
                self.metric = load(self.metrics[0])
            else:
                self.metric = combine(self.metrics)
        else:
            # load the metric corresponding with the dataset
            if self.dataset_path == '':
                # loading from huggingface
                self.metric = load(*self.dataset_name)
            else:
                # load from the local with the loading script
                self.metric = load(path=self.dataset_path, config_name=self.dataset_name[1]) 
        
    def _load_prompt_template(self) -> None:
        """Load the prompt template, which is called in self._load during __init__
        """
        if self.prompt_template == '':
            print(info_str("Warning: the prompt_template is not set"))
        pass
    
    def _load_max_new_tokens(self) -> None:
        """Load the max new tokens, which is called in self._load during __init__
        """
        if self.max_new_tokens == 0:
            print(info_str("Warning: the max_new_tokens is not set"))
        pass
        
    def _load_few_shot_prefix_suffix(self) -> None:
        """Load the few_shot_prefix / few_shot_suffix, which is called in self._load during __init__
        which should be set either by user, or by the subclasses
        """
        if self.few_shot_prefix == "":
            self.few_shot_prefix = "请阅读以下若干任务示例：\n\n"
        if self.few_shot_suffix == "":
            self.few_shot_suffix = "以下给出真正的任务：\n\n"
        
    def _load(self) -> None:
        """Load the dataset and corresponding metric
        with the prompt template and max new tokens
        """
        ### load dataset
        # verbose info
        if self.args.verbose:
            print(info_str(
                f"Loading dataset {info_list(self.dataset_name)} from {self.dataset_path if self.dataset_path != '' else 'HuggingFace'}"
            ))
        # load
        self._load_dataset()
         # verbose info
        if self.args.verbose:
            print(info_str(f"There are {self.num_samples} samples in the {self.split} split of the dataset with {len(self)} batches"))
            if hasattr(self.dataset, 'info'): 
                print(info_str(f"The info of the dataset {info_list(self.dataset_name)} is as below: \n {self.dataset.info}\n"))
            else:
                print(info_str(f"The info of the dataset {info_list(self.dataset_name)} is as below: \n {self.dataset}\n"))
        
        ### load few-shot dataset
        if self.args.verbose:
            print(info_str(f"Loading {self.few_shot}-shot examples from the {self.few_shot_split} split of the dataset"))
        self._load_few_shot_dataset()
        
        ### load few-shot prefix
        self._load_few_shot_prefix_suffix()
        
        ### load metric
        # verbose info
        if self.args.verbose:
            if self.metrics != []: 
                print(info_str(f"Loading metric {info_list(self.metrics)} from HuggingFace"))
            else:
                print(info_str(f"Loading metric corresponding to the dataset"))
        # load
        self._load_metric()
        # verbose info
        if self.args.verbose:
            if hasattr(self.metric, 'info'):
                print(info_str(f"The info of the metric is as below: \n{self.metric.info}\n")) 
            else:
                print(info_str(f"The info of the metric is as below: \n{self.metric}\n")) 

        ### load prompt template
        self._load_prompt_template()
        if self.args.verbose:
            print(info_str(f"The prompt template is as below: \n{self.prompt_template}\n"))
        
        ### load max new tokens
        self._load_max_new_tokens()
        if self.args.verbose:
            print(info_str(f"The max new tokens is: {self.max_new_tokens}"))
        
    def _adjust_metric(self, score_output: dict) -> dict:
        """To adjust the metric order in score output 
        to let the main metric to be first metric, if it is set and there're multiple metrics
        """
        if self.main_metric != "":
            if self.main_metric in score_output:
                ordered_score_output = {self.main_metric: score_output[self.main_metric]}
                for k, v in score_output.items():
                    if k != self.main_metric:
                        ordered_score_output[k] = v
                return ordered_score_output
            else:
                raise KeyError(f"The main metric {self.main_metric} does not exist")
        return score_output    
    
    def _compute_speed(self, predictions: List[str], lengths: List[int], speeds: List[float]) -> float:
        """Compute the average speed for a bunch of predictions => unit: sec / word
        """
        total_lengths = np.array([
            len(prediction) + prompt_length
            for prediction, prompt_length in zip(predictions, lengths)
        ])
        avg_speed = (speeds * total_lengths).sum() / total_lengths.sum()
        
        return avg_speed
    
    def _compute_ppl(self, lengths: List[int], ppls: List[float]) -> float:
        """Compute the average perplexity for a bunch of inputs
        """
        avg_ppl = (ppls * lengths).sum() / lengths.sum()
        
        return avg_ppl
    
    def _compute(self, predictions: List[str], references: List) -> Dict:
        """Score the predictions with the references, 
        and if the main_metric is set, adjust the metric key order in score output
        """
        score_output = self.metric.compute(predictions=predictions, references=references)
        score_output = self._adjust_metric(score_output)
        return score_output

    def forward(self, pred_outputs: List[Dict], split: bool = True) -> Dict:
        """score the predictions from certain model, with corresponding refenreces for each split length range and the all length range
        Param:
            pred_outputs: [
                {
                    prediction: "", 
                    reference: [""] or "", 
                    prompt_length: n,
                    speed: (sec/word)
                }
            ]
        Return:
            score_outputs: {
                'all': {...'speed': },
                '1k~2k': {...'speed': },
                ...
                '16k+': {...'speed': },
            }
        """
        ## get the params
        predictions = np.array([po[self.prediction_key] for po in pred_outputs], dtype=object)
        references = np.array([po[self.reference_key] for po in pred_outputs], dtype=object)
        lengths = np.array([po[self.length_key] for po in pred_outputs])
        speeds = np.array([po[self.speed_key] for po in pred_outputs])
        if self.ppl_metric.lower() == 'true': ppls = np.array([po[self.ppl_key] for po in pred_outputs])

        ## get the samples outputs for each length range and all length range           
        # init the score outputs
        score_outputs, split_idxs = {}, {'all': range(len(lengths)), }
        # init the split strings and conditions
        self.length_splits_str, self.length_splits_cond = get_split_len_strs_and_conds(self.length_splits)
        # get the split idxs
        if split:
            for split_str, split_cond in zip(self.length_splits_str, self.length_splits_cond):
                idxs = get_cond_idxs(lengths, split_cond)
                if idxs is not None and len(idxs) > 0:
                    split_idxs[split_str] = idxs
        # score for each split
        for split_str, split_idx in split_idxs.items():
            # compute the metric scores
            score_output = self._compute(
                predictions=predictions[split_idx],
                references=references[split_idx]
            )
            # add optional ppl score
            if self.ppl_metric.lower() == 'true':
                score_output[self.ppl_key] = self._compute_ppl(
                    lengths=lengths[split_idx],
                    ppls=ppls[split_idx]
                )
            # add inference speed
            score_output[self.speed_key] = self._compute_speed( # compute the average speed for this split
                predictions=predictions[split_idx],
                lengths=lengths[split_idx],
                speeds=speeds[split_idx]
            )
            # add num of samples
            score_output[self.num_samples_key] = len(predictions[split_idx])
            # append this split to outputs dict
            score_outputs[split_str] = score_output
            
        if self.args.verbose:
            print(info_str(f"The score outputs for the prediction outputs: \n{info_dict(score_outputs)}\n"))
            
        return score_outputs
        
    def _get_few_shot_example(self, input_idx) -> str:
        """Sampling from the few-shot dataset and get the few shot example strings as the predix for the original input
        NOTE:   
            if the few-shot dataset and the dataset shares the same split, 
            then the few-shot example idxs should avoid the input's idx
        """
        # zero-shot setting
        if self.few_shot == 0: return "" 
        
        fewshot = f"{self.few_shot_prefix}\n"
        if self.split == self.few_shot_split: # avoid the input idx
            rng = [i for i in range(len(self.fewshot_dataset)) if i != input_idx]
        else:
            rng = range(len(self.fewshot_dataset))
        idxs = np.random.choice(rng, size=self.few_shot, replace=False).tolist()
        
        for idx in idxs:
            input, ref, _, _ = self._get_single_sample(self.fewshot_dataset[idx])
            fewshot += f"{input}\n{ref}\n\n"
            
        fewshot += f"{self.few_shot_suffix}\n\n"
        
        return fewshot # if self.few_shot == 0, then use zero-shot mode, i.e. empty few-shot string
        
    def _get_single_sample(self, sample: dict) -> Tuple[str, str, int, dict]:
        """Get single input, ref, length, and some info from one sample
        NOTE: which should be specific to different dataset sample design in subclass
        """
        input, ref = sample['input'], sample['ref'] # default input key and ref key, which may not be compatible with the specific dataset
        length, info = sample['length'], sample['info']
        return input, ref, length, info
        
    def __iter__(self) -> Iterator:
        """Iterate the dataset batch samples 
        """
        start_idx = 0
        while start_idx < len(self.dataset):
            inputs, refs, lengths, infos = [], [], [], []
            end_idx = start_idx + self.batch_size
            # generate the batch samples(inputs and refs) from the dataset in the range [start_idx, end_idx]
            for idx in range(start_idx, min(end_idx, len(self.dataset))):
                fewshot = self._get_few_shot_example(idx)
                input, ref, length, info = self._get_single_sample(self.dataset[idx])
                inputs.append(fewshot + input) # NOTE: do truncation here
                refs.append(ref)
                lengths.append(length)
                infos.append(info)
            yield inputs, refs, lengths, infos
            start_idx = end_idx
            
    def __len__(self) -> int:
        """Get the length of the dataset, i.e. the num of batchs in the dataset
        if num of samples is needed, use self.num_samples
        """
        return int(np.ceil(len(self.dataset) / self.batch_size))


class CLoTLCWDataset(LCWDataset):
    """The LCW dataset class for CLoT benchmark"""
    def __init__(self, dataset_config, args) -> None:
        super().__init__(dataset_config, args)
        
        self.non_info_keys = ['context', 'question', 'answers', 'length']
        self.sample_len_key = 'length'
        self.sample_ref_key = 'answers'
        self.meta_prompt_template = "User: {prompt}\n\nAssistant:"
            
    def _load_prompt_template(self) -> None:
        if self.prompt_template == '':
            builder = load_dataset_builder(self.dataset_path, *self.dataset_name[1:])
            self.prompt_template =  builder.config.prompt_template
            
    def _load_max_new_tokens(self) -> None:
        if self.max_new_tokens == 0:
            builder = load_dataset_builder(self.dataset_path, *self.dataset_name[1:])
            self.max_new_tokens =  builder.config.max_new_tokens
        
    def _get_single_sample(self, sample: dict) -> Tuple[str, str, int, dict]:
        """Get single input and ref from one sample
        NOTE: which should be specific to different dataset sample design in subclass
        """
        prompt = self.prompt_template.format(**sample)
        if vars(self).get('use_meta_prompt_template', 'true').lower() == "true":
            input = self.meta_prompt_template.format(prompt=prompt)
        else:
            input = prompt
        ref = sample[self.sample_ref_key]
        length = sample[self.sample_len_key]
        info = {}
        for k, v in sample.items():
            if k not in self.non_info_keys:
                info[k] = v
        return input, ref, length, info
      
                
class LongBenchLCWDataset(LCWDataset):
    """tHE LCW dataset class for LongBench benchmark"""
    def __init__(self, dataset_config, args) -> None:
        
        ### additional config for LongBench before init
        
        # prompt template for each subtask dataset
        dataset_config.dataset2prompt = {
            "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
            "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
            "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
            "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
            "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
        }
       
        # all classes for each dataset (if not classification task, then None)
        dataset_config.all_classes = {
            'lsht': ["农业、农村", "军事", "文学、艺术", "体育", "传媒业", 
                     "电子信息产业", "文化、休闲娱乐", "社会、劳动", "经济", 
                     "服务业、旅游业", "环境、气象", "能源、水务、水利", "财政、金融", 
                     "教育", "科学技术", "对外关系、国际关系", "矿业、工业", "政治", 
                     "交通运输、邮政、物流", "灾难、事故", "基本建设、建筑业、房地产", 
                     "医药、卫生", "法律、司法", "商业、外贸、海关"],
            'dureader': None,
            'vcsum': None,
            "multifieldqa_zh": None,
            "passage_retrieval_zh": None, 
        }

        # max-length for each sub dataset
        dataset_config.dataset2maxlen = {
            "lsht": 64,
            "dureader": 128,
            "vcsum": 512,
            "passage_retrieval_zh": 32,
            "multifieldqa_zh": 64,
        }
        
        # metric for each sub dataset
        dataset_config.dataset2metric = {
            "dureader": LongBenchLCWDataset.RougeScoreZH(),
            "multifieldqa_zh": LongBenchLCWDataset.QAF1ScoreZH(),
            "vcsum": LongBenchLCWDataset.RougeScoreZH(),
            "lsht": LongBenchLCWDataset.ClassificationScoreZH(all_classes=dataset_config.all_classes['lsht']),
            "passage_retrieval_zh": LongBenchLCWDataset.RetrievalScoreZH(),
        }
        
        super().__init__(dataset_config, args)
        
    def _load_metric(self) -> None:
        """Load the metric, which is called in self._load during __init__
        """
        subtask_name = self.dataset_name[1]
        self.metric = self.dataset2metric[subtask_name] 
        
    def _load_dataset(self) -> None:
        """Load the LongBench dataset, which is called in self._load during __init__
        """
        if self.dataset_path == '':
            # load from the huggingface
            self.dataset = load_dataset(*self.dataset_name, split=self.split)
        else:
            # load from the local with the loading script
            builder = load_dataset_builder(self.dataset_path, *self.dataset_name[1:],
                                           # NOTE: additional config for LongBench
                                           split_size=vars(self).get('split_size', 'small'),
                                           split_version=vars(self).get('split_version','v1.0')
                                           )
            builder.download_and_prepare(
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
            )
            self.dataset = builder.as_dataset(split=self.split)
        
    def _load_prompt_template(self) -> None:
        subtask_name = self.dataset_name[1]
        if self.prompt_template == '':
            self.prompt_template = self.dataset2prompt[subtask_name]
            
    def _load_max_new_tokens(self) -> None:
        subtask_name = self.dataset_name[1]
        if self.max_new_tokens == 0:
            self.max_new_tokens = self.dataset2maxlen[subtask_name]
        
    def _get_single_sample(self, sample) -> Tuple[str, str, int, dict]:
        """Get single input and ref from one sample
        NOTE: which should be specific to different dataset sample design in subclass
        """
        input = self.prompt_template.format(**sample)
        ref = sample['answers']
        length = sample['length']
        info = {}
        
        return input, ref, length, info

    @staticmethod
    def normalize_zh_answer(s):
        """Lower text and remove punctuation, extra whitespace."""
        import string
        def white_space_fix(text):
            return "".join(text.split())

        def remove_punc(text):
            cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
            all_punctuation = set(string.punctuation + cn_punctuation)
            return "".join(ch for ch in text if ch not in all_punctuation)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))
   
    @staticmethod 
    def rouge_score(prediction, ground_truth, m='f', **kwargs) -> dict:
        """
        m = p: precision | r: recall | f: f1
        """
        from rouge import Rouge
        rouger = Rouge()
        try:
            scores = rouger.get_scores([prediction], [ground_truth], avg=True)
        except:
            return {
                f'Rouge-1 ({m})': 0.0,
                f'Rouge-2 ({m})': 0.0,
                f'Rouge-L ({m})': 0.0,
            }
        
        return {
            f'Rouge-1 ({m})': scores["rouge-1"][m],
            f'Rouge-2 ({m})': scores['rouge-2'][m],
            f'Rouge-L ({m})': scores['rouge-l'][m],
        }

    @staticmethod
    def f1_score(prediction, ground_truth, **kwargs):
        from collections import Counter
        common = Counter(prediction) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0: return {'F1': 0.}
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return {'F1': f1}

    class LongBenchScore(object):
        def compute(self, predictions, references, **kwargs):
            from collections import defaultdict
            total_score = defaultdict(float)
            for (prediction, ground_truths) in zip(predictions, references):
                score = {}
                for ground_truth in ground_truths:
                    score_out = self._compute_single(prediction, ground_truth, **kwargs)
                    for score_type, score_value in score_out.items():
                        score[score_type] = max(score_value, score.get(score_type, -np.inf))
                for score_type, score_value in score.items():
                    total_score[score_type] += score_value
            for score_type in total_score:
                total_score[score_type] =  round(total_score[score_type] / len(predictions), 2) if len(predictions) > 0. else -1.0
            
            return dict(total_score)

        def _compute_single(self, prediction, ground_truth, **kwargs):
            """No Implementation in base class"""
            return {}
        
    class RougeScoreZH(LongBenchScore):
        def _compute_single(self, prediction, ground_truth, **kwargs):
            import jieba
            prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
            ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
            return LongBenchLCWDataset.rouge_score(prediction, ground_truth)

    class QAF1ScoreZH(LongBenchScore):
        def _compute_single(self, prediction, ground_truth, **kwargs):
            import jieba
            prediction_tokens = list(jieba.cut(prediction, cut_all=False))
            ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
            prediction_tokens = [LongBenchLCWDataset.normalize_zh_answer(token) for token in prediction_tokens]
            ground_truth_tokens = [LongBenchLCWDataset.normalize_zh_answer(token) for token in ground_truth_tokens]
            prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
            ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
            return LongBenchLCWDataset.f1_score(prediction_tokens, ground_truth_tokens)

    class ClassificationScoreZH(LongBenchScore):
        
        def __init__(self, all_classes=[]) -> None:
            super().__init__()
            self.all_classes = all_classes
        
        def _compute_single(self, prediction, ground_truth, **kwargs):
            import difflib
            em_match_list = []
            all_classes = self.all_classes
            
            for class_name in all_classes:
                if class_name in prediction:
                    em_match_list.append(class_name)
            for match_term in em_match_list:
                if match_term in ground_truth and match_term != ground_truth:
                    em_match_list.remove(match_term)
            if em_match_list != 0:
                if ground_truth in em_match_list:
                    score = (1.0 / len(em_match_list))
                else:
                    score = 0.0
            else:
                best_match = None
                highest_similarity = 0
                for string in all_classes:
                    similarity = difflib.SequenceMatcher(None, string, prediction).ratio()
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = string
                score = float(best_match == ground_truth)
            return {'accuracy': score}
            
    class RetrievalScoreZH(LongBenchScore):
        def _compute_single(self, prediction, ground_truth, **kwargs):
            import re
            pattern = r'段落(\d+)'
            matches = re.findall(pattern, ground_truth)
            ground_truth_id = matches[0]
            numbers = re.findall(r"\d+", prediction)
            right_num = 0
            for number in numbers:
                if str(number) == str(ground_truth_id):
                    right_num += 1
            final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
            return {'accuracy': float(final_score)}


############################    dataset registry    ############################


dataset_registry = {
    'clot': CLoTLCWDataset,
    'long_bench': LongBenchLCWDataset,
}


############################       load func     ###############################


def load_dataset(dataset_config, args):
    dataset_class = dataset_config.register_class
    if dataset_class not in dataset_registry:
        raise KeyError(f'The class {dataset_class} is not registered yet')
    
    dataset = dataset_registry[dataset_class](dataset_config, args)
    return dataset


############################       main func     ############################### 


def main(args):
    args.verbose = True
    dataset_config = DatasetConfig.load(args.config_path)
    lcw_dataset = load_dataset(dataset_config, args)
    inputs, refs = next(iter(lcw_dataset))
    idx = np.random.randint(0, len(inputs))
    print(info_str(f"Here's a random example from the dataset: \n" +
                   f"input: {inputs[idx]} \n" +
                   f"reference: {refs[idx]}\n"
    ))
    
    user_input = input("\nPlease input your answer for this example to take a look at the metrics: \n")
    score_output = lcw_dataset([{
        lcw_dataset.prediction_key: user_input,
        lcw_dataset.reference_key: refs[idx],
        lcw_dataset.length_key: len(inputs[idx]) + len(user_input)
    }])
    
    print(info_str(f"Here's your score output: \n{score_output}\n"))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This the script for taking a first quick look at some LCWDataset subclass, \
                                     espeicially when a new dataset class is built and haven't been tested with the evaluator yet"
                                     )
    parser.add_argument("--config-path", type=str, default="", help="The dataset config json file path")
    args = parser.parse_args()
    main(args)
