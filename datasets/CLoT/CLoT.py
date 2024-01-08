import os
import argparse
import json 
from enum import Enum
import datasets
from datasets import Split, Version, Features, Value, Sequence
from datasets import GeneratorBasedBuilder, BuilderConfig, DatasetInfo, SplitGenerator
import evaluate


############################     basic infos    ############################ 


_VERSION = '2.0.0' # this is the 2.0.0 updated version of CLoT, which adds many new datasets, unifies the sample format and is aware of long-short samples to test the model's ability to handle long contexts while keeping the ability to handle shorter ones

_DESCRIPTION = "CLoT is a bunchmark dataset to test LLMs' abilities on a bunch of Chinese Long-Text NLU and NLG subtasks" # brief introduction to dataset class

_METRIC_DESCRIPTION = "This is the metric class for the benchmark CLoT, which access the metrics for each subtask datasets" # brief introduction to metric class

_MCQA_METRIC_INPUT_DESCRIPTION = "The metric input format for MCQA subtask is the prediction and reference(s), both of which are option strings for the question, where the option string can be only a single letter 'A', 'B', or a digital number '5', '12' "

_ExtQA_METRIC_INPUT_DESCRIPTION = "The metric input format for ExtQA subtask is the prediction and reference(s), both of which are free-form strings for the question"

_Summ_METRIC_INPUT_DESCRIPTION = "The metric input format for Summ subtask is the prediction and reference(s), both of which are summative texts for the contents"


_HOMEPAGE = '' # homepage of the dataset for documentation

_LICENSE = '' # license for dataset if available

_CITATION = '' # citation for the dataset


############################     basic configs    ############################ 


_BASE_FEATURES = Features({ # the base features design
    'context': Value('string'),
    'question': Value('string'),
    'answers': Sequence(Value("string")),
    'base': Value('string'),
    'length': Value('int32'),
    'difficulty': Value('string')
})

_THUCNEWSEXTQA_FEATURES = Features({ # the base features design
    'context': Value('string'),
    'question': Value('string'),
    'answers': Sequence(Value("string")),
    'base': Value('string'),
    'length': Value('int32'),
    'difficulty': Value('string'),
    'category': Value('string')
})

_META_PROMPT_TEMPLATE = "User: {prompt}\n\nAssistant:" # assistant style prompt

_BASE_PROMPT_TEMPLATE = '请阅读以下材料：\n\n{prompt}\n\n请根据以上材料回答所提出的问题：\n\n{question}\n\n'

_BASE_METRIC_FEATURES = Features({ # the base metric features which compares two texts
    'predictions': Value("string"),
    'references': Sequence(Value("string")), 
})

_BASE_FILE_FORMAT = 'jsonl'

_BASE_SPLITS = [ # besides the standard train/val/test splits, here adds a dev split to offer a few elaborated examples for quick browse and few-shot learning
    'train', 
    'val', 
    'test', 
    'dev'
]

_BASE_TEST_SPLIT = 'test' # the base split name when running the testing split


############################     metric infos    ############################ 


class Metask(Enum):
    """The meta-task enum type in CLoT benchmark"""
    MCQA = 'MCQA' # Multi-Choice QA
    ExtQA = 'ExtQA' # Extractive QA
    Summ = 'Summ' # Summarization
    OpenW = 'OpenW' # Open Writing
    Others = 'Others' # Others


class MetricType(Enum):
    """The metric type in CLoT benchmark"""
    MCQA_ABCD = 'MCQA-ABCD' # Multi-Choice QA with answer type like: 'A', 'ABC’
    MCQA_NUMBER = 'MCQA-NUMBER' # Multi-Choice QA with answer type like: '1', ’8‘，’14‘
    ExtQA = 'ExtQA-metric' # Retrieval QA
    Summ = 'Summ-metric' # Summarization
    OpenW = 'OpenW-metric' # Open Writing


############################     subtask infos    ############################ 


# all of the needed subtask datasets info to create configs, split, generate, and so on
_SUBTASK_INFOS = { 
    #########       MCQA      ######### 
    'PlotPatcher': {
        'metask': Metask.MCQA,
        'subtask': 'Story Plot Cloze Test',
        'desc': 
            "This is a subtask dataset belonging to the MCQA metask in CLoT benchmark, " + \
            "which asks LLMs to select the right plot sentence from 4 candidates for one of a bunch of storys, " + \
            "each of which has only one <MASK> place to put the candidate plot sentence",
        'urls': '../datasets/CLoT/MCQA/PlotPatcher',
        
        'features': _BASE_FEATURES,
        'prompt_template': "这是一个单选题：\n\n请阅读以下若干则故事，其中每个故事里都有一段情节被<mask>掩盖：\n\n{context}基于以上故事，回答以下问题：\n\n{question}\n\n仅输出所选选项前[ ]内的字母即可，不必输出选项本身",
        'max_new_tokens': 32, # only one option letter is allowed, but leave 10 tokens to be more flexible
        
        'metric_desc': _MCQA_METRIC_INPUT_DESCRIPTION,
        'metric_features': _BASE_METRIC_FEATURES,
        'metric': MetricType.MCQA_ABCD,
        
        'splits': _BASE_SPLITS,
        'test_split': _BASE_TEST_SPLIT,
        'file_format': _BASE_FILE_FORMAT,
        'base': 'LOT',
    },
    'NewsCater': {
        'metask': Metask.MCQA,
        'subtask': 'Multi-News Categorization',
        'desc':
            "This is a subtask dataset belonging to the MCQA metask in CLoT benchmark, " + \
            "which asks LLMs to choose the right category that a bunch of news most probably belong to",
        'urls': '../datasets/CLoT/MCQA/NewsCater',
        
        'features': _BASE_FEATURES,
        'prompt_template': "这是一个单选题：\n\n请阅读以下若干则来自同一栏目的新闻：\n\n{context}请判断上述新闻最有可能出自下列哪个栏目：\n\n{question}\n\n仅输出所选选项前[ ]内的字母即可，不必输出选项本身",
        'max_new_tokens': 32, # only one option letter is allowed, but leave 10 tokens to be more flexible
        
        'metric_desc': _MCQA_METRIC_INPUT_DESCRIPTION,
        'metric_features': _BASE_METRIC_FEATURES,
        'metric': MetricType.MCQA_ABCD,
        
        'splits': _BASE_SPLITS,
        'test_split': _BASE_TEST_SPLIT,
        'file_format': _BASE_FILE_FORMAT,
        'base': 'LSHT(LongBench)', 
    },
    'THUCNewsMCQA': {
        'metask': Metask.MCQA,
        'subtask': 'Single Long News Categorization',
        'desc':
            "This is a subtask dataset belonging to the MCQA metask in CLoT benchmark, " + \
            "which asks LLMs to choose the right category that a long news most probably belongs to",
        'urls': '../datasets/CLoT/MCQA/THUCNewsMCQA',
        
        'features': _BASE_FEATURES,
        'prompt_template': "这是一个单选题：\n\n请阅读以下一段长新闻：\n\n{context}请从下列选项中选出上述新闻最有可能的类别：\n\n{question}\n\n仅输出所选选项前[ ]内的字母即可，不必输出选项本身",
        'max_new_tokens': 32, # only one option letter is allowed, but leave 10 tokens to be more flexible
        
        'metric_desc': _MCQA_METRIC_INPUT_DESCRIPTION,
        'metric_features': _BASE_METRIC_FEATURES,
        'metric': MetricType.MCQA_ABCD,
        
        'splits': _BASE_SPLITS,
        'test_split': _BASE_TEST_SPLIT,
        'file_format': _BASE_FILE_FORMAT,
        'base': 'THUCNews', 
    },
     
    #########       ExtQA      #########
    'MFReader': {
        'metask': Metask.ExtQA,
        'subtask': 'Multi/Single-Doc QA',
        'desc': 'This is a subtask dataset belonging to the ExtQA metask in CLoT benchmark, which asks to LLMs to retrieve the possible answers corresponding to the questions in the single/multi-documents',
        'urls': '../datasets/CLoT/ExtQA/MFReader',
       
        'features': _BASE_FEATURES,
        'prompt_template': "这是一个问答题：\n\n请阅读以下若干篇文档：\n\n{context}\n\n基于以上文档，请依次回答下列若干问题：\n\n{question}", 
        'max_new_tokens': 200, # free-form response
        
        'metric_desc': _ExtQA_METRIC_INPUT_DESCRIPTION,
        'metric_features': _BASE_METRIC_FEATURES,
        'metric': MetricType.ExtQA,
        
        'splits': _BASE_SPLITS,
        'test_split': _BASE_TEST_SPLIT,
        'file_format': _BASE_FILE_FORMAT,
        'base': 'MultifieldQA_zh(LongBench) + Dureader(robust)', 
    },
    'LongChatter': {
        'metask': Metask.ExtQA,
        'subtask': 'Timestamp-aware Chat History QA',
        'desc': 'This is a subtask dataset belonging to the ExtQA metask in CLoT benchmark, which asks to LLMs to retrieve the possible answers corresponding to the questions in a long chat history with specific timestamps',
        'urls': '../datasets/CLoT/ExtQA/LongChatter',
        
        'features': _BASE_FEATURES,
        'prompt_template': "这是一个问答题：\n\n请阅读以下若干天用户的聊天记录：\n\n{context}基于以上聊天记录，请依次回答下列若干问题：\n\n{question}",
        'max_new_tokens': 200, # free-form response
        
        'metric_desc': _ExtQA_METRIC_INPUT_DESCRIPTION,
        'metric_features': _BASE_METRIC_FEATURES,
        'metric': MetricType.ExtQA,
        
        'splits': _BASE_SPLITS,
        'test_split': _BASE_TEST_SPLIT,
        'file_format': _BASE_FILE_FORMAT,
        'base': 'C3(CLUE)', 
    },
    'THUCNewsExtQA': {
        'metask': Metask.ExtQA,
        'subtask': 'News Reading Comprehension QA',
        'desc': 'This is a subtask dataset belonging to the ExtQA metask in CLoT benchmark, which asks to LLMs to retrieve the possible answers corresponding to the questions in a long news',
        'urls': '../datasets/CLoT/ExtQA/THUCNewsExtQA',
        
        'features': _THUCNEWSEXTQA_FEATURES,
        'prompt_template': "这是一个问答题：\n\n请阅读以下新闻材料：\n\n{context}基于以上新闻材料，请依次回答下列若干问题：\n\n{question}",
        'max_new_tokens': 200, # free-form response
        
        'metric_desc': _ExtQA_METRIC_INPUT_DESCRIPTION,
        'metric_features': _BASE_METRIC_FEATURES,
        'metric': MetricType.ExtQA,
        
        'splits': _BASE_SPLITS,
        'test_split': _BASE_TEST_SPLIT,
        'file_format': _BASE_FILE_FORMAT,
        'base': 'THUCNews', 
    },
    
    
    # 'REDTag': {
    #     'metask': Metask.RetQA,
    #     'subtask': 'To generate the tag for an article in RED',  
    # },
    
    
    #########       Summ      #########
    'Summer': {
        'metask': Metask.Summ,
        'subtask': 'Multi-Doc Specific Summarization',
        'desc': 'This is a subtask dataset belonging to the Summ metask in CLoT benchmark, which asks to LLM to summarize the specific contents from one user or all the users, given a bunch of contents from different users',
        'urls': '../datasets/CLoT/Summ/Summer',
        
        'features': _BASE_FEATURES,
        'prompt_template': "这是一个摘要总结题：\n\n请阅读以下多个不同用户提供的内容：\n\n{context}基于以上内容，{question}", 
        'max_new_tokens': 512, # free-form summarization
        
        'metric_desc': _Summ_METRIC_INPUT_DESCRIPTION,
        'metric_features': _BASE_METRIC_FEATURES,
        'metric': MetricType.Summ,
        
        'splits': _BASE_SPLITS,
        'test_split': _BASE_TEST_SPLIT,
        'file_format': _BASE_FILE_FORMAT,
        'base': 'Passage_Retrieval_zh(LongBench) + VCSum(LongBench)',  
    }
    # 'VCSum': {
    #     'metask': Metask.Summ,
    #     'subtask': 'Meeting Summarization',
    #     'desc': 'This is a subtask dataset belonging to the Summ metask in CLoT benchmark, which asks to LLM to summarize the long meeting contents',
    #     'urls': '../datasets/CLoT/Summ/VCSum',
        
    #     'features': _BASE_FEATURES,
    #     'test_masked_feature': _BASE_TEST_MASK_FEATURE, # used only for uploading-leaderboard style, not supported now
    #     'prompt_template':'下面有一段会议记录：\n\n{prompt}\n\n{question}\n\n会议总结：', 
    #     'max_new_tokens': 512, # free-form summarization
        
    #     'metric_desc': _Summ_METRIC_INPUT_DESCRIPTION,
    #     'metric_features': _BASE_METRIC_FEATURES,
    #     'metric': MetricType.Summ,
        
    #     'splits': _BASE_SPLITS,
    #     'test_split': _BASE_TEST_SPLIT,
    #     'file_format': _BASE_FILE_FORMAT,
    #     'base': 'LongBench',  
    # },
    
    
    #########       OpenW      #########
    # 'TitleWave': {
    #     'metask': Metask.OpenW,
    #     'subtask': 'Story Generation from Title and Outlines with some few-shots examples',
    #     'desc': 'This is a subtask dataset belonging to the OpenW metask in CLoT benchmark, whhich asks LLMs to generate a whole story based on the title and some outlines, with some few-shot examples for references',
    #     'urls': '../datasets/CLoT/OpenW/TitleWave', # to be replaced with web link
        
    #     'features': _BASE_FEATURES,
    #     'test_masked_feature': _BASE_TEST_MASK_FEATURE, # used only for uploading-leaderboard style, not supported now
    #     'prompt_template': "请参考以下基于故事标题和故事关键词进行完整故事写作的若干示例：\n\n{prompt}\n\n 现给定以下故事标题和故事关键词：\n\n{question}\n\n请仿照示例续写完整故事：",
    #     'max_new_tokens': 1, # only one option letter is allowed
        
    #     'metric_desc': _MCQA_METRIC_INPUT_DESCRIPTION,
    #     'metric_features': _BASE_METRIC_FEATURES,
    #     'metric': MetricType.OpenW,
        
    #     'splits': _BASE_SPLITS,
    #     'test_split': _BASE_TEST_SPLIT,
    #     'file_format': _BASE_FILE_FORMAT,
    #     'base': 'LOT',
    # },
}
    

############################     utils funcs     ############################ 


def save_infos(info, save_dir=None):
    """
    Two steps:
    1. generate the dataset metadata in README.md and make sure the loading script works correctly
    2. save the dataset infos as a json file
    """
    if not save_dir:
        save_dir = "."
    # step 1
    import subprocess
    command = f"datasets-cli test {save_dir}/CLoT.py --save_info --all_configs"
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # step2
    info.write_to_directory(save_dir)
        

############################     dataset class    ############################ 


class CLoTConfig(BuilderConfig):
    """BuilderConfig for subtask dataset in CLoT bunchmark"""
    
    def __init__(self, metask, subtask='', 
                 features=_BASE_FEATURES, 
                 prompt_template=_BASE_PROMPT_TEMPLATE, 
                 max_new_tokens=512, 
                 **kwargs):
        super().__init__(**kwargs)
        
        if metask not in Metask:
            raise ValueError(f"The metask {metask} is not in the valid list: {[mt.value for mt in Metask]}")
        
        self.metask = metask.value
        self.subtask = subtask
        self.features = features
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens


class CLoTDataset(GeneratorBasedBuilder):
    """The GeneratorBasedBuilder class for CLoT benchmark following the hugging-face style
    to be loaded with: dataset = datasets.load_dataset('CLOT', config_name)
    """
    
    VERSION = Version(_VERSION) 
    
    # load one subtask dataset(configuration) with: 
    # subtask_dataset = datasets.load_dataset('CLoT', 'subtask_name')
    # and the selected configuration name is stored in self.config.name
    BUILDER_CONFIGS = [ 
        CLoTConfig(
            # original config
            name=subtask_name, 
            version=_VERSION, 
            description=_SUBTASK_INFOS[subtask_name]['desc'],
            # CLoT specific config
            metask=_SUBTASK_INFOS[subtask_name]['metask'], 
            subtask=_SUBTASK_INFOS[subtask_name]['subtask'],
            features=_SUBTASK_INFOS[subtask_name]['features'],
            prompt_template=_SUBTASK_INFOS[subtask_name]['prompt_template'],
            max_new_tokens=_SUBTASK_INFOS[subtask_name]['max_new_tokens'],
            )
        for subtask_name in _SUBTASK_INFOS
    ]
    
    # DEFAULT_CONFIG_NAME = '' # it's not mandatory to have a default configuration, just using one if it make sense
    
    def _info(self):
        """This method specifies the datasets.DatasetInfo object which contains information and typings for the dataset"""
        
        return DatasetInfo(
            # common information shared with all subtask datasets
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            supervised_keys=None,
            # unique information for a specified subtask dataset 
            features=_SUBTASK_INFOS[self.config.name]['features'],
        )
    
    def _split_generators(self, dl_manager):
        """This method is tasked with downloading / extracting the data and defining the splits depending on the configuration"""
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        urls = _SUBTASK_INFOS[self.config.name]['urls']
        if os.path.isdir(urls):
            sub_dataset_dir = urls
        else:
            sub_dataset_dir = dl_manager.download_and_extract(urls) # this directory contains all split files of the subtask dataset corresponding to configuration
            
            
        splits = _SUBTASK_INFOS[self.config.name]['splits'] # all split
        file_format = _SUBTASK_INFOS[self.config.name]['file_format'] # file format, like json, jsonl, csv, ...
        
        return [
            SplitGenerator(
                name=split,
                gen_kwargs={ # these kwargs will be passed to _generate_examples
                    'filepath': os.path.join(sub_dataset_dir, 
                                             f"{split}.{file_format}"),
                    'split': split
                }
            )
            for split in splits
        ]
        
    def _generate_examples(self, filepath, split):
        """This method handles input defined in _split_generators to yield (key, example) tuples from the split file of the subtask dataset"""
        file_format = _SUBTASK_INFOS[self.config.name]['file_format']
        
        for key, sample in enumerate(CLoTDataset._sample_extractor(filepath, file_format)):
            # key is for legacy reasons (tfds) and is not important in itself,
            # but must be unique for each sample in the split file
            key = f"{self.config.name}-{key}"
            if 'id' in sample: # id is also must be unique for each sample in the split file
                key = sample['id']
            yield key, sample
            
    @staticmethod       
    def _sample_extractor(filepath, file_format):
        with open(filepath, 'r', encoding='utf-8') as f:
            if file_format == 'jsonl':
                for line in f:
                    sample = json.loads(line)
                    yield sample
            elif file_format == 'json':
                samples = json.load(f)
                for sample in samples:
                    yield sample
            else:
                raise TypeError(f"The file format {file_format} is not supported yet")

        
############################     metric class    ############################


class CLoTMetricInfo(evaluate.MetricInfo):
    """MetricInfo class for CLoT bunchmark"""
    def __init__(self, metric_type, **kwargs):
        super().__init__(**kwargs)
        
        self.metric_type = metric_type


class CLoTMetric(evaluate.Metric):
    """The Metric class for CLoT benchmark following the hugging-face style
    to be loaded with: metric = evaluate.load('CLOT', config_name) (new, recommended) or datasets.load_metric('CLoT', config_name)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        ########    define the metric funcs dict here    ########
        self.metric_funcs = {
            MetricType.MCQA_ABCD: CLoTMetric._mcqa_abcd_metric_func,
            MetricType.MCQA_NUMBER: CLoTMetric._mcqa_number_metric_func,
            MetricType.ExtQA: CLoTMetric._extqa_metric_func,
            MetricType.Summ: CLoTMetric._summ_metric_func,
            MetricType.OpenW: CLoTMetric._openw_metric_func,
        }
        
    def _info(self):
        """this method specifies the evaluate.MetricInfo which contains information and typings for the metric"""
        
        return CLoTMetricInfo(
            module_name="CLoTMetric",
            metric_type=_SUBTASK_INFOS[self.config_name]['metric'],
            description=_METRIC_DESCRIPTION,
            inputs_description=_SUBTASK_INFOS[self.config_name]['metric_desc'],
            citation=_CITATION,
            features=_SUBTASK_INFOS[self.config_name]['metric_features'],
        )
        
    def _compute(self, predictions, references, **kwargs) -> dict:
        return self.metric_funcs[_SUBTASK_INFOS[self.config_name]['metric']](
            predictions=predictions, 
            references=references, 
            **kwargs)
    
    @staticmethod
    def _mcqa_abcd_metric_func(predictions, references, **kwargs) -> dict:
        """
        predictions / references: a list of option letter strings, where each string contains all the chose option letters
        e.g. if there is only a single right option, then:
            the references may be like: ['A', 'B', 'C', 'D']
            the predictions may be like: ['C', 'B', 'AB', 'D']
        then we find all option letter (split char does not matter) in the string and match the letter sets between references and predictions
        e.g. if there is multiple right options, then:
            the references may be like: ['ABC', 'A,C', 'A D']
            the predictions may be like: ['C', 'A.B.', 'A,D'] 
        """
        import re
        correct = 0
        get_options = lambda s: set([option for option in re.findall(r'[ABCD]', s)])
        
        for prediction, reference in zip(predictions, references):
            prediction_options = get_options(prediction)
            reference_options = get_options(reference[0]) # single true option
            if prediction_options == reference_options:
                correct += 1
        
        acc = correct / len(references) if len(reference) > 0 else -1.0
        return {
            'accuracy': round(acc, 2)
        }
    
    @staticmethod
    def _mcqa_number_metric_func(predictions, references, **kwargs) -> dict:
        """
        predictions / references: a list of option number strings, where each string contains the chosen option number
        e.g. if there is only a single right option, then:
            the references may be like: ['文章1', '27']
            the predictions may be like: ['文章1', '文章27']
        then we find the first option number string (other chars do not matter) in the string and match the number between references and predictions
        """
        import re
        correct = 0
        def get_option_number(s):
            match = re.search(r'\d+', s)
            if match:
                return int(match.group())
            else:
                return 0
        
        for prediction, reference in zip(predictions, references):
            prediction_option = get_option_number(prediction)
            reference_option = get_option_number(reference[0])
            if prediction_option == reference_option:
                correct += 1
        
        acc = correct / len(references) if len(reference) > 0 else -1.0
        return {
            'accuracy': round(acc, 2)
        }

    @staticmethod
    def _extqa_metric_func(predictions, references, **kwargs) -> dict:
        """
        predictions: a list of strings, each of which is the output for one question
        reference: a list of list[strings], each of which is the candidate ground-truth answers for one question
        match every prediction to each of its candidates and get the max score
        score type: { ROUGE-1[F1], ROUGE-2[F1], ROUGE-L[F1] }
        """
        import jieba
        from collections import defaultdict
        
        predictions = [ " ".join(list(jieba.cut(prediction, cut_all=False))) for prediction in predictions ]
        references = [ [" ".join(list(jieba.cut(groundtruth, cut_all=False))) for groundtruth in reference ] for reference in references] 
        
        total_score = defaultdict(float)
        for prediction, reference in zip(predictions, references):
            sample_score = defaultdict(float)
            for groundtruth in reference:
                score = CLoTMetric.rouge_score(prediction, groundtruth, **kwargs)
                for score_type in score:
                    sample_score[score_type] = max(sample_score[score_type], score[score_type])
        
            for score_type, score in sample_score.items():
                total_score[score_type] += score
                
        for score_type in total_score:
            total_score[score_type] = total_score[score_type] / len(references) if len(references) > 0 else -1.0
        
        return dict(total_score)   
        
    @staticmethod
    def _summ_metric_func(predictions, references, **kwargs) -> dict:
        """
        predictions / references: a list of summative strings for each content
        score type: { ROUGE-1[F1], ROUGE-2[F1], ROUGE-L[F1] }
        """
        import jieba
        from collections import defaultdict
        
        predictions = [ " ".join(list(jieba.cut(prediction, cut_all=False))) for prediction in predictions ]
        references = [ " ".join(list(jieba.cut(reference[0], cut_all=False))) for reference in references]
        
        total_score = defaultdict(float)
        for prediction, reference in zip(predictions, references): 
            sample_score = CLoTMetric.rouge_score(prediction, reference, **kwargs)
            for score_type, score in sample_score.items():
                total_score[score_type] += score
                
        for score_type in total_score:
            total_score[score_type] = total_score[score_type] / len(references) if len(references) > 0 else -1.0
        
        return dict(total_score)   
        
    @staticmethod
    def _openw_metric_func(predictions, references, **kwargs) -> dict:
        """
        predictions / references: a list of free-form texts for each writing object
        score type: { BLEU (sentence-level) }
        """
        from collections import defaultdict
        total_score = defaultdict(float)
        
        for prediction, reference in zip(predictions, references):
            sample_score = CLoTMetric.bleu_score(prediction, reference, **kwargs)
            for score_type, score in sample_score.items():
                total_score[score_type] += score
        
        for score_type in total_score:
            total_score[score_type] = total_score[score_type] / len(references) if len(references) > 0 else -1.0
        
        return dict(total_score)
        
    @staticmethod
    def bleu_score(prediction, reference, **kwargs) -> dict:
        """sentence bleu score for Chinese
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import jieba
        import numpy as np
        
        pred_tokens = jieba.lcut(prediction, cut_all=True)
        ref_tokens = jieba.lcut(reference, cut_all=True)
        
        smoothie = SmoothingFunction().method4
        
        weights = [
            (0.5, 0.5),
            (0.333, 0.333, 0.333),
            (0.25, 0.25, 0.25, 0.25),
            (0.2, 0.2, 0.2, 0.2, 0.2),
        ]
        
        sen_bleu_scores = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie, weights=weights)
        
        return {"BLEU": np.mean(sen_bleu_scores)}
           
    @staticmethod
    def rouge_score(prediction, groundtruth, m='f', **kwargs) -> dict:
        """
        m = p: precision | r: recall | f: f1
        """
        from rouge import Rouge
        rouger = Rouge()
        try:
            scores = rouger.get_scores([prediction], [groundtruth], avg=True)
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
   
         
############################     main    ############################ 


def main(args):
    from datasets import load_dataset
    from evaluate import load
    
    script_path = './CLoT.py'
    
    dataset = load_dataset(script_path, args.subtask, split=args.split)
    metric = load(script_path, config_name=args.subtask)
    
    if args.run_mode == 'overview':
        print(f"=> The loaded subtask dataset {args.subtask} with the split {args.split} is shown below: \n\n {dataset.info} \n")
        print(f"=> The corresponding metric is shown below: \n\n {metric.info} \n")
    elif args.run_mode == 'save_info':
        save_infos(dataset.info, args.save_dir)
        print("=> The dataset infos saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The dataset / metric loading script for CLoT benchmark following the hugging-face style")
    parser.add_argument('-r', "--run-mode", type=str, default='overview', 
                        help='The running mode of this script itself', 
                        choices=['overview', 'save_info'])
    parser.add_argument("--subtask", type=str, default='PlotPatcher', help='The subtask datasetname')
    parser.add_argument('--split', type=str, default='train', help='The split to choose')
    parser.add_argument("--save-dir", type=str, default='./', help='The dir to save the infos when running save_info mode')
    args = parser.parse_args()
    
    main(args)