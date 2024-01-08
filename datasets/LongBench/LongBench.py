# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import datasets
import json


_DESCRIPTION = """LongBench is a comprehensive benchmark for multilingual and multi-task purposes, 
with the goal to fully measure and evaluate the ability of pre-trained language models to understand long text. 
This dataset consists of twenty different tasks, covering key long-text application scenarios such as multi-document QA, single-document QA, summarization, few-shot learning, synthetic tasks, and code completion.

And For our own purpose and usage, we cut it into four size-splits, debug, small, medium and large for different experiment time and resouce limits requirements

For version1.0:
    debug: random choose 1 sample for each subtask
    small: random choose 10 samples for each subtask
    medium: random choose 50 samples for each subtask
    large: random choose 200 samples for each subtask (the original size of LongBench)
    
For version2.0: (cuz of OOM problem and tensor parallel lock problem)
    debug: random choose 1 sample for each subtask with length range [1k, 20k]
    small: random choose 10 samples for each subtask with length range [1k, 20k]
    medium: random choose 50 samples for each subtask with length range [1k, 20k]
    large: random choose 100 samples for each subtask with length range [1k, 20k]

"""


_HOMEPAGE = "https://github.com/THUDM/LongBench"


## remote url
# _URL = r"https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"

## local abstract url
# _URL = "/mnt/cfs_bj/huangyunpeng/baidu/personal-code/lcw_benchmark/datasets/LongBench/data.zip"
# _URL = '/mnt/cfs_bj/huangyunpeng/baidu/personal-code/lcw_benchmark/datasets/LongBench/'

## local relative url
_URL = '../datasets/LongBench'


task_list = [
    ## en
    # "multifieldqa_en",
    # "lcc",
    # "qasper",
    # "nq",
    # "passage_retrieval_en",
    # "gov_report",
    # "triviaqa",
    # "qmsum",
    # "trec",
    # "2wikimqa",
    # "passage_count",
    # "repobench-p",
    # "hotpotqa",
    # "narrativeqa",
    # "musique",
    
    ## zh
    "dureader",
    "lsht",
    "passage_retrieval_zh",
    "vcsum",
    "multifieldqa_zh"
]


class LongBenchConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("2.0.0"), **kwargs)


class LongBench(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        LongBenchConfig(
            name=task_name,
            split_size='small',
            split_version='v1.0',
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "input": datasets.Value("string"), 
                "context": datasets.Value("string"), 
                "answers": [datasets.Value("string")], 
                "length": datasets.Value("int32"), 
                "dataset": datasets.Value("string"), 
                "language": datasets.Value("string"), 
                "all_classes": [datasets.Value("string")],
                "_id": datasets.Value("string"), 
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        ## remove download and extract
        # data_dir = dl_manager.download_and_extract(_URL)
        ## local read 
        data_dir = _URL
        
        task_name = self.config.name
        split_size = self.config.split_size
        split_version = self.config.split_version
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, split_version, split_size, f"{task_name}.jsonl"
                    ),
                },
            )
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                key = f"{self.config.name}-{idx}"
                item = json.loads(line)
                yield key, {
                    "input": item["input"],
                    "context": item["context"],
                    "answers": item["answers"],
                    "length": item["length"],
                    "dataset": item["dataset"],
                    "language": item["language"],
                    "_id": item["_id"],
                    "all_classes": item["all_classes"],
                }