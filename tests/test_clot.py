import datasets 
from datasets import load_dataset, load_dataset_builder
from evaluate import load


####### CLoT
dir = '../datasets/CLoT'

## MCQA
# subtask = 'PlotPatcher'
# subtask = 'AbsMatcher'

## RetQA
# subtask = 'MFReader' 

## Summ
# subtask = 'VCSum'

## OpenW
subtask = 'TitleWave'


split = 'dev'
idx = 0

builder = load_dataset_builder(dir, subtask)
builder.download_and_prepare()
dataset = builder.as_dataset(split=split)
dataset.max_new_tokens = builder.config.max_new_tokens 
dataset.prompt_template = builder.config.prompt_template
metric = load(dir, subtask)

print(f"\n => dataset: \n{dataset}\n\n")
print(f"\n => metric: \n{metric}\n\n")
print(f"\n => example: \n{dataset[idx]}\n\n")
print(f"\n => prompt: {dataset.prompt_template.format(**dataset[idx])}\n\n")

input = input('Please input your answer to this example: \n')

score = metric.compute(predictions=[input], references=[dataset[idx]['output']])

print(f"\n => the score of your answer is {score}\n")

