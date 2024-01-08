# LeaderBoard 

## LeaderBoard for LongBench-small-dureader
 => *Each sub-leaderboard is for a certain length range, where each row shows the scores for one model on each metric equipped in the subtask LongBench-small-dureader* 
### all: 
|                              |   Rouge-1 (f) |   Rouge-2 (f) |   Rouge-L (f) |   num_samples |
|:-----------------------------|--------------:|--------------:|--------------:|--------------:|
| chatglm-6b                   |          0.12 |          0    |          0.1  |            10 |
| chatglm2-6b-32k              |          0.3  |          0.09 |          0.26 |            10 |
| chatglm2-6b                  |          0.23 |          0.03 |          0.19 |            10 |
| cllama(rerope)               |          0.21 |          0.07 |          0.18 |            10 |
| cllama2(rerope)              |          0.3  |          0.06 |          0.24 |            10 |
| openbuddy-llama2-13b(rerope) |          0.33 |          0.16 |          0.28 |            10 |
--- 
### 8k~16k: 
|                              |   Rouge-1 (f) |   Rouge-2 (f) |   Rouge-L (f) |   num_samples |
|:-----------------------------|--------------:|--------------:|--------------:|--------------:|
| chatglm-6b                   |          0.15 |          0.01 |          0.13 |             6 |
| chatglm2-6b-32k              |          0.33 |          0.08 |          0.29 |             6 |
| chatglm2-6b                  |          0.25 |          0.05 |          0.2  |             6 |
| cllama(rerope)               |          0.23 |          0.08 |          0.2  |             9 |
| cllama2(rerope)              |          0.31 |          0.05 |          0.25 |             9 |
| openbuddy-llama2-13b(rerope) |          0.34 |          0.16 |          0.3  |             7 |
--- 
### 16k+: 
|                              |   Rouge-1 (f) |   Rouge-2 (f) |   Rouge-L (f) |   num_samples |
|:-----------------------------|--------------:|--------------:|--------------:|--------------:|
| chatglm-6b                   |          0.07 |          0    |          0.05 |             4 |
| chatglm2-6b-32k              |          0.26 |          0.12 |          0.22 |             4 |
| chatglm2-6b                  |          0.19 |          0.02 |          0.16 |             4 |
| openbuddy-llama2-13b(rerope) |          0.29 |          0.15 |          0.25 |             3 |
--- 
### 4k~8k: 
|                 |   Rouge-1 (f) |   Rouge-2 (f) |   Rouge-L (f) |   num_samples |
|:----------------|--------------:|--------------:|--------------:|--------------:|
| cllama(rerope)  |          0    |          0    |          0    |             1 |
| cllama2(rerope) |          0.26 |          0.09 |          0.21 |             1 |
--- 


## LeaderBoard for LongBench-small-lsht
 => *Each sub-leaderboard is for a certain length range, where each row shows the scores for one model on each metric equipped in the subtask LongBench-small-lsht* 
### all: 
|                              |   accuracy |   num_samples |
|:-----------------------------|-----------:|--------------:|
| chatglm-6b                   |       0.1  |            10 |
| chatglm2-6b-32k              |       0.1  |            10 |
| chatglm2-6b                  |       0.23 |            10 |
| cllama(rerope)               |       0.4  |            10 |
| cllama2(rerope)              |       0.6  |            10 |
| openbuddy-llama2-13b(rerope) |       0.5  |            10 |
--- 
### 8k~16k: 
|                              |   accuracy |   num_samples |
|:-----------------------------|-----------:|--------------:|
| chatglm-6b                   |       0.2  |             5 |
| chatglm2-6b-32k              |       0    |             5 |
| chatglm2-6b                  |       0.4  |             5 |
| cllama(rerope)               |       0.5  |             8 |
| cllama2(rerope)              |       0.62 |             8 |
| openbuddy-llama2-13b(rerope) |       0.75 |             4 |
--- 
### 16k+: 
|                              |   accuracy |   num_samples |
|:-----------------------------|-----------:|--------------:|
| chatglm-6b                   |       0    |             5 |
| chatglm2-6b-32k              |       0.2  |             5 |
| chatglm2-6b                  |       0.07 |             5 |
| cllama2(rerope)              |       1    |             1 |
| openbuddy-llama2-13b(rerope) |       0.33 |             6 |
--- 
### 4k~8k: 
|                 |   accuracy |   num_samples |
|:----------------|-----------:|--------------:|
| cllama(rerope)  |          0 |             2 |
| cllama2(rerope) |          0 |             1 |
--- 


## LeaderBoard for LongBench-small-multifieldqa_zh
 => *Each sub-leaderboard is for a certain length range, where each row shows the scores for one model on each metric equipped in the subtask LongBench-small-multifieldqa_zh* 
### all: 
|                              |   F1 |   num_samples |
|:-----------------------------|-----:|--------------:|
| chatglm-6b                   | 0.29 |            10 |
| chatglm2-6b-32k              | 0.41 |            10 |
| chatglm2-6b                  | 0.26 |            10 |
| cllama(rerope)               | 0.53 |            10 |
| cllama2(rerope)              | 0.59 |            10 |
| openbuddy-llama2-13b(rerope) | 0.51 |            10 |
--- 
### 2k~4k: 
|                              |   F1 |   num_samples |
|:-----------------------------|-----:|--------------:|
| chatglm-6b                   | 0.29 |             5 |
| chatglm2-6b-32k              | 0.44 |             5 |
| chatglm2-6b                  | 0.28 |             5 |
| cllama(rerope)               | 0.55 |             5 |
| cllama2(rerope)              | 0.6  |             5 |
| openbuddy-llama2-13b(rerope) | 0.39 |             4 |
--- 
### 4k~8k: 
|                              |   F1 |   num_samples |
|:-----------------------------|-----:|--------------:|
| chatglm-6b                   | 0.45 |             3 |
| chatglm2-6b-32k              | 0.31 |             3 |
| chatglm2-6b                  | 0.33 |             3 |
| cllama(rerope)               | 0.59 |             4 |
| cllama2(rerope)              | 0.59 |             4 |
| openbuddy-llama2-13b(rerope) | 0.56 |             4 |
--- 
### 8k~16k: 
|                              |   F1 |   num_samples |
|:-----------------------------|-----:|--------------:|
| chatglm-6b                   | 0.05 |             2 |
| chatglm2-6b-32k              | 0.5  |             2 |
| chatglm2-6b                  | 0.12 |             2 |
| openbuddy-llama2-13b(rerope) | 0.64 |             2 |
--- 
### 1k~2k: 
|                 |   F1 |   num_samples |
|:----------------|-----:|--------------:|
| cllama(rerope)  | 0.2  |             1 |
| cllama2(rerope) | 0.53 |             1 |
--- 


## LeaderBoard for LongBench-small-passage_retrieval_zh
 => *Each sub-leaderboard is for a certain length range, where each row shows the scores for one model on each metric equipped in the subtask LongBench-small-passage_retrieval_zh* 
### all: 
|                              |   accuracy |   num_samples |
|:-----------------------------|-----------:|--------------:|
| chatglm-6b                   |        0   |            10 |
| chatglm2-6b-32k              |        0.5 |            10 |
| chatglm2-6b                  |        0.1 |            10 |
| cllama(rerope)               |        0.2 |            10 |
| cllama2(rerope)              |        0.1 |            10 |
| openbuddy-llama2-13b(rerope) |        0.2 |            10 |
--- 
### 4k~8k: 
|                              |   accuracy |   num_samples |
|:-----------------------------|-----------:|--------------:|
| chatglm-6b                   |       0    |            10 |
| chatglm2-6b-32k              |       0.5  |            10 |
| chatglm2-6b                  |       0.1  |            10 |
| cllama(rerope)               |       0.25 |             8 |
| cllama2(rerope)              |       0.1  |            10 |
| openbuddy-llama2-13b(rerope) |       0.2  |            10 |
--- 
### 2k~4k: 
|                |   accuracy |   num_samples |
|:---------------|-----------:|--------------:|
| cllama(rerope) |          0 |             2 |
--- 


