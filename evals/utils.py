############################       base libs import     ############################

import os
import argparse
import time
import subprocess
import threading
from typing import Any, Tuple, List, Union, Optional
from contextlib import contextmanager

############################       dev libs import     ############################

import numpy as np
import torch

############################       utils func     ############################


def get_torch_dtype(dtype_str):
    if not hasattr(torch, dtype_str):
        raise AttributeError(f"The dtype {dtype_str} is not supported in pytorch")
    
    return getattr(torch, dtype_str)

def num_of_params(model: torch.nn.Module, format=True) -> Union[str, float]:
    num_p = sum([p.numel() for p in model.parameters() if p.requires_grad])
    if not format:
        return num_p
    
    num_b = num_p // 1_000_000_000
    num_m = (num_p % 1_000_000_000) // 1_000_000
    
    return f"{num_b}B {num_m}M"

def info_str(center_content: str = "", side_str: str = "=", side_num: int = 15) -> str:
    return "\n" + side_str * side_num + " " + center_content + " " + side_str * side_num + "\n"

def info_output(content: str = "", 
                truncate: bool = True, 
                visible_words: int = 200,
                truncate_mode: str = "middle",
                return_info: bool = False,
                ) -> str:
    info = f"[Total: {len(content)} words | Omitted: {max(0, len(content)-visible_words)} words] "
    
    def get_info():
        if return_info: return info
        return ""
    
    if not truncate or len(content) <= visible_words:
        return get_info() + content
    
    if truncate_mode == "left":
        truncated_content = get_info() + "(omitted)...... " + content[-visible_words:].strip('\n')
    elif truncate_mode == "right":
        truncated_content = get_info() + content[:visible_words].strip('\n') + " ......(omitted)"
    elif truncate_mode == "middle":
        truncated_content = get_info() + content[:visible_words//2].strip('\n') + "...(omitted)..." + content[-visible_words//2:].strip('\n')
    
    return truncated_content

def info_list(list_content=[], split='-') -> str:
    return split.join(list_content)

def info_ordinal(idx: int) -> str:
    if idx < 1:
        raise ValueError(f"The idx {idx} is not valid (>= 1)")
    if idx == 1:
        return "1st"
    elif idx == 2:
        return "2nd"
    else:
        return str(idx) + "th"
    
def info_dict(d, t=1, precision=2) -> str:
    s = "{\n"
    for k, v in d.items():
        s += "\t"*t + str(k)
        s += " : "
        if isinstance(v, dict):
            vd = info_dict(v, t+1)
            s += vd
        else:
            if isinstance(v, float):
                if len(str(v)) > len("0.001"): s += f"{v:.{precision}e}"
                else: s += str(v)
            else: s += str(v)
                    
        s += "\n"
    s +=  "\t"*(t-1) + "}"

    return s
    
def set_visible_gpus(gpu_ids: List[int]) -> None:
    """Set cuda visible gpu indexs like:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7' # 8 A800
    Keep in mind that: pytorch will reindex the gpus from the visible list
    """
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
def memory_used(unit: str ='G', 
                mem: Optional[float] = None, 
                format: bool = True,
                reduce: str = 'sum',
                ) -> Union[str, float]:
    if mem:
        if format:
            return f"{mem:.2f} {unit}B"
        return mem
         
    unit_dom = {
        'G': 1024 ** 3,
        'M': 1024 ** 2,
    }[unit]
    
    def total_gpus_memory_allocated(reduce):
        mems = [
            torch.cuda.memory_allocated(i)
            for i in range(torch.cuda.device_count())
        ]
        reduce_func = {
            'sum': sum, 'min': min, 'max': max, 'mean': np.mean
        }
        if reduce not in reduce_func:
            raise KeyError(f"The reduce method {reduce} is not supported")
        
        return reduce_func[reduce](mems)
    
    mem = total_gpus_memory_allocated(reduce) / unit_dom
    if format:
        return f"{mem:.2f} {unit}B"
    return mem

@contextmanager
def time_manager(name: str = ""):
    start_time = time.time()
    suffix = f" for {name}" if name != "" else ""
    try:
        print(info_str(f"Timing Begins{suffix}"))
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(info_str(f"Time Costed{suffix}: {int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds"))

class Timer(object):
    
    def __init__(self, precision: int = 2) -> None:
        self.precision = precision
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc, value, traceback):
        pass
            
    @property
    def time(self) -> float:
        elapsed_time = time.time() - self.start_time
        return round(elapsed_time, self.precision)

def get_cond_idxs(array: np.ndarray, cond_func) -> np.ndarray:
    return np.where(cond_func(array))[0]

def get_split_len_strs_and_conds(split_lens: List[int]) -> Tuple[List[str], List]:
    
    def get_split_len_str(len1, len2=None, mode='middle', unit='k'):
        def get_len_str(len, unit):
            if unit == 'k':
                return str(len//1000) + 'k'
            elif unit == 'w':
                return str(len//10000) + 'w'
        if mode == 'middle':
            if len2 is None:
                raise ValueError(f"The second split length must be set when 'mode' is 'middle'")
            return get_len_str(len1, unit) + '~' + get_len_str(len2, unit)
        if mode == 'left':
            return '<' + get_len_str(len1, unit)
        if mode == 'right':
            return get_len_str(len1, unit) + '+'
    def cond(l=None, r=None):
        if l is None and r is None: # no comparison
            return lambda x: True
        if l is None: # right comparion
            return lambda x: x >= r
        if r is None: # left comparion
            return lambda x: x < l
        
        return lambda x: (x>=l) & (x<r) # double comparison
    
    split_lens = sorted(split_lens)
    split_strs, split_conds = [], []
    
    for i, split_len in enumerate(split_lens):
        if i == len(split_lens)-1: # split after the max split length
            split_strs.append(get_split_len_str(split_len, mode='right'))
            split_conds.append(cond(r=split_len))
            break
        # split before the min split length
        if i == 0:
            split_strs.append(get_split_len_str(split_len, mode='left'))
            split_conds.append(cond(l=split_len))
        # split between the ith split length ~ (i+1)th split length
        split_strs.append(get_split_len_str(split_len, split_lens[i+1], mode='middle'))
        split_conds.append(cond(l=split_len, r=split_lens[i+1]))
            
    return split_strs, split_conds

def check_package_version(pacakge, 
                  min_version: Optional[str] = None, 
                  max_version: Optional[str] = None):
    def get_comps(s: str) -> List[int]: return list(map(int, s.split('.')))
    
    version = pacakge.__version__
    if '+' in version: version = version.split('+')[0] # in case of torch2.0.0+cu117
    version_comps = get_comps(version)
    if min_version is not None:
        min_version_comps = get_comps(min_version)
        if version_comps < min_version_comps:
            raise ImportError(f"The version of {pacakge.__name__} should be at least v{min_version}, (current v{version})")
    if max_version is not None:
        max_version_comps = get_comps(max_version)
        if version_comps > max_version_comps:
            raise ImportError(f"The version of {pacakge.__name__} should not be above v{max_version}, (current v{version})")

def check_str_bool(content):
    if content == True: return True
    if content == False: return False     
    if isinstance(content, str):
        if content.lower() == 'true': return True
        elif content.lower() == 'false': return False
        else:
            raise ValueError(f"The value of {content} is not supported (str.lower(): ['true'/'false']).")
    raise TypeError(f"The type of {content} is not supported (bool or str).")

class GPUMemoryMonitor:
    def __init__(self, 
                 unit: str = 'G', 
                 reduce: str = 'sum',
                 elapse: float = 1.0,
                 ) -> None:
        self.unit = unit
        self.reduce = reduce
        self.elapse = elapse
        
        self.memories = []
        self.gpu_monitor_thread = None
        self.memory_update_event = threading.Event()
        
        self.reduce_funcs = {
            'max': max, 'min': min, 'mean': np.mean, 'all': None,
        }
        
    def __enter__(self):
        self._start_monitor()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.gpu_monitor_thread is not None:
            self._stop_monitor()
    
    def _start_monitor(self):
        if self.gpu_monitor_thread is not None: return
        
        self.memories = []
        
        self.memory_update_event.clear()
        self.gpu_monitor_thread = threading.Thread(target=self._monitor)
        self.gpu_monitor_thread.setDaemon(True)
        self.gpu_monitor_thread.start()
    
    def _stop_monitor(self):
        if self.gpu_monitor_thread is None: return
        
        self.gpu_monitor_thread.join()
        self.gpu_monitor_thread = None
        
    def _monitor(self):
        while True:
            self.memories.append(self._get_gpu_memory())
            self.memory_update_event.set()
            time.sleep(self.elapse)
            
    def _get_gpu_memory(self):
        return memory_used(unit=self.unit, format=False, reduce=self.reduce)
    
    def get_memory(self, reduce: str = 'max'):
        reduce_func = self.reduce_funcs[reduce]
        self.memory_update_event.wait()
        if reduce_func is None: return self.memories
        return reduce_func(self.memories)
    

############################       main func     ############################


def main(args):
    if args.test_report:
        outputs = {
            "LongBench-multifieldqa_zh": {
                "openbuddy-llama2-13b": {
                    "pred_output": [
                        {
                            "prediction": "。。。。。。。。。。。。。。。。。。。。。的的的的的的的的的的的。的的的的。的的的的的的。。。。。。。。。。。。。。。。。。。。",
                            "refererence": [
                                "厦门大学。"
                            ],
                            "context_length": 9544
                        },
                        {
                            "prediction": "的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的的",
                            "refererence": [
                                "6.5亿元。"
                            ],
                            "context_length": 6603
                        },
                        {
                            "prediction": "57081.86 元。\n",
                            "refererence": [
                                "人民币57081.86元。"
                            ],
                            "context_length": 3038
                        },
                        {
                            "prediction": "，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，的，，，，，，，，，的的的的的的的的的的",
                            "refererence": [
                                "郗鉴拒绝了外戚庾亮废王导的建议。"
                            ],
                            "context_length": 6348
                        },
                        {
                            "prediction": "\n",
                            "refererence": [
                                "10万元。"
                            ],
                            "context_length": 4521
                        },
                        {
                            "prediction": "73mm\n",
                            "refererence": [
                                "10mm。"
                            ],
                            "context_length": 3188
                        },
                        {
                            "prediction": "的的，，，，，，，，，，，，，，，，的的的的的的的的的的，，，，，，，，，，，，，，，，，，的的的的，的的的，，，，，，，，的的",
                            "refererence": [
                                "出狱后接触到毒品并输掉大量赌资。"
                            ],
                            "context_length": 9047
                        },
                        {
                            "prediction": "织田信长、丰臣秀吉、斋藤道三、毛利元就、武田信玄、上杉氏、后北条氏、今川氏、岛津氏、长宗我部氏。\n",
                            "refererence": [
                                "织田信长和丰臣秀吉。"
                            ],
                            "context_length": 3421
                        },
                        {
                            "prediction": "63.73 亿元。\n",
                            "refererence": [
                                "2012年，铜仁市的财政总收入增长了25.89%。"
                            ],
                            "context_length": 2392
                        },
                        {
                            "prediction": "「九州之。」\n",
                            "refererence": [
                                "禹遇到了山川地势复杂、土壤贫瘠、水患频繁等问题。"
                            ],
                            "context_length": 4414
                        }
                    ],
                    "score_output": {
                        "all": {
                            "F1": 0.11,
                            "num_samples": 10
                        },
                        "2k~4k": {
                            "F1": 0.24,
                            "num_samples": 4
                        },
                        "4k~8k": {
                            "F1": 0.02,
                            "num_samples": 4
                        },
                        "8k~16k": {
                            "F1": 0.0,
                            "num_samples": 2
                        }
                    }
                }
            }
        }

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        pred_key, score_key, num_samples_key, length_key = 'pred_output', 'score_output', 'num_samples', 'prompt_length'
        # report leaderboard.md
        score_str = report_leaderboard(outputs, score_key=score_key)
        leaderboard_path = os.path.join(args.output_dir, 'leaderboard.md')
        with open(leaderboard_path, 'w') as f:
            f.write(score_str)
        # report plot.png
        fig = report_lengthplot(outputs, 
                                score_key=score_key,
                                num_samples_key=num_samples_key
                                )
        plot_path = os.path.join(args.output_dir, 'plot.png')
        fig.savefig(plot_path, dpi=300)
        # report radarchart
        fig = report_radarchart(outputs, score_key=score_key)
        radarchart_path = os.path.join(args.output_dir, 'radarchart.png')
        fig.savefig(radarchart_path, dpi=300)
        

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="This the script to test the utils / report funcs")
    parser.add_argument("--test-report", action='store_true', help='to test the report funcs')
    parser.add_argument("--output-dir", type=str, default='./outputs/test_utils', help='the output dir')
    args = parser.parse_args()
    main(args)
