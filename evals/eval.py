############################       base libs import     ############################

import os
import argparse
import warnings
import time
warnings.filterwarnings('ignore')

############################       own libs import     ############################

from evaluator import load_evaluator, Evaluator
from config import EvaluatorConfig
from utils import set_visible_gpus, info_str

############################     main func    ############################ 

def main(args):
    # set visible gpus
    set_visible_gpus(args.gpus)
    # do evaluation exp
    if args.run_mode == 'eval':
        evaluator_config = EvaluatorConfig.load(args.config_path)
        evaluator = load_evaluator(evaluator_config, args)
        evaluator()
    # only report from previous exp outputs
    elif args.run_mode =='report':
        evaluator = Evaluator()
        evaluator.report(args.report_dir)
    else:
        raise NotImplementedError(f"The running mode {args.run_mode} is not supported")
    
    print(info_str("All Done!"))


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="This is the script to evaluate the lcw-aug LLMs on the chinese long-text datasets")
    parser.add_argument('-r', "--run-mode", type=str, default='eval', help='the running mode for this script', choices=['eval', 'report'])
    parser.add_argument('-c', "--config-path", type=str, default='', help="The config file path.")
    parser.add_argument('-o', '--output-dir', type=str, default='./outputs', help='The root dir to store output files within test results')
    parser.add_argument('-g', '--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7], help='the visible gpus indices')
    # parser.add_argument('-d', '--device', type=str, default='cuda:0', help='The device to use for evaluating')
    parser.add_argument('-v', "--verbose", action='store_true', help="Whether to print debug info")
    
    parser.add_argument('--report-dir', type=str, default='', 
                        help='The root dir to store output files within test results in old exps, only used in report mode')
    args = parser.parse_args()
    
    main(args)
    
