############################       dev libs import     ############################

import torch
import torch.nn as nn


############################     base module class    ############################


class BaseModule(nn.Module):
    """The base parent class of any other classes like LCWAugModel, LCWDataset, Evaluator, etc
    to share some common attributes like constants, to make the whole pipeline consistent
    """
    def __init__(self):
        super().__init__()
        
        ## some constants reported in output.json
        self.pred_output_key = 'pred_output' # preodiction output list key for certain model and certain dataset
        self.score_output_key = 'score_output' # scoring output dict key for certain model and certain dataset
        
        self.input_key = 'prompt' # the input(prompt) key for one sample
        self.prediction_key = 'prediction' # prediction key in each sample prediction output
        self.reference_key = 'reference' # reference key in each sample prediction output
        self.length_key = 'prompt_length' # length key in each sample prediction output to log the 
        self.single_score_output_key = 'score' # the score output for the single (prediction, reference) pair for each sample
        
        self.num_samples_key = 'num_samples' # number-of-samples key in score_output
        self.speed_key = 'speed' # the speed of inference (sec/word)
        self.ppl_key = 'perplexity' # the perplexity score to evaluate a model's average confidence(exp of negative log-likelihood) for its predictions
             