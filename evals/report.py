############################       base libs import     ############################

import os
from tqdm import tqdm
import json
from typing import Any, Tuple, List, Union, Optional
import time

############################       dev libs import     ############################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import seaborn as sns
import pandas as pd
from tabulate import tabulate

############################       own libs import     ############################

from base import BaseModule

############################     reporter class    ############################


class Reporter(BaseModule):
    def __init__(self) -> None:
        super().__init__()
        
        self.score_key = self.score_output_key
        
        self.error_list = []
        
    def log_error(self, error, model_name, dataset_name, sample_idx=None):
        self.error_list.append({
            "error_message": str(error),
            "error_time": time.strftime('%Y-%m-%d %H:%M'),
            "error_model": model_name,
            "error_dataset": dataset_name,
            "error_sample_idx": sample_idx if sample_idx is not None else -1,
        })
        
    def report_error(self, path: str) -> None:
        # write json file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.error_list, f, ensure_ascii=False)
         
    def report_outputs(self, outputs: dict, path: str) -> None:
        # wrute json file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, ensure_ascii=False)
        
    def report_leaderboard(self, outputs: dict, path: str) -> None:
        """
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
        Functions:
            this func is used in report func to report the leaderbord.md by turning the output dict into the score markdown string
        """ 
        
        def report_leaderboard_for_dataset(output, dataset_name):
            score_dicts = {}
            for model_name, model_output in output.items():
                for split_name, split_output in model_output[self.score_key].items():
                    for score_name, score_value in split_output.items():
                        score_dicts.setdefault(split_name, {}).setdefault(model_name, {})[score_name] = score_value
            
            score_dfs, unit = {}, '(sec/word)'
            for split_name, score_dict in score_dicts.items():   
                score_df = pd.DataFrame.from_dict(score_dict, orient='index')
                for col in score_df.columns:
                    if col != self.num_samples_key:
                        score_df[col] = score_df[col].apply(lambda x: "{:.2e}".format(x))
                score_df.rename(columns={self.speed_key: self.speed_key + unit}, inplace=True)
                score_dfs[split_name] = score_df 
            
            score_str = f"## LeaderBoard for {dataset_name}" +  f"\n => *Each sub-leaderboard is for a certain length range, " + \
                        f"where each row shows the scores for one model on each metric equipped in the subtask {dataset_name}* \n"
            for split_name, score_df in score_dfs.items():
                score_str += f"### {split_name}: \n"
                df_str = tabulate(score_df, tablefmt="pipe", headers="keys",
                                disable_numparse=True,
                                )
                score_str += df_str + "\n"
                score_str += "--- \n"
                
            return score_str


        score_str = f"# LeaderBoard \n\n"
        
        for dataset_name, output in outputs.items():
            score_str += report_leaderboard_for_dataset(output, dataset_name)
            score_str += "\n\n"
            
        # write md file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(score_str)
            
        return score_str
        
    def report_lengthplot(self, outputs: dict, path: str, unit: str = 'ms') -> None:
        """
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
        Functions:
            this func is used in report func to report the plot.png by turning the output dict into the length-score multi-curve subplots
        """
        
        # get score dicts
        score_dicts = {}
        unit_factor = {'ms': 1000, 's': 1}
        for dataset_name, output in outputs.items():
            for model_name, model_output in output.items():
                for split_name, split_output in model_output[self.score_key].items():
                    # if split_name == 'all':
                    #     continue
                    for score_name, score_value in split_output.items():
                        score_name += (f"({unit} / word)" if score_name == self.speed_key else "") # add unit for speed
                        score_dicts.setdefault(dataset_name, {}).setdefault(score_name, {}).setdefault('length', []).append(split_name)
                        score_dicts.setdefault(dataset_name, {}).setdefault(score_name, {}).setdefault('model', []).append(model_name)
                        score_dicts.setdefault(dataset_name, {}).setdefault(score_name, {}).setdefault(score_name, []).append( 
                            score_value * (unit_factor[unit] if self.speed_key in score_name else 1) 
                            # use miliseconds instead of seconds for better visulization
                        )
        
        # sort the dataset with num of metrics
        num_metrics_dict = {k: len(v) for k, v in score_dicts.items()}
        dataset_names = sorted(list(score_dicts.keys()), key=lambda d: num_metrics_dict[d])
        score_dicts = {d: score_dicts[d] for d in dataset_names}
        score_dicts[dataset_names[0]] 
        
        # get the num_datasets(nrows), num_metrics(ncols) and the legend index
        num_metrics = np.array([len(v) for v in score_dicts.values()])
        num_datasets, max_num_metrics, min_num_metrics = len(score_dicts), num_metrics.max(), num_metrics.min()
        if min_num_metrics == max_num_metrics: # the plot will be full cols
            max_num_metrics += 1 # add one empty col to show the legend on the first row, last col for all the models
            legend_idx = (0, max_num_metrics-1)
        else: # use the first row's first empty subplot to show the legend
            legend_idx = (0, min_num_metrics)
        
        # create the figure    
        fig, axs = plt.subplots(
            figsize = (4*max_num_metrics, 4*num_datasets), # each row is for a dataset, and each column is for a metric score type
            nrows=num_datasets, 
            ncols=max_num_metrics,
        )
        if not isinstance(axs, np.ndarray):
            axs = np.array([[axs]])
        elif len(axs.shape) == 1: # only one row
            axs = np.array([axs])
            plt.subplots_adjust(wspace=0.4)
        else:
            plt.subplots_adjust(hspace=0.4, wspace=0.4)
        
        # draw the plots
        legend_data, legend_score_name = None, "" # a placeholder data to show the model legend
        for i, (dataset_name, score_dict) in enumerate(score_dicts.items()):
            for j, (score_name, data) in enumerate(score_dict.items()):
                axs[i,j].set_title(f"{score_name} on {dataset_name}", fontdict={'size':6})
                if score_name == self.num_samples_key: # ignore the num_samples but plot the length hist
                    sns.barplot(
                        data=data, x='length', y=score_name, 
                        palette='husl',
                        ax=axs[i,j]
                    )
                    axs[i,j].set_ylabel(f"{score_name}")
                    axs[i,j].set_xlabel("length")
                else:
                    if legend_data is None: legend_data, legend_score_name = data, score_name
                    sns.lineplot(
                        data=data, x='length', y=score_name, # add unit
                        hue='model', style='model', markers=True, dashes=False, palette='husl',
                        ax=axs[i,j]
                    )
                    axs[i,j].legend_.remove() # remove the legend on the plot to avoid overlapping problem
            for j in range(len(score_dict), max_num_metrics): # remove the redunant plot
                if (i,j) == legend_idx: # to show the model legend
                    sns.lineplot(
                        data=legend_data, x='length', y=legend_score_name, # add unit
                        hue='model', style='model', markers=True, dashes=False, palette='husl',
                        ax=axs[i,j]
                    )
                    for line in axs[i,j].lines: line.set_visible(False) 
                axs[i,j].axis('off')
                
        # global title
        fig.suptitle(f"Score / Speed / Number of Samples - Length Plots")
        
        # write png file
        fig.savefig(path, dpi=300)

    def report_radarchart(self, outputs: dict, path: str) -> None:
        """
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
        Functions:
            this func is used in report func to report the radarchart.png by turning the output dict into the radarchart plot
        """ 
        
        def radar_factory(num_vars, frame='circle'):
            """
            Create a radar chart with `num_vars` axes.

            This function creates a RadarAxes projection and registers it.

            Parameters
            ----------
            num_vars : int
                Number of variables for radar chart.
            frame : {'circle', 'polygon'}
                Shape of frame surrounding axes.

            """
            # calculate evenly-spaced axis angles
            theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

            # class RadarTransform(PolarAxes.PolarTransform):

            #     def transform_path_non_affine(self, path):
            #         # Paths with non-unit interpolation steps correspond to gridlines,
            #         # in which case we force interpolation (to defeat PolarTransform's
            #         # autoconversion to circular arcs).
            #         if path._interpolation_steps > 1:
            #             path = path.interpolated(num_vars)
            #         return Path(self.transform(path.vertices), path.codes)

            class RadarAxes(PolarAxes):

                name = 'radar'
                # PolarTransform = RadarTransform

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    # rotate plot such that the first axis is at the top
                    self.set_theta_zero_location('N')

                def fill(self, *args, closed=True, **kwargs):
                    """Override fill so that line is closed by default"""
                    return super().fill(closed=closed, *args, **kwargs)

                def plot(self, *args, **kwargs):
                    """Override plot so that line is closed by default"""
                    lines = super().plot(*args, **kwargs)
                    for line in lines:
                        self._close_line(line)

                def _close_line(self, line):
                    x, y = line.get_data()
                    # FIXME: markers at x[0], y[0] get doubled-up
                    if x[0] != x[-1]:
                        x = np.append(x, x[0])
                        y = np.append(y, y[0])
                        line.set_data(x, y)

                def set_varlabels(self, labels):
                    self.set_thetagrids(np.degrees(theta), labels, fontsize=8)
                    
                    # NOTE: enhance the radar chart xtick labels to avoid overlapping problem
                    for label, angle in zip(self.get_xticklabels(), theta):
                        if 0 < angle < np.pi:
                            label.set_horizontalalignment('right')
                        elif angle > np.pi:
                            label.set_horizontalalignment('left')

                def _gen_axes_patch(self):
                    # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
                    # in axes coordinates.
                    if frame == 'circle':
                        return Circle((0.5, 0.5), 0.5)
                    elif frame == 'polygon':
                        return RegularPolygon((0.5, 0.5), num_vars,
                                            radius=.5, edgecolor="k")
                    else:
                        raise ValueError("Unknown value for 'frame': %s" % frame)

                def _gen_axes_spines(self):
                    if frame == 'circle':
                        return super()._gen_axes_spines()
                    elif frame == 'polygon':
                        # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                        spine = Spine(axes=self,
                                    spine_type='circle',
                                    path=Path.unit_regular_polygon(num_vars))
                        # unit_regular_polygon gives a polygon of radius 1 centered at
                        # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                        # 0.5) in axes coordinates.
                        spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                            + self.transAxes)
                        return {'polar': spine}
                    else:
                        raise ValueError("Unknown value for 'frame': %s" % frame)

            register_projection(RadarAxes)
            return theta

        def extract_data(output: dict) -> list:
            """extract data from the outputs dict
            default use the first metric as the metric to represent the model's performance
            """
            fake_key = '~fake'
            # get dataset list (>=3, if not, generate some fake datasets to fill the blank)
            dataset_list = list(output.keys())
            while len(dataset_list) < 3: dataset_list.append(f'{fake_key}{3-len(dataset_list)}')
            # get model list
            model_list = []
            for dataset_output in output.values():
                model_list.extend(list(dataset_output.keys()))
            model_list = list(dict.fromkeys(model_list)) # remove the repeated models
            # get split list
            # split_list = []
            # for dataset_output in output.values(): 
            #     for model_output in dataset_output.values(): 
            #         split_list.extend(list(model_output[self.score_key].keys()))
            # split_list = list(dict.fromkeys(split_list)) # remove the repeated split
            split_list = ['Norm Score', 'Norm Speed']
            
            # get the score and speed for each model on each dataset
            values = np.zeros((len(split_list), len(model_list), len(dataset_list)))
            for dataset_name, dataset_output in outputs.items():
                di = dataset_list.index(dataset_name)
                for model_name, model_output in dataset_output.items():
                    mi = model_list.index(model_name)
                    for split_name, split_output in model_output[self.score_key].items():
                        if split_name == 'all': # only keep the 'all' split to simplify the radarchart
                            # for score (take the first metric as the main metric)
                            score_value = next(iter(split_output.values())) # extract the first score
                            values[0, mi, di] = score_value
                            # for speed (sec / word)
                            values[1, mi, di] = split_output[self.speed_key]
                            break
                    
            # normalize the score and the speed for each dataset to [0, 1]
            # and enlarge the difference cuze the performances are often too close
            def normalize(a, factor=1, reverse=False):
                eps = 1e-5
                if len(a) == 1: return a / a.max()
                a *= factor
                if reverse: a = 1. / a # NOTE: for speed
                alpha = np.exp(a.min()/(a.max()+eps))
                a = a ** alpha # enlarge difference
                a /= (a.max() + eps) 
                return a
                    
            for si in range(len(values)):
                for di, dataset_name in enumerate(dataset_list):
                    if fake_key in dataset_name: # set fake dataset score value to 1. for visualization
                        values[si, :, di] = 1.
                    else:
                        values[si, :, di] = normalize(values[si, :, di], factor=1000,
                                                    reverse=si==1 # let the speed be greater be better
                                                    )
            
            # generate the data
            data = [dataset_list, model_list, ]
            for si in range(len(values)):
                data.append((
                    split_list[si], 
                    values[si].tolist(),
                ))
                
            return data
            
        # init data
        data = extract_data(outputs)
        N, spoke_labels, labels, cmap  = len(data[0]), data[0], data[1], plt.get_cmap('Set3')
        colors = [cmap((i+1) / len(labels)) for i in range(len(labels))]
        nrows, ncols = 2, int(np.ceil((len(data)-2) / 2))
        theta = radar_factory(N)
        # init figure
        fig, axs = plt.subplots(figsize=(6*ncols, 4*nrows), 
                                nrows=nrows, ncols=ncols,
                                subplot_kw=dict(projection='radar'))
        if not isinstance(axs, np.ndarray):
            axs = np.array([[axs]])
        elif len(axs.shape) == 1: # only one row
            axs = np.array([axs])
        fig.subplots_adjust(wspace=0.2, hspace=0.5, top=0.85, bottom=0.1, 
                            left=0.15, right=0.7)

        # plot the four length from the example data on separate axes
        for i, ax  in enumerate(axs.flat):
            if i >= len(data[2:]): # the blank subplot
                ax.axis('off')
                continue
            
            title, case_data =  data[2:][i]
            ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
            ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                        horizontalalignment='center', verticalalignment='center')
            for d, color in zip(case_data, colors):
                ax.plot(theta, d, color=color)
                ax.fill(theta, d, facecolor=color, alpha=0.5, label='_nolegend_')
            ax.set_varlabels(spoke_labels)
            ax.set_yscale('linear')
        
        # add legend relative to top-left plot
        # axs[0, 0].legend(labels, loc=(0.9, .95),
        #                 labelspacing=0.1, fontsize='small')
        axs[0, 0].legend(labels, loc='upper left', bbox_to_anchor=(0.95, 1.3), fontsize='small')

        fig.text(0.5, 0.965, 'Radar Chart for Normalized Score and Speed',
                horizontalalignment='center', color='black', weight='bold',
                size='large')

        # write png file
        fig.savefig(path, dpi=300)

    def collect_report(self, output_dir: str) -> dict:
        """
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
        Functions:
            collect all sub outputs and aggregate togeter
        """
        outputs = {}
        for filename in tqdm(os.listdir(output_dir)):
            if filename.endswith('json') and filename.startswith('output'):
                with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
                    suboutput = json.load(f) # only has one key: subtask_name
                    # specific output for certain model and certain dataset
                    if any([isinstance(v, str) for v in suboutput.values()]): 
                        content = {}
                        for k, v in suboutput.items():
                            if isinstance(v, str): continue
                            content[k] = v
                        if outputs == {} or suboutput['dataset_name'] not in outputs: # new dataset
                            outputs.setdefault(suboutput['dataset_name'], {}).setdefault(suboutput['model_name'], {}).update(content)
                        else: # the same dataset
                            if suboutput['model_name'] not in outputs[suboutput['dataset_name']]: # new model
                                outputs[suboutput['dataset_name']].setdefault(suboutput['model_name'], {}).update(content)
                            else:
                                print(f"\n=> Warning: The output for model {suboutput['model_name']} in dataset {suboutput['dataset_name']} is in conflict, maybe check farther") 
                    # complete output
                    else:
                        if outputs == {}:
                            outputs.update(suboutput)
                        else:
                            for dataset_name in suboutput:
                                if dataset_name not in outputs: # new dataset
                                    outputs[dataset_name] = suboutput[dataset_name]
                                else: # the same dataset
                                    for model_name in suboutput[dataset_name]:
                                        if model_name not in outputs[dataset_name]: # new model
                                            outputs[dataset_name][model_name] = suboutput[dataset_name][model_name]
                                        else: # the same model, do no overwrite for now
                                            print(f"\n=> Warning: The output for model {model_name} in dataset {dataset_name} is in conflict, maybe check farther")
                                        
        return outputs

