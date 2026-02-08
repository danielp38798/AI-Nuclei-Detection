# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:05:58 2025
Version 3.0 from 01.04.2025
@author: pod38798
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as st
import os

from statannotations.Annotator import Annotator
from AutoStatistics import AutoStatistics as ast

    
import matplotlib.patches as patches


sns.set(style="whitegrid", font="CMU Serif", font_scale=1.0)

class SeaPlot:
    def __init__(self, verbose=False, output_dir=None):
        self.verbose = verbose
        self.color = {'boxplot': '#FDFEFE',
                      'median':'#117A65',
                      'mean': '#2980B9',
                      'ci': '#C0392B',
                      'flier': "#17202A",
                      'separator': '#BFC9CA',
                      'annotation': '#17202A',
                      'lineplot1': '#E74C3C',
                      'lineplot2': '#1ABC9C',
                      'lineplot3': '#2E86C1',
                      'ul_inlet': '#641E16',
                      'ul_outlet': '#D98880',
                      'ur_inlet': '#512E5F',
                      'ur_outlet': '#C39BD3',
                      'll_inlet': '#154360',
                      'll_outlet': '#7FB3D5',
                      'lr_inlet': '#186A3B',
                      'lr_outlet': '#82E0AA'
                      }
        self.marker = {'mean': 'x',
                       'ci': 'x',
                       'flier': 'd',
                       'annotation': 'star',
                       'median': '_',
                       'lineplot': 'x'}
        self.output_dir = output_dir
        #https://htmlcolorcodes.com/color-chart/

    def totalBoxPlot(self, data1:pd.DataFrame=None, data2:pd.DataFrame=None,title:str=None, 
                     xlabel:str='xlabel', ylabel:str='ylabel',
                     ci:bool=True, ci_alpha:float=0.95, max_y_val:int=100):
        

        modality_list = ["GT Annotation", "AI Prediction"]
        modality = list(range(len(modality_list)))
        
        data = pd.concat([data1, data2], axis=1, ignore_index=False)
        data.columns = modality
        fig = sns.boxplot(data=data, color=self.color['boxplot'], width=0.6,
                    medianprops=dict(color=self.color['median'], alpha=0.7),
                    flierprops=dict(markerfacecolor="white", 
                                    marker=self.marker['flier'],
                                    markeredgecolor=self.color['flier'], 
                                    markersize=8),
                    showmeans=True, 
                    meanprops=dict(marker=self.marker['mean'], 
                                   markerfacecolor="white",
                                     markeredgecolor=self.color['mean'], 
                                     markersize=8), zorder=1)
        
        fig.set(xlabel=xlabel, ylabel=ylabel, title=title, xticklabels=modality_list,
                #yticks=[0, 25, 50, 75])
                yticks=[0, 0.25*max_y_val, 0.50*max_y_val, 0.75*max_y_val])
        plt.vlines(0.5, 0, 0.75*max_y_val, colors=self.color['separator'])

        # place modality on top 
        #top_xticks = fig.secondary_xaxis(location=1)
        #top_xticks.set_xticks([0, 1, 2], 
        #               labels=modality)
        #top_xticks.set_xticks([0, 1],
        #               labels=modality_list)
        # insert number of data points
        fig.text(x=0, y=max_y_val, s=f'n={len(data1)}', horizontalalignment='center') # y=352
        fig.text(x=1, y=max_y_val, s=f'n={len(data2)}', horizontalalignment='center')

        # Konfidenzintervall einfügen
        if ci:
            ci_low_1, ci_high_1 = self.ci(data=data1, alpha=ci_alpha)
            plt.scatter([0, 0], [ci_low_1, ci_high_1], 
                        color=self.color['ci'], lw=1, marker=self.marker['ci'],
                        zorder=2)
            ci_low_2, ci_high_2 = self.ci(data=data2, alpha=ci_alpha)
            plt.scatter([1, 1], [ci_low_2, ci_high_2], 
                        color=self.color['ci'], lw=1, marker=self.marker['ci'],
                        zorder=2)
            
        #pairs = [(0, 1),(1,2), (0,2)]
        pairs = [(0, 1)]
        data = data.astype(float)
        
        #p_values = [mannwhitneyu(data.iloc[:,0], data.iloc[:,1]).pvalue,
         #           mannwhitneyu(data.iloc[:,1], data.iloc[:,2]).pvalue,
         #           mannwhitneyu(data.iloc[:,0], data.iloc[:,2]).pvalue]
         
        test_1_dict = ast.test_significance(self, data1=data.iloc[:,0], data2=data.iloc[:,1], paired=False)
        
        p_values = [test_1_dict['pvalue']]
          
        annotator = Annotator(fig, pairs, data=data)
        annotator.configure(test=None, 
                            text_format=self.marker['annotation'], 
                            loc='outside', color=self.color['annotation'],
                            line_width=1)
        annotator.set_pvalues(p_values)
        annotator.annotate()
        plt.ylim(0,max_y_val*1.1)
        #plt.ylim(0,100)
        
        # insert used stat test
        fig.text(x=0.5, y=max_y_val*0.9, s=test_1_dict['test_used'], horizontalalignment='center')
        #fig.text(x=1.5, y=210, s=test_2_dict['test_used'], horizontalalignment='center')
        #fig.text(x=1, y=230, s=test_3_dict['test_used'], horizontalalignment='center')
        

    
    

    def ci(self, data:pd.DataFrame=None, alpha:float = 0.95) -> (float, float):
        """
        Function to calculate the confidence interval of data set

        Parameters
        ----------
        data : pd.DataFrame
            DESCRIPTION. Data set for calculation. The default is None.
        alpha : float, optional
            DESCRIPTION. Definition of interval. The default is 0.95.

        Returns
        -------
        (float, float)
            DESCRIPTION. Interval returned as tuple of floats

        """
        self.ci_low, self.ci_high = st.norm.interval(confidence=alpha, loc=np.mean(data), 
                scale=st.sem(data))
        return self.ci_low, self.ci_high
    
    def is_normal_distribution(self, data:pd.DataFrame=None, 
                               significance_level:float=0.05) ->bool:
        """
        Überprüft, ob der gegebene Datensatz normalverteilt ist, indem der Shapiro-Wilk-Test verwendet wird.
        
        Parameter:
        data (array-like): Der zu testende Datensatz.
        significance_level (float): Das Signifikanzniveau für den Test. Standard ist 0,05.
        
        Rückgabe:
        bool: True, wenn der Datensatz normalverteilt ist, sonst False.
        float: p-Wert
        """
        stat, self.p_value = st.shapiro(data)
        self.gauss = self.p_value > significance_level
        return self.gauss, self.p_value
    
    def compare_detection_models(self, data: pd.DataFrame = None, 
                                xticks_labels: list = None,
                                title: str = None,
                                figsize: tuple = None,
                                step: float = 0.5,
                                lower_limit: float = -0.3,
                                max_y_val: float = 100,
                                secondary_xaxis_location: float = 2,
                                n_pos: float = 0.8,
                                x_bar_pos: float = 0.9,
                                md_pos: float = 0.95,
                                y_bar_lim: float = 1.0,
                                xlabel: str = 'xlabel', ylabel: str = 'ylabel',
                                ci: bool = False, ci_alpha: float = 0.95, relative: bool = False,
                                output_name: str = None,
                                text_size: int = 9) -> str:
        """
        Compare detection models and plot boxplots with fixed text size.

        Parameters
        ----------
        text_size : int, optional
            Fixed text size for all plot text elements. Default is 12.
        """
        sources = list(range(data.shape[1]))
        data.columns = sources

        # adapt the size of the figure to the number of columns
        if figsize is None:
            figsize = (len(sources) * 3, len(sources)*1.5)
        fig, ax = plt.subplots(figsize=figsize)

        sns.set(style="whitegrid", font="Times New Roman", font_scale=0.9)
        ax = sns.boxplot(data=data, color=self.color['boxplot'],
                        medianprops=dict(color=self.color['median'], alpha=0.7),
                        flierprops=dict(markerfacecolor="white",
                                        marker=self.marker['flier'],
                                        markeredgecolor=self.color['flier'],
                                        markersize=8),
                        showmeans=True,
                        showfliers=False,
                        meanprops=dict(marker=self.marker['mean'],
                                        markerfacecolor="white",
                                        markeredgecolor=self.color['mean'],
                                        markersize=8)
                        )

        # Set fixed text size for all relevant elements
        plt.xticks(fontsize=text_size)
        plt.yticks(fontsize=text_size)
        ax.set_xlabel(xlabel, fontsize=text_size)
        ax.set_ylabel(ylabel, fontsize=text_size)
        ax.set_title(title, fontsize=text_size)
        
        if ci:
            for i in range(data.shape[1]):
                ci_low, ci_high = self.ci(data=data.iloc[:, i], alpha=ci_alpha)
                plt.scatter([i, i], [ci_low, ci_high], color=self.color['ci'], lw=1, marker=self.marker['ci'], zorder=2)

        ax.set(xlabel=xlabel, ylabel=ylabel, title=title,
               xticklabels=[i.replace("mask_rcnn_", "") for i in xticks_labels])
        ax.tick_params(axis='x', rotation=60, labelsize=text_size)
        # Move the secondary x-axis higher
        top_xticks = ax.secondary_xaxis(location=secondary_xaxis_location)
        top_xticks.set_xticks([2, 5, 7.0, 8.0],
                            labels=['Mask R-CNN ResNet 50', 
                                    'Mask R-CNN ResNet 101', "Cellpose", "StarDist"])
        top_xticks.set_xlabel('', fontsize=text_size)
        top_xticks.set_xticklabels(['Mask R-CNN ResNet 50', 
                                    'Mask R-CNN ResNet 101', "Cellpose", "StarDist"], fontsize=text_size)

        # Add vertical lines to separate the groups
        plt.axvline(0.5, color=self.color['separator'])
        plt.axvline(3.5, color=self.color['separator'])
        plt.axvline(6.5, color=self.color['separator'])
        plt.axvline(8.5, color=self.color['separator'])
        data = data.dropna().astype(float)

        # Significance tests and annotations
        pairs = [(0, i) for i in range(1, data.shape[1])]
        test_dicts = [ast.test_significance(self, data1=data.iloc[:, 0], data2=data.iloc[:, i], paired=False) for i in range(1, data.shape[1])]
        mean_values = [data.iloc[:, i].mean() for i in range(data.shape[1])]
        median_values = [data.iloc[:, i].median() for i in range(data.shape[1])]

        annotator = Annotator(ax, pairs, data=data)
        annotator.configure(test="Mann-Whitney", text_format="simple")
        annotator.apply_and_annotate()

        for i, tick in enumerate(ax.get_xticks()):
            ax.scatter(tick, lower_limit, color='black', marker='|', s=20, zorder=3)

        plt.ylim(lower_limit, y_bar_lim)

        # Position the annotations above the significance bars
        for i in range(data.shape[1]):
            n = len(data.iloc[:, i])
            n_formatted = f"{n:,}"
            ax.text(x=i, y=n_pos, s=f'n={n_formatted}', horizontalalignment='center', fontsize=text_size)
            x_bar = r"$\bar{x}$"
            ax.text(x=i, y=x_bar_pos, s=f"{x_bar}: {mean_values[i]:.3f}", horizontalalignment='center', fontsize=text_size)
            ax.text(x=i, y=md_pos, s=f"md: {median_values[i]:.3f}", horizontalalignment='center', fontsize=text_size)

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            plot_path = os.path.join(self.output_dir, f"{output_name}")
            plt.savefig(plot_path, bbox_inches='tight')
            return plot_path
        else:
            plt.show()
            return None
def main():
    sp = SeaPlot()
    
        
if __name__ == "__main__":
    main()
    
