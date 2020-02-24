import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import missingno as msn
import warnings
import math, datetime
import pandas as pd
from pathlib import Path


class QuickViz:
    def __init__(self,
                 df,
                 cat_cols=None,
                 num_cols=None,
                 plot_width=7,
                 plot_height=7,
                 target=None,
                 target_type=None):
        """
        :param df: dataframe to explore
        :param target: the colname of the target variable of the dataset
        :param target_type: either 'num' or 'cat'
        """
        self.df = df.copy()
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.target = target
        self.target_type = target_type

    def missing_vals(self):
        """
        Wrapper for missingno matrix. Renders missing values matrix for self.df
        """
        warnings.filterwarnings("ignore")
        msn.matrix(self.df)
        plt.show()
        warnings.filterwarnings("default")

    def viz_column(self, colname, save=False, save_to_path=None):
        """
        get some quick visuals of a numeric column. Includes:
            histogram with missing values count
            boxplot
            scatterplot vs. target (if target defined and is numeric)
            barplot vs. target (if target defined and is categorical)
        :param colname: column of self.df to be examined
        :param save: whether to save plots in a file (True, False), default False
        :param save_to_path: if save is set True, the path to directory where plots should be put
        """
        rows = 1 + (self.target is not None)
        cols = 2

        fig = plt.figure(figsize=(self.plot_width * cols, self.plot_height * rows))
        gs = gridspec.GridSpec(nrows=rows, ncols=cols, height_ratios=[1] * rows)
        fig.suptitle('Variable: ' + str(colname))
        # add distplot
        ax_distplot = fig.add_subplot(gs[0, 0])
        sns.distplot(self.df[colname], ax=ax_distplot)
        ax_distplot.set_title('histogram. Missing vals: ' + str(self.df[colname].isna().sum()))

        # add boxplot
        ax_boxplot = fig.add_subplot(gs[0, 1])
        sns.boxplot(self.df[colname], ax=ax_boxplot)
        ax_boxplot.set_title('boxplot')

        # add scatterplot (if target defined and numeric)
        if (self.target is not None and self.target_type == 'num'):
            ax_scatter = fig.add_subplot(gs[1, 0])
            sns.scatterplot(self.df[colname], self.df[self.target], ax=ax_scatter)
            ax_scatter.set_title('scatterplot vs. target variable')

        # add barplot (if target defined and categorical)
        if (self.target is not None and self.target_type == 'cat'):
            ax_barplot = fig.add_subplot(gs[1, 0])
            sns.barplot(x=self.target, y=colname, data=self.df, ax=ax_barplot)
            ax_barplot.set_title('barplot vs target variable')

        if save:
            plt.savefig(Path(save_to_path) / (''.join(c for c in colname if c.isalpha()) + '_analysis.png'))
        plt.show()

    def viz_num_vs_num(self, colname1, colname2, line = False, hue = None, save=False, save_to_path=None):
        """
        Look into relationship between two numeric (continuous or discrete, but not categorical) features. Includes:
            scatterplot colname1 x colname2 of self.df
            computes Pearson correlation
            shows if there is correlation in missing values (are they missing at the same time?)
        :param colname1: name of the first column for analysis in self.df
        :param colname2: name of the second column for analysis in self.df
        :param line: boolean, whether to plot a lineplot instead of scatterplot
        :param hue: defines the coloring of data points/lines
        :param save: whether to save plots in a file (True, False), default False
        :param save_to_path: if save is set True, the path to directory where plots should be put
        """
        fig = plt.figure(figsize=(self.plot_width * 2, self.plot_height))
        gs = gridspec.GridSpec(nrows=1, ncols=2, height_ratios=[1])
        fig.suptitle(str(colname1) + ' to ' + str(colname2) + ' relation')

        if not line:
            # add scatterplot
            ax_scatter = fig.add_subplot(gs[0, 0])
            sns.scatterplot(colname1, colname2, data=self.df, ax=ax_scatter, hue = hue)
            # set title to the value of Pearson correlation
            corr_val = self.df[[colname1, colname2]].corr().iloc[0, 1]
            ax_scatter.set_title('Pearson correlation value: ' + str(corr_val))
        else:
            # add lineplot
            ax_scatter = fig.add_subplot(gs[0, 0])
            sns.lineplot(colname1, colname2, data=self.df, ax=ax_scatter, hue = hue)
            # set title to the value of Pearson correlation
            corr_val = self.df[[colname1, colname2]].corr().iloc[0, 1]
            ax_scatter.set_title('Pearson correlation value: ' + str(corr_val))
        # add missing values matrix
        ax_missing = fig.add_subplot(gs[0, 1])
        msn.matrix(self.df[[colname1, colname2]], ax=ax_missing, sparkline=False)
        ax_missing.axes.get_xaxis().set_ticks([])
        ax_missing.set_title('missing values matrix')

        if save:
            plt.savefig(Path(save_to_path) / (''.join(c for c in colname1 if c.isalpha()) + \
                                              '_vs_' + ''.join(c for c in colname2 if c.isalpha()) + '_analysis.png'))
        plt.show()


    def corr_heatmap(self, save=False, save_to_path=None):
        """
        Draw a diagonal correlation heatmap using seaborn library. 
        :param save: whether to save plots in a file (True, False), default False
        :param save_to_path: if save is set True, the path to directory where plots should be put
        """
        relevant_columns = list()
        if self.num_cols is not None:
            relevant_columns = [col for col in self.num_cols]
        elif self.cat_cols is not None:
            relevant_cols = [col for col in self.df.columns if col not in self.cat_cols]
        else:
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            relevant_cols = self.df.select_dtypes(include=numerics).columns

        df_subset = self.df[relevant_cols].copy()
        # limiting the length of column names
        df_subset.columns = [str(col).split('(')[0] for col in df_subset.columns]
        df_subset.columns = [col[0:min(20, len(col))] for col in df_subset.columns]
        corr_mtrx = df_subset.corr()
        # mask for the upper triangle
        mask = np.zeros_like(corr_mtrx, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        figure, ax = plt.subplots(figsize=(self.plot_width * 2, self.plot_height * 2))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(corr_mtrx, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

        ax.set_title('Correlation heatmap for numerical columns')
        if save:
            plt.savefig(Path(save_to_path) / ('Correlation_heatmap.png'))
        plt.show()
