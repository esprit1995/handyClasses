import pandas as pd
from sklearn import preprocessing
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
from sklearn.model_selection import ShuffleSplit, TimeSeriesSplit

##################################################################################
###################         Helper functions     #################################
##################################################################################


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


def calculate_lgbm_feature_importances(df, target_name, date_columns=None, n_splits=3, timesplit=True):
    """
    Calculate feature importances of a given dataset based on naive LGBM run
    :param df: dataframe in question
    :param target_name: name of the column that contains target values
    :param date_columns: column names that contain data in date format. For some reason, it breaks LGBM
    :param n_splits: n_splits parameter for timeseries  split
    :param timesplit: whether to use timesplit instead of regular train_test split. Default True.
    """
    warnings.filterwarnings("ignore")

    # simplify the names as non-ASCII characters are not supported
    trainingData = df.drop(target_name, axis=1)
    if date_columns is not None:
        trainingData = trainingData.drop(date_columns, errors='ignore', axis=1)
    targetData = pd.Series(df[target_name].tolist())
    trainingData.columns = [str(col).encode('ascii', errors='ignore') for col in trainingData.columns]
    targetData.name = str(target_name).encode('ascii', errors='ignore')

    feature_importances = np.zeros(trainingData.shape[1])
    model = lgb.LGBMRegressor(objective='regression', boosting_type='gbdt')

    if timesplit:
        data_splitter = TimeSeriesSplit(n_splits=n_splits)
    else:
        data_splitter = ShuffleSplit(n_splits=n_splits)
    for train_index, test_index in data_splitter.split(trainingData.to_numpy()):
        # Split into training and validation set
        train_features = trainingData.iloc[train_index]
        train_y = targetData.iloc[train_index]
        valid_features = trainingData.iloc[test_index]
        valid_y = targetData.iloc[test_index]
        cat_features = trainingData.select_dtypes(exclude=numerics).columns.tolist()
        for col in cat_features:
            train_features.loc[:, col] = train_features[col].astype("category")
            valid_features.loc[:, col] = valid_features[col].astype("category")

        # Train using early stopping
        model.fit(train_features, train_y,
                  early_stopping_rounds=100,
                  eval_set=[(valid_features, valid_y)],
                  eval_metric='l2',
                  verbose=200)

        # Record the feature importances
        feature_importances += model.feature_importances_ / n_splits

    feature_importances = pd.DataFrame({'feature': list(trainingData.columns),
                                        'importance': feature_importances}).sort_values('importance',
                                                                                        ascending=False)

    return feature_importances


# credits: Shaked Zychlinski (https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9)
def cramers_v(x, y):
    """
    https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


class ColumnDropper:
    def __init__(self, df, target_name, NA_threshold=0.9, corr_threshold=0.7, chi2_threshold=0.9):
        """
        :param df: dataframe to analyze
        :param target_name: name of the column that contains the target values
        :param NA_threshold: maximum allowed rate of NA in a column
        :param corr_threshold: maximum allowed correlation between 2 vars (abs value)
        """
        self.target_name = target_name
        self.original_df = df.copy()
        self.df = df.copy()
        self.NA_threshold = NA_threshold
        self.corr_threshold = corr_threshold
        self.chi2_threshold = chi2_threshold

        # class variables to be defined later on:
        self.const_val_cols = None  # list of columns names with 0 variance
        self.sparse_cols = None  # list of columns with too many missing values
        self.correlated_cols_num = None  # num cols highly correlated to other num cols
        self.correlation_group_representatives_num = None  # num cols to represent other num cols correlated to them
        self.correlated_num_todrop = None  # correlated num cols that should be dropped
        self.cat_corr_frame = None  # confusion matrix for correlated categorical variables
        self.lgbm_importances = None  # data frame containing lgbm importances of chosen variables

    def get_const_val_cols(self):
        """
        finds columns with zero variance
        """
        self.const_val_cols = [col for col in self.df.columns if self.df[col].nunique() == 1]

    def get_sparse_cols(self):
        """
        finds columns that have a higher NA rate than allowed
        """
        self.sparse_cols = [col for col in self.df.columns if
                            self.df[col].isna().sum() / self.df.shape[0] > self.NA_threshold]

    def estimate_correlations_numerical(self):
        """
        carries out correlation analysis of the numerical columns of self.df
        for each group of correlated variables, chooses a single representative with highest normalized variance
        """
        newdf = self.df.select_dtypes(include=numerics)

        # get correlation matrix and NA everything that is below threshold
        corr_mtrx = newdf.corr()
        corr_mtrx = corr_mtrx[corr_mtrx > self.corr_threshold]

        # create a dictionary of format 'colname': [high_corr_varname1, hight_corr_varname2, ...]
        corr_dict = {}
        for col in corr_mtrx.columns:
            corr_dict[col] = corr_mtrx[col].dropna().index.tolist()

        # A correlated with B, B correlated with C => A correlated with C
        processed_cols = list()
        self.correlated_cols_num = list()
        self.correlation_group_representatives_num = list()
        min_max_scaler = preprocessing.MinMaxScaler()
        for colname in corr_dict.keys():
            if colname in processed_cols:
                continue
            correlation_group = corr_dict[colname]

            if len(correlation_group) < 2:
                continue

            processed_cols = processed_cols + correlation_group
            self.correlated_cols_num = self.correlated_cols_num + correlation_group

            # choose a correlation group representative with highest normalized variance
            max_var = 0
            cur_candidate = ''

            for candidate in correlation_group:
                if candidate == self.target_name:
                    continue
                cur_var = min_max_scaler.fit_transform(self.df[candidate].to_numpy().reshape(-1, 1)).var()
                if cur_var > max_var:
                    max_var = cur_var
                    cur_candidate = candidate
            self.correlation_group_representatives_num.append(cur_candidate)
        self.correlated_cols_num = list(set(self.correlated_cols_num).difference(set([self.target_name])))
        self.correlated_num_todrop = list(
            set(self.correlated_cols_num).difference(self.correlation_group_representatives_num))

    def estimate_correlations_categorical(self):
        """
        seek out correlated categorical variables using Cramer's V metric
        """
        newdf = self.df.select_dtypes(exclude=numerics)

        correlated_pairs_cat = list()
        # get all possible pairs of categorical variables
        colnames = newdf.columns
        pairs = list()
        for i in range(len(colnames) - 1):
            for j in range(i + 1, len(colnames)):
                # skip empty columns
                if self.df[colnames[i]].size == 0 or self.df[colnames[j]].size == 0:
                    continue
                pairs.append((colnames[i], colnames[j]))

        # compute Cramer's V for each pair, save correlated ones
        for pair in pairs:
            if cramers_v(self.df[pair[0]].fillna('NA'), self.df[pair[1]].fillna('NA')) > self.chi2_threshold:
                correlated_pairs_cat.append(pair)

        correlated_columns = list(set([item for t in correlated_pairs_cat for item in t]))
        self.cat_corr_frame = pd.DataFrame(columns=correlated_columns, index=correlated_columns)
        for colname1 in self.cat_corr_frame.index:
            for colname2 in self.cat_corr_frame.columns:
                if (colname1, colname2) in correlated_pairs_cat or colname1 == colname2:
                    self.cat_corr_frame.loc[colname1, colname2] = 1
                else:
                    self.cat_corr_frame.loc[colname1, colname2] = 0

    def remove_sparse_const(self):
        """
        removes sparse columns and columns with constant values,
        Assumes target variable can be neither
        :return:
        """
        self.df = self.df.drop((self.const_val_cols + self.sparse_cols), axis=1)

    def show_cat_corr_heatmap(self, height=10, width=10):
        """
        visualize correlations between categorical variables
        """
        figure, ax1 = plt.subplots(figsize=(width, height))
        sns.heatmap(self.cat_corr_frame, linewidths=1, linecolor='red', ax=ax1)

        #########################################################
        # workaround for a bug in matplotlib-seaborn interaction.
        #########################################################
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.5  # Add 0.5 to the bottom
        t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values
        #########################################################

    def remove_corr_num(self):
        """
        remove correlated numerical columns and leave only one
        representative for each correlation group. self.analyze must be performed prior to this
        function
        :return: void
        """
        if self.target_name in self.correlated_num_todrop:
            self.correlated_num_todrop = list(set(self.correlated_num_todrop).difference(set([self.target_name])))
        self.df = self.df.drop(self.correlated_num_todrop, axis=1)

    def compute_lgbm_importances(self, date_columns=None, n_splits=3, timesplit=True):
        """
        estimate feature importances using the naive lgbm algorithm
        :param n_splits: n_splits argument for lgbm algorithm
        :param date_columns: column names that contain dates. These break lgbm for some reason.
        :param timesplit: whether to use timesplit instead of regular train_test split. Default True
        :return: void, creates self.lgbm_importances
        """
        try:
            self.analyze()
            self.remove_corr_num()
        except Exception:
            pass
        self.lgbm_importances = calculate_lgbm_feature_importances(self.df, self.target_name, date_columns, n_splits,
                                                                   timesplit)

    def analyze(self, drop_sparse_const=True):
        self.get_const_val_cols()
        self.get_sparse_cols()
        if drop_sparse_const:
            self.remove_sparse_const()
        self.estimate_correlations_numerical()
        self.estimate_correlations_categorical()

    def get_filtered_df(self, remove_zero_importance=False, top_n_lgbm=None,
                        date_columns=None, n_splits=3, timesplit=True, predifined_to_keep=None):
        """
        get a dataframe with potentially useful features according to initial filtering as well as lgbm naive estimation
        :param remove_zero_importance: whether to remove features that have zero importance according to naive lgbm
        :param top_n_lgbm: if an integer is given, keeps only top top_n_lgbm features according to naive lgbm run
        :param date_columns: columns containing dates (for lgbm correct run)
        :param n_splits: lgbm parameter
        :param timesplit: whether timesplit is preferred to regular shuffle.
        :param predifined_to_keep: allows to pass a list of columns to keep regardless of analysis
        :return:
        """
        # resetting the dataframe
        self.df = self.original_df.copy()
        self.analyze(drop_sparse_const=True)
        self.remove_corr_num()
        self.compute_lgbm_importances(date_columns, n_splits, timesplit)
        features_to_keep = pd.DataFrame({'varname': self.df.drop(self.target_name, axis=1).columns})
        if remove_zero_importance:
            non_zero_indices = self.lgbm_importances.reset_index().loc[self.lgbm_importances.importance != 0][
                'index'].tolist()
            features_to_keep = features_to_keep.iloc[non_zero_indices]
        if top_n_lgbm is not None:
            try:
                top_n_indices = self.lgbm_importances \
                    .reset_index() \
                    .sort_values('importance', ascending=False) \
                    .head(top_n_lgbm)['index'].tolist()
                features_to_keep = features_to_keep.iloc[top_n_indices]
            except Exception as e:
                print('Selection of top n features failed with error: ' + str(e))
                print('Try asking for less features or setting remove_zero_importance to False')
                return None
        colnames_to_keep = features_to_keep.varname.tolist() + [self.target_name]
        if predifined_to_keep is not None:
            try:
                colnames_to_keep = list(set(colnames_to_keep + predifined_to_keep))
            except Exception as e:
                print('!!Warning!!: error while merging predefined columns: ' + str(e))
                pass
        return self.df[colnames_to_keep]
