# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import warnings

warnings.filterwarnings('ignore')

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
from tqdm import tqdm

# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

import json
config = json.load(open("config.json"))

nrows = None
sub = pd.read_csv(config["sample_submission"], nrows=nrows)
len_sub = len(sub)

# Load csv
print("\n[+] Load csv ")
train_df = pd.read_csv(config["train_csv"], parse_dates=["activation_date"],
                       index_col="item_id", nrows=nrows)
train_df = train_df.replace(np.nan, -1, regex=True)
test_df = pd.read_csv(config["test_csv"], parse_dates=["activation_date"], index_col="item_id",
                      nrows=nrows)
test_df = test_df.replace(np.nan, -1, regex=True)

user_df = pd.read_csv('./aggregated_features.csv', nrows=nrows)
# city_region_unique = pd.read_csv('./avito_region_city_features.csv', nrows=nrows)

# Merge two dataframes
n_train = len(train_df)
df = pd.concat([train_df, test_df])
del train_df, test_df
gc.collect()

print("before:", df.shape)
df = pd.merge(left=df, right=user_df, how="left", on=["user_id"])
print("after :", df.shape)

# Fillin missing data
for col in ["description", "title", "param_1", "param_2", "param_3"]:
    df[col] = df[col].fillna(" ")

df["params"] = df.apply(lambda row: " ".join([
    str(row["param_1"]),
    str(row["param_2"]),
    str(row["param_3"]),
]), axis=1)

# # Fill-in missing value by mean
# for col in ["price", "image_top_1"]:
#     m = df[col].mean()
#     df[col] = df[col].fillna(-1, inplace=True)
#     df[col] = df[col].replace(-1, m, regex=True)

# Get log of price
df["price"] = df["price"].apply(np.log1p)
df["price"] = df["price"].apply(lambda x: -1 if x == -np.inf else x)

# df['city'] = df['city'] + '_' + df['region']
df['no_img'] = pd.isna(df.image).astype(int)
df['no_dsc'] = pd.isna(df.description).astype(int)
df['no_p1'] = pd.isna(df.param_1).astype(int)
df['no_p2'] = pd.isna(df.param_2).astype(int)
df['no_p3'] = pd.isna(df.param_3).astype(int)
df['weekday'] = df['activation_date'].dt.weekday
# df['monthday'] = df['activation_date'].dt.day
df["item_seq_bin"] = df["item_seq_number"] // 100
df["ads_count"] = df.groupby("user_id", as_index=False)["user_id"].transform(lambda s: s.count())

textfeats = ['description', 'params', 'title']
for col in textfeats:
    df[col] = df[col].astype(str)
    df[col] = df[col].fillna('NA')  # FILL NA
    df[col] = df[col].str.lower()  # Lowercase all text, so that capitalized words dont get treated differently
    df[col] = df[col].str.replace("[^[:alpha:]]", " ")
    df[col] = df[col].str.replace("\\s+", " ")
    # df[col + '_num_chars'] = df[col].apply(len)
    df[col + '_num_words'] = df[col].apply(lambda s: len(s.split()))
    # df[col + '_num_unique_words'] = df[col].apply(lambda s: len(set(w for w in s.split())))
    # df[col + '_words_vs_unique'] = df[col+'_num_unique_words'] / df[col+'_num_words'] * 100
    df[col + '_num_capE'] = df[col].str.count("[A-Z]")
    df[col + '_num_capR'] = df[col].str.count("[А-Я]")
    df[col + '_num_lowE'] = df[col].str.count("[a-z]")
    df[col + '_num_lowR'] = df[col].str.count("[а-я]")
    df[col + '_num_pun'] = df[col].str.count("[[:punct:]]")
    df[col + '_num_dig'] = df[col].str.count("[[:digit:]]")

cat_cols = ['region', 'city', 'parent_category_name', 'category_name',
            'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1', "item_seq_bin"]

# Encoder:
for col in cat_cols:
    df[col], _ = pd.factorize(df[col])

# print(df.head(5).T)
X = df[:n_train]
test = df[n_train:]

X_train_df, X_val_df = train_test_split(X, shuffle=True, test_size=0.1, random_state=42)


class FeaturesStatistics():
    def __init__(self, cols):
        self._stats = None
        self._agg_cols = cols

    def fit(self, df):
        '''
        Compute the mean and std of some features from a given data frame
        '''
        self._stats = {}

        # For each feature to be aggregated
        for c in tqdm(self._agg_cols, total=len(self._agg_cols)):
            # Compute the mean and std of the deal prob and the price.
            gp = df.groupby(c)[['deal_probability', 'price']]
            desc = gp.describe()
            self._stats[c] = desc[[('deal_probability', 'mean'), ('deal_probability', 'std'),
                                   ('price', 'mean'), ('price', 'std')]]

    def transform(self, df):
        '''
        Add the mean features statistics computed from another dataset.
        '''
        # For each feature to be aggregated
        for c in tqdm(self._agg_cols, total=len(self._agg_cols)):
            # Add the deal proba and price statistics corrresponding to the feature
            df[c + '_dp_mean'] = df[c].map(self._stats[c][('deal_probability', 'mean')])
            df[c + '_dp_std'] = df[c].map(self._stats[c][('deal_probability', 'std')])
            df[c + '_price_mean'] = df[c].map(self._stats[c][('price', 'mean')])
            df[c + '_price_std'] = df[c].map(self._stats[c][('price', 'std')])

            df[c + '_to_price'] = df.price / df[c + '_price_mean']
            df[c + '_to_price'] = df[c + '_to_price'].fillna(1.0)

    def fit_transform(self, df):
        '''
        First learn the feature statistics, then add them to the dataframe.
        '''
        self.fit(df)
        self.transform(df)


fStats = FeaturesStatistics(['region', 'city', 'parent_category_name', 'category_name',
                             'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1', "item_seq_bin"])

fStats.fit_transform(X_train_df)
fStats.transform(X_val_df)
fStats.transform(test)

columns_to_drop = ['title', 'description', 'params', 'image',
                   'activation_date', 'deal_probability', 'user_id']

y_train = X_train_df['deal_probability']
y_val = X_val_df['deal_probability']

X_train_df = X_train_df.fillna(-1)
X_val_df = X_val_df.fillna(-1)
test = test.fillna(-1)

X_train_df.drop(columns_to_drop, axis=1, inplace=True)
X_val_df.drop(columns_to_drop, axis=1, inplace=True)
test.drop(columns_to_drop, axis=1, inplace=True)


# test = test.replace(np.nan, -1, regex=True)


# print(X_train_df.columns.values[43])
# print(X_train_df.columns.values[50])
# print(X_train_df.columns.values[68])

# Prepare Categorical Variables
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]


categorical_features_pos = column_index(X_train_df, cat_cols)

# Train Model
print("Train CatBoost Decision Tree")
cb_model = CatBoostRegressor(iterations=5000,
                             learning_rate=0.08,
                             depth=16,
                             # loss_function='RMSE',
                             eval_metric='RMSE',
                             random_seed=23,  # reminder of my mortality
                             od_type='Iter',
                             l2_leaf_reg=11,
                             metric_period=50,
                             od_wait=200)
cb_model.fit(X_train_df, y_train,
             eval_set=(X_val_df, y_val),
             cat_features=categorical_features_pos,
             use_best_model=True,
             verbose=True)

print("Model Evaluation Stage")
catpred = cb_model.predict(test)
sub['deal_probability'] = catpred
sub['deal_probability'].clip(0.0, 1.0, inplace=True)
sub.to_csv('submission_catboost.csv', index=False)