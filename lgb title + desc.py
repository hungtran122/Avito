#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 19:07:14 2018

@author: pooh
"""

import gc
import lightgbm as lgb
# import xgboost as xgb
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

# import string


##############################################################################
NFOLDS = 5
SEED = 42


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool=True):
        if (seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntr,))
    oof_test = np.zeros((nte,))
    oof_test_skf = np.empty((NFOLDS, nte))

    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))


###############################################################################

patch = '/home/ai/Documents/AI/Kaggle qvi/input/'

nrows = None
data_tr1 = pd.read_csv(patch + 'train_norm.csv', index_col="item_id",
                       parse_dates=["activation_date"], nrows=nrows)
data_te1 = pd.read_csv(patch + 'test_norm.csv', index_col="item_id", parse_dates=["activation_date"], nrows=nrows)
user_df = pd.read_csv(patch + 'aggregated_features.csv', nrows=nrows)

data_tr = pd.read_csv(patch + 'train.csv', index_col="item_id", parse_dates=["activation_date"], nrows=nrows)

data_te = pd.read_csv(patch + 'test.csv', index_col="item_id", parse_dates=["activation_date"], nrows=nrows)

data_tr["description_norm"] = data_tr1["description_norm"]
data_te["description_norm"] = data_te1["description_norm"]

data_tr["title_norm"] = data_tr1["title_norm"]
data_te["title_norm"] = data_te1["title_norm"]

y = data_tr.deal_probability.copy()

ntr = len(data_tr)
nte = len(data_te)
full = pd.concat([data_tr, data_te])
full = full.merge(user_df, on='user_id', how='left')

del data_tr1, data_te1
del user_df
gc.collect()
cols_to_fill = ['description', 'param_1', 'param_2', 'param_3', 'description_norm']
full[cols_to_fill] = full[cols_to_fill].fillna(' ')

# full['params_feat'] = full.apply(lambda row: ' '.join([
#    str(row['param_1']),
#    str(row['param_2']),
#    str(row['param_3'])]), axis=1)

## Tramnsform log
eps = 0.001

full["price"] = np.log(full["price"] + eps)
full["price"].fillna(full["price"].mean(), inplace=True)
full["image_top_1"].fillna(full["image_top_1"].mean(), inplace=True)
full['avg_days_up_user'].fillna(-1, inplace=True)
full['avg_days_up_user'].fillna(-1, inplace=True)
full['avg_times_up_user'].fillna(-1, inplace=True)
full['n_user_items'].fillna(-1, inplace=True)

full['city'] = full['city'] + '_' + full['region']
full['has_image'] = pd.notnull(full.image).astype(int)
full['weekday'] = full['activation_date'].dt.weekday
full['ads_count'] = full.groupby('user_id', as_index=False)['user_id'].transform(lambda s: s.count())
# full.loc[full["item_seq_number"].value_counts()[full["item_seq_number"]].values < 25, "item_seq_number"] = 0
full["item_seq_bin"] = full["item_seq_number"] // 100

full['has_image'] = pd.notnull(full.image).astype(int)
full['has_desc'] = pd.notnull(full.description).astype(int)
full['has_p1'] = pd.notnull(full.param_1).astype(int)
full['has_p2'] = pd.notnull(full.param_2).astype(int)
full['has_p3'] = pd.notnull(full.param_3).astype(int)

# full[col + '_num_chars'] = full[col].apply(len)
textfeats1 = ['description', "title", 'param_1', 'param_2', 'param_3', 'description_norm', "title_norm"]
for col in textfeats1:
    full[col] = full[col].astype(str)
    full[col] = full[col].astype(str).fillna(' ')
    full[col] = full[col].str.lower()

textfeats2 = ['description', "title"]
for col in textfeats2:
    full[col + '_num_words'] = full[col].apply(lambda s: len(s.split()))
    full[col + '_num_unique_words'] = full[col].apply(lambda s: len(set(w for w in s.split())))
    full[col + '_words_vs_unique'] = full[col + '_num_unique_words'] / full[col + '_num_words'] * 100
    full[col + '_num_lowE'] = full[col].str.count("[a-z]")
    full[col + '_num_lowR'] = full[col].str.count("[а-я]")
    full[col + '_num_pun'] = full[col].str.count("[[:punct:]]")
    full[col + '_num_dig'] = full[col].str.count("[[:digit:]]")

full['param_2'] = full['param_1'] + ' ' + full['param_2']
# full['param_1'] = full['param_3']
full['param_3'] = full['param_2'] + ' ' + full['param_3']


###############################################################################
full['params'] = full['param_3']+' '+full['title_norm']
full['text']=full['description_norm']+ ' ' + full['title_norm']+ " "+full['category_name']
###############################################################################

names = ["city", "param_1", "user_id"]
for i in names:
    full.loc[full[i].value_counts()[full[i]].values < 100, i] = "Rare_value"

full.loc[full["image_top_1"].value_counts()[full["image_top_1"]].values < 150, "image_top_1"] = -1
full.loc[full["item_seq_number"].value_counts()[full["item_seq_number"]].values < 150, "item_seq_number"] = -1
###############################################################################
cat_cols = ['user_id', 'region', 'city', 'category_name', "parent_category_name",
            'param_1', 'param_2', 'param_3', 'user_type',
            'weekday', 'ads_count']

for c in cat_cols:
    le = LabelEncoder()
    allvalues = np.unique(full[c].values).tolist()
    le.fit(allvalues)
    full[c] = le.transform(full[c].values)

###############################################################################

data_tr = full[:ntr]
data_te = full[ntr:]

del full
gc.collect()


###############################################################################


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
                             'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1',
                             'ads_count', 'weekday'])

###############################################################################

russian_stop = set(stopwords.words('russian'))

titles_tfidf = TfidfVectorizer(
    stop_words=russian_stop,
    max_features=30000,#ngram_range=(1, 2),
    norm='l2',
    sublinear_tf=True,
    smooth_idf=False,
    dtype=np.float32,
)

tr_titles = titles_tfidf.fit_transform(data_tr.text)
te_titles = titles_tfidf.transform(data_te.text)

desc_tfidf = TfidfVectorizer(
    stop_words=russian_stop,ngram_range=(1, 2),
    max_features=15000,
    norm='l2',
    sublinear_tf=True,
    smooth_idf=False,
    dtype=np.float32,
)

tr_desc = desc_tfidf.fit_transform(data_tr.params)
te_desc = desc_tfidf.transform(data_te.params)

#params_cv = CountVectorizer(
#    stop_words=russian_stop,
#    max_features=5000,
#    dtype=np.float32,
#)

#tr_params = params_cv.fit_transform(data_tr.params)
#te_params = params_cv.transform(data_te.params)

###############################################################################


###############################################################################
from sklearn.metrics import mean_squared_error
from math import sqrt

kf = KFold(ntr, n_folds=NFOLDS, shuffle=True, random_state=SEED)

ridge_params = {'alpha': 2.0, 'fit_intercept': True, 'normalize': False, 'copy_X': True,
                'max_iter': None, 'tol': 0.001, 'solver': 'auto', 'random_state': SEED}

# Ridge oof method from Faron's kernel

ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
ridge_oof_train_desc, ridge_oof_test_desc = get_oof(ridge, tr_desc, y, te_desc)
ridge_oof_train_title, ridge_oof_test_title = get_oof(ridge, tr_titles, y, te_titles)
#ridge_oof_train_params, ridge_oof_test_params = get_oof(ridge, tr_params, y, te_params)

rms = sqrt(mean_squared_error(y, ridge_oof_train_desc))
print('Ridge OOF RMSE: {}'.format(rms))

print("Modeling Stage")

data_tr['ridge_oof_desc'] = ridge_oof_train_desc
data_te['ridge_oof_desc'] = ridge_oof_test_desc

data_tr['ridge_preds_title'] = ridge_oof_train_title
data_te['ridge_preds_title'] = ridge_oof_test_title

#data_tr['ridge_preds_params'] = ridge_oof_train_params
#data_te['ridge_preds_params'] = ridge_oof_test_params

del  ridge_oof_train_title, ridge_oof_test_title, ridge_oof_train_desc, ridge_oof_test_desc

gc.collect()

data_tr, data_va = train_test_split(data_tr, shuffle=True, test_size=0.1, random_state=42)

fStats.fit_transform(data_tr)
fStats.transform(data_va)
fStats.transform(data_te)

tr_titles = titles_tfidf.fit_transform(data_tr.text)
va_titles = titles_tfidf.transform(data_va.text)
te_titles = titles_tfidf.transform(data_te.text)

tr_desc = desc_tfidf.fit_transform(data_tr.params)
va_desc = desc_tfidf.transform(data_va.params)
te_desc = desc_tfidf.transform(data_te.params)

#tr_params = params_cv.fit_transform(data_tr.params)
#va_params = params_cv.transform(data_va.params)
#te_params = params_cv.transform(data_te.params)

###############################################################################

columns_to_drop = ['title', 'description', 'params', 'image',
                   'activation_date', 'deal_probability', 'title_norm', 'description_norm', 'text']

X_tr = hstack([csr_matrix(data_tr.drop(columns_to_drop, axis=1)), tr_titles, tr_desc])
y_tr = data_tr['deal_probability']
del tr_titles, tr_desc, data_tr

gc.collect()
X_va = hstack([csr_matrix(data_va.drop(columns_to_drop, axis=1)), va_titles, va_desc])
y_va = data_va['deal_probability']
del va_titles, va_desc, data_va
gc.collect()
X_te = hstack([csr_matrix(data_te.drop(columns_to_drop, axis=1)), te_titles, te_desc])

del te_titles, te_desc, data_te

gc.collect()

###############################################################################

# tr_data = lgb.Dataset(X_tr, label=y_tr)
# va_data = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=tr_data)

# Train the model
# parameters = {
#    'objective':        'regression',
#    'metric':           'rmse',
#    'num_leaves':       255,
#    'learning_rate':    0.04,
#    'feature_fraction': 0.4,
#    'bagging_fraction': 0.8,
#    'bagging_freq':     2
# }

# model = lgb.train(parameters, tr_data, valid_sets=va_data,
#                  num_boost_round=25000, early_stopping_rounds=100, verbose_eval=200)

###############################################################################

X_tr = lgb.Dataset(X_tr, label=y_tr)
X_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=X_tr)

parameters = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 300,
    'learning_rate': 0.02,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.8,
    'bagging_freq': 2
}

model = lgb.train(parameters, X_tr, valid_sets=X_va,
                  num_boost_round=7000, early_stopping_rounds=300, verbose_eval=20)

##############################################################

y_pred = model.predict(X_te)
y_pred[y_pred < 0.005] = 0
y_pred[y_pred > 0.95] = 1
sub = pd.read_csv(patch + 'sample_submission.csv', nrows=nrows)
sub['deal_probability'] = y_pred
# sub['deal_probability'].clip(0.0, 1.0, inplace=True)
sub.to_csv(patch + 'lgb+text.csv', index=False)
print('done + enjoy')