#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 19:07:14 2018

@author: Hung Tran
"""

import gc
import lightgbm as lgb
# import xgboost as xgb
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, vstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os

# from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
import utils
import argparse
import json
import time
first_start = time.time()

import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FM_FTRL
from sklearn.feature_selection.univariate_selection import SelectKBest, f_regression
from sklearn.decomposition import TruncatedSVD



# import string

##############################################################################
not_important = ['param_2_num_words', 'param_3_num_chars', 'weekend', 'param_1_num_words', 'weekday', 'title_4', 'title_1']
NFOLDS = 5
SEED = 2018
nrows = None
useTFIDF = True
useOOF = True
useKfold = False
addWordBatch = True
addVGG16 = True
dropNotImportant = True
if useTFIDF is True:
    useOOF = True
saveModel = False
createDataFrameFeature = True
cat_cols = ['user_id', 'region', 'city', 'category_name', "parent_category_name",
                'param_1', 'param_2', 'param_3', 'user_type',
                'weekday', 'ads_count']
columns_to_drop = ['title', 'description', 'params', 'image',
                       'activation_date', 'deal_probability', 'title_norm', 'description_norm', 'text']

##############################################################################
import utils

parser = argparse.ArgumentParser()
parser.add_argument('feature', choices=['load', 'new'])
args = parser.parse_args()

# Load config
config = json.load(open("config.json"))
root_dir = './'
input_dir = '/home/rsa-key-20180618/kaggle/avito/input/'
index_dir = '/home/deeplearning/Kaggle/avito/index'
if not os.path.exists(os.path.join(root_dir + 'lgbm_root')):
    os.mkdir(os.path.join(root_dir + 'lgbm_root'))
lgbm_dir = "lgbm_root"

sub = pd.read_csv(config["sample_submission"], nrows=nrows)
len_sub = len(sub)
print("Sample submission len {}".format(len_sub))

##############################################################################
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

if args.feature == "new":
    data_tr1 = pd.read_csv(input_dir + 'train_stem.csv', index_col="item_id",
                           parse_dates=["activation_date"], nrows=nrows)
    data_te1 = pd.read_csv(input_dir + 'test_stem.csv', index_col="item_id", parse_dates=["activation_date"], nrows=nrows)
    user_df = pd.read_csv(input_dir + 'aggregated_features.csv', nrows=nrows)

    data_tr = pd.read_csv(input_dir + 'train.csv', index_col="item_id", parse_dates=["activation_date"], nrows=nrows)

    data_te = pd.read_csv(input_dir + 'test.csv', index_col="item_id", parse_dates=["activation_date"], nrows=nrows)

    data_tr["description_norm"] = data_tr1["description_stem"]
    data_te["description_norm"] = data_te1["description_stem"]

    data_tr["title_norm"] = data_tr1["title_stem"]
    data_te["title_norm"] = data_te1["title_stem"]

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
    #full['day_of_month'] = full['activation_date'].dt.day
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
    print('Creating difference in num_words of description and title')
    #full['diff_num_word_des_title'] = full['description_num_words'] - full['title_num_words']
    full['param_2'] = full['param_1'] + ' ' + full['param_2']
    # full['param_1'] = full['param_3']
    full['param_3'] = full['param_2'] + ' ' + full['param_3']


    ###############################################################################
    full['params'] = full['param_3']+' '+full['title_norm']
    full['text']=full['description_norm']+ ' ' + full['title_norm']
    ###############################################################################

    names = ["city", "param_1", "user_id"]
    for i in names:
        full.loc[full[i].value_counts()[full[i]].values < 100, i] = "Rare_value"

    full.loc[full["image_top_1"].value_counts()[full["image_top_1"]].values < 150, "image_top_1"] = -1
    full.loc[full["item_seq_number"].value_counts()[full["item_seq_number"]].values < 150, "item_seq_number"] = -1
    ###############################################################################

    for c in cat_cols:
        le = LabelEncoder()
        allvalues = np.unique(full[c].values).tolist()
        le.fit(allvalues)
        full[c] = le.transform(full[c].values)

    if addVGG16:
        vgg_train = utils.load_imfeatures(os.path.join(input_dir, 'vgg16-train-features'))
        vgg_test = utils.load_imfeatures(os.path.join(input_dir, 'vgg16-test-features'))
        if nrows is None:
            vgg_full = vstack([vgg_train, vgg_test])
        else:
            vgg_full = vstack([vgg_train[:nrows], vgg_test[:nrows]])
        del vgg_train, vgg_test
        gc.collect()
        ### Categorical image feature (max and min VGG16 feature) ###
        full['im_max_feature'] = vgg_full.argmax(axis=1)  # This will be categorical
        full['im_min_feature'] = vgg_full.argmin(axis=1)  # This will be categorical

        full['im_n_features'] = vgg_full.getnnz(axis=1)
        full['im_mean_features'] = vgg_full.mean(axis=1)
        full['im_meansquare_features'] = vgg_full.power(2).mean(axis=1)

        ### Let`s reduce 512 VGG16 featues into 32 ###
        tsvd = TruncatedSVD(32)
        ftsvd = tsvd.fit_transform(vgg_full)
        del vgg_full
        gc.collect()

        ### Merge image features into full ###
        df_ftsvd = pd.DataFrame(ftsvd, index=full.index).add_prefix('im_tsvd_')

        full = pd.concat([full, df_ftsvd], axis=1)

        del df_ftsvd, ftsvd
        gc.collect()


    ###############################################################################

    data_tr = full[:ntr]
    data_te = full[ntr:]

    del full
    gc.collect()
    ########################ADDITIONAL FEATURES ###################################
    if addWordBatch:
        start_time = time.time()
        y_train = y
        with utils.timer('Wordbatch title ...'):
            wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2,
                                                                          "hash_ngrams_weights": [1.5, 1.0],
                                                                          "hash_size": 2 ** 29,
                                                                          "norm": None,
                                                                          "tf": 'binary',
                                                                          "idf": None,
                                                                          }), procs=8)
            wb.dictionary_freeze = True
            X_name_train = wb.fit_transform(data_tr['title_norm'])
            print(X_name_train.shape)
            X_name_test = wb.transform(data_te['title_norm'])
            print(X_name_test.shape)
            del (wb)
            gc.collect()
        mask = np.where(X_name_train.getnnz(axis=0) > 3)[0]
        X_name_train = X_name_train[:, mask]
        print(X_name_train.shape)
        X_name_test = X_name_test[:, mask]
        print(X_name_test.shape)
        print('[{}] Vectorize `title` completed.'.format(time.time() - start_time))

        X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_name_train, y_train,
                                                                      test_size=0.5,
                                                                      shuffle=False)
        print('[{}] Finished splitting'.format(time.time() - start_time))

        model = Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=5)
        model.fit(X_train_1, y_train_1)
        print('[{}] Finished to train name ridge (1)'.format(time.time() - start_time))
        name_ridge_preds1 = model.predict(X_train_2)
        name_ridge_preds1f = model.predict(X_name_test)
        print('[{}] Finished to predict name ridge (1)'.format(time.time() - start_time))
        model = Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=5)
        model.fit(X_train_2, y_train_2)
        print('[{}] Finished to train name ridge (2)'.format(time.time() - start_time))
        name_ridge_preds2 = model.predict(X_train_1)
        name_ridge_preds2f = model.predict(X_name_test)
        print('[{}] Finished to predict name ridge (2)'.format(time.time() - start_time))
        name_ridge_preds_oof = np.concatenate((name_ridge_preds2, name_ridge_preds1), axis=0)
        name_ridge_preds_test = (name_ridge_preds1f + name_ridge_preds2f) / 2.0
        print('RMSLE OOF: {}'.format(rmse(name_ridge_preds_oof, y_train)))
        gc.collect()

        with utils.timer('Wordbatch description ...'):
            wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2,
                                                                          "hash_ngrams_weights": [1.0, 1.0],
                                                                          "hash_size": 2 ** 28,
                                                                          "norm": "l2",
                                                                          "tf": 1.0,
                                                                          "idf": None}), procs=8)
            wb.dictionary_freeze = True
            X_description_train = wb.fit_transform(data_tr['description_norm'].fillna(''))
            print(X_description_train.shape)
            X_description_test = wb.transform(data_te['description_norm'].fillna(''))
            print(X_description_test.shape)
            print('-')
            del (wb)
            gc.collect()

            mask = np.where(X_description_train.getnnz(axis=0) > 8)[0]
            X_description_train = X_description_train[:, mask]
            print(X_description_train.shape)
            X_description_test = X_description_test[:, mask]
            print(X_description_test.shape)
            print('[{}] Vectorize `description` completed.'.format(time.time() - start_time))

            X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_description_train, y_train,
                                                                          test_size=0.5,
                                                                          shuffle=False)
            print('[{}] Finished splitting'.format(time.time() - start_time))

            # Ridge adapted from https://www.kaggle.com/object/more-effective-ridge-script?scriptVersionId=1851819
            model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
            model.fit(X_train_1, y_train_1)
            print('[{}] Finished to train desc ridge (1)'.format(time.time() - start_time))
            desc_ridge_preds1 = model.predict(X_train_2)
            desc_ridge_preds1f = model.predict(X_description_test)
            print('[{}] Finished to predict desc ridge (1)'.format(time.time() - start_time))
            model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
            model.fit(X_train_2, y_train_2)
            print('[{}] Finished to train desc ridge (2)'.format(time.time() - start_time))
            desc_ridge_preds2 = model.predict(X_train_1)
            desc_ridge_preds2f = model.predict(X_description_test)
            print('[{}] Finished to predict desc ridge (2)'.format(time.time() - start_time))
            desc_ridge_preds_oof = np.concatenate((desc_ridge_preds2, desc_ridge_preds1), axis=0)
            desc_ridge_preds_test = (desc_ridge_preds1f + desc_ridge_preds2f) / 2.0
            print('RMSLE OOF: {}'.format(rmse(desc_ridge_preds_oof, y_train)))
            gc.collect()

            del X_train_1
            del X_train_2
            del y_train_1
            del y_train_2
            del name_ridge_preds1
            del name_ridge_preds1f
            del name_ridge_preds2
            del name_ridge_preds2f
            del desc_ridge_preds1
            del desc_ridge_preds1f
            del desc_ridge_preds2
            del desc_ridge_preds2f
            gc.collect()
            print('[{}] Finished garbage collection'.format(time.time() - start_time))

            print('Remerged')

            dummy_cols = ['parent_category_name', 'category_name', 'user_type',
                          'region', 'city', 'param_1', 'param_2', 'param_3']
            numeric_cols = list(set(data_tr.columns.values) - set(dummy_cols + columns_to_drop))
            print(numeric_cols)
            for c in data_tr.drop(columns_to_drop,axis=1).columns:
                #print('Columns {} has min = {}, max = {} and dtype {}'.format(c, data_tr[c].min(), data_tr[c].max(), data_tr[c].dtype))
                if data_tr[c].isnull().sum() > 0:
                    print("Found null in train data at ", c)
                    data_tr[c].fillna(0,inplace=True)
            for c in data_te.drop(columns_to_drop,axis=1).columns:
                #print('Columns {} has min = {}, max = {} and dtype {}'.format(c, data_tr[c].min(), data_tr[c].max(), data_tr[c].dtype))
                if data_te[c].isnull().sum() > 0:
                    print("Found null in test data at ", c)
                    data_te[c].fillna(0,inplace=True)

            from sklearn.preprocessing import StandardScaler
            from sklearn.base import BaseEstimator, TransformerMixin


            # https://stackoverflow.com/questions/37685412/avoid-scaling-binary-columns-in-sci-kit-learn-standsardscaler
            class Scaler(BaseEstimator, TransformerMixin):
                def __init__(self, columns, copy=True, with_mean=True, with_std=True):
                    self.scaler = StandardScaler(copy, with_mean, with_std)
                    self.columns = columns

                def fit(self, X, y=None):
                    self.scaler.fit(X[self.columns], y)
                    return self

                def transform(self, X, y=None, copy=None):
                    init_col_order = X.columns
                    X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns, index=X.index)
                    X_not_scaled = X[list(set(init_col_order) - set(self.columns))]
                    return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


            print('Scaler')
            scaler = Scaler(columns=numeric_cols)
            data_tr[numeric_cols] = scaler.fit_transform(data_tr[numeric_cols])
            data_te[numeric_cols] = scaler.transform(data_te[numeric_cols])

            print(data_tr.columns.tolist())

            from sklearn.preprocessing import LabelBinarizer

            sparse_merge_train = hstack((X_name_train, X_description_train)).tocsr()
            sparse_merge_test = hstack((X_name_test, X_description_test)).tocsr()
            print(sparse_merge_train.shape)
            for col in dummy_cols:
                print(col)
                lb = LabelBinarizer(sparse_output=True)
                sparse_merge_train = hstack((sparse_merge_train, lb.fit_transform(data_tr[[col]].fillna('')))).tocsr()
                print(sparse_merge_train.shape)
                sparse_merge_test = hstack((sparse_merge_test, lb.transform(data_te[[col]].fillna('')))).tocsr()

            del X_description_test, X_name_test
            del X_description_train, X_name_train
            del lb, mask
            gc.collect()

            print("\n FM_FTRL Starting...........")
            iters = 1

            model = FM_FTRL(alpha=0.035, beta=0.001, L1=0.00001, L2=0.15, D=sparse_merge_train.shape[1],
                            alpha_fm=0.05, L2_fm=0.0, init_fm=0.01,
                            D_fm=100, e_noise=0, iters=iters, inv_link="identity", threads=4)

            model.fit(sparse_merge_train, y_train)
            print('[{}] Train FM completed'.format(time.time() - start_time))
            predsFM = model.predict(sparse_merge_test)
            print('[{}] Predict FM completed'.format(time.time() - start_time))

            del model
            gc.collect()
            # 0.23046 in 1/3

            X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(sparse_merge_train, y_train,
                                                                          test_size=0.5,
                                                                          shuffle=False)
            print('[{}] Finished splitting'.format(time.time() - start_time))

            model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
            model.fit(X_train_1, y_train_1)
            print('[{}] Finished to train ridge (1)'.format(time.time() - start_time))
            ridge_preds1 = model.predict(X_train_2)
            ridge_preds1f = model.predict(sparse_merge_test)
            print('[{}] Finished to predict ridge (1)'.format(time.time() - start_time))
            model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3.3)
            model.fit(X_train_2, y_train_2)
            print('[{}] Finished to train ridge (2)'.format(time.time() - start_time))
            ridge_preds2 = model.predict(X_train_1)
            ridge_preds2f = model.predict(sparse_merge_test)
            print('[{}] Finished to predict ridge (2)'.format(time.time() - start_time))
            ridge_preds_oof = np.concatenate((ridge_preds2, ridge_preds1), axis=0)
            ridge_preds_test = (ridge_preds1f + ridge_preds2f) / 2.0
            print('RMSLE OOF: {}'.format(rmse(ridge_preds_oof, y_train)))

            fselect = SelectKBest(f_regression, k=48)
            train_features = fselect.fit_transform(sparse_merge_train, y_train)
            test_features = fselect.transform(sparse_merge_test)
            print('[{}] Select best completed'.format(time.time() - start_time))

            del sparse_merge_train
            del sparse_merge_test
            gc.collect()
            print('[{}] Garbage collection'.format(time.time() - start_time))

            del ridge_preds1
            del ridge_preds1f
            del ridge_preds2
            del ridge_preds2f
            del X_train_1
            del X_train_2
            del y_train_1
            del y_train_2
            del model
            gc.collect()
            print('[{}] Finished garbage collection'.format(time.time() - start_time))

            data_tr['ridge'] = ridge_preds_oof
            data_tr['name_ridge'] = name_ridge_preds_oof
            data_tr['desc_ridge'] = desc_ridge_preds_oof
            data_te['ridge'] = ridge_preds_test
            data_te['name_ridge'] = name_ridge_preds_test
            data_te['desc_ridge'] = desc_ridge_preds_test
            print('[{}] Finished adding submodels'.format(time.time() - start_time))

            del ridge_preds_oof
            del ridge_preds_test
            gc.collect()
            print('[{}] Finished garbage collection'.format(time.time() - start_time))


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
        max_features=20000,
        norm='l2',
        sublinear_tf=True,
        smooth_idf=False,
        dtype=np.float32,
    )

    tr_titles = titles_tfidf.fit_transform(data_tr.params).astype('float32')
    te_titles = titles_tfidf.transform(data_te.params).astype('float32')
    title_tfidf_feature_name = titles_tfidf.get_feature_names()

    desc_tfidf = TfidfVectorizer(
        stop_words=russian_stop,
        max_features=20000,
        norm='l2',
        sublinear_tf=True,
        smooth_idf=False,
        dtype=np.float32,
    )

    tr_desc = desc_tfidf.fit_transform(data_tr.text)
    te_desc = desc_tfidf.transform(data_te.text)
    desc_tfidf_feature_name = desc_tfidf.get_feature_names()
    print('Finished tfidf features fitting ...')

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

    if nrows is None:
        y.to_csv(os.path.join(root_dir, lgbm_dir, '2204/train_labels.csv'))
        data_tr.to_csv(os.path.join(root_dir, lgbm_dir, '2204/df_tr_not_statistics.csv'))
        data_te.to_csv(os.path.join(root_dir, lgbm_dir, '2204/df_te_not_statistics.csv'))
        print('++++++++++ Finished saving train and test sets as csv files ...')

    #data_tr['ridge_preds_params'] = ridge_oof_train_params
    #data_te['ridge_preds_params'] = ridge_oof_test_params

    del ridge_oof_train_title, ridge_oof_test_title, ridge_oof_train_desc, ridge_oof_test_desc

    gc.collect()

    ##########################################################################################
    '''
    Split train and val data
    '''
    # tr_idx = np.load(os.path.join(index_dir,'train_index_2018.npy'))
    # val_idx = np.load(os.path.join(index_dir,'val_index_2018.npy'))
    # print('********************')
    # data_va = data_tr.iloc[val_idx, :]
    # data_tr = data_tr.iloc[tr_idx, :]
    # print(data_va.columns)
    # print(data_tr.columns)
    # data_tr, data_va = train_test_split(data_tr, shuffle=True, test_size=0.1, random_state=SEED)
    ##########################################################################################


    fStats.fit_transform(data_tr)
    # fStats.transform(data_va)
    fStats.transform(data_te)


    # if nrows is None:
    #     y.to_csv(os.path.join(root_dir, lgbm_dir, '2204/train_labels.csv'))
    #     data_tr.to_csv(os.path.join(root_dir, lgbm_dir, '2204/df_tr.csv'))
    #     data_va.to_csv(os.path.join(root_dir, lgbm_dir, '2204/df_va.csv'))
    #     data_te.to_csv(os.path.join(root_dir, lgbm_dir, '2204/df_te.csv'))

    if useTFIDF:
        print('Start tfidf transforming ...')
        feature_names = data_tr.drop(columns_to_drop,
                                     axis=1).columns.tolist() + title_tfidf_feature_name + desc_tfidf_feature_name
        tr_titles = titles_tfidf.fit_transform(data_tr.params).astype('float32')
        # va_titles = titles_tfidf.transform(data_va.params).astype('float32')
        te_titles = titles_tfidf.transform(data_te.params).astype('float32')

        tr_desc = desc_tfidf.fit_transform(data_tr.text).astype('float32')
        # va_desc = desc_tfidf.transform(data_va.text).astype('float32')
        te_desc = desc_tfidf.transform(data_te.text).astype('float32')

        if nrows is None:
            utils.save_sparse_matrix(os.path.join(root_dir, lgbm_dir, '2204/tr_titles'), tr_titles)
            # utils.save_sparse_matrix(os.path.join(root_dir, lgbm_dir, '2204/va_titles'), va_titles)
            utils.save_sparse_matrix(os.path.join(root_dir, lgbm_dir, '2204/te_titles'), te_titles)
            utils.save_sparse_matrix(os.path.join(root_dir, lgbm_dir, '2204/tr_desc'), tr_desc)
            # utils.save_sparse_matrix(os.path.join(root_dir, lgbm_dir, '2204/va_desc'), va_desc)
            utils.save_sparse_matrix(os.path.join(root_dir, lgbm_dir, '2204/te_desc'), te_desc)


        print('Finished tfidf transforming ...')
    ###############################################################################


    print("++++++++++++++++++++++++ Columns in data frame after drop unused: ", data_tr.drop(columns_to_drop, axis=1).columns.tolist())
    if useTFIDF:
        print('Start hstacking sparse matrix  ...')
        X_tr = hstack([csr_matrix(data_tr.drop(columns_to_drop, axis=1)), tr_titles, tr_desc])
        y_tr = data_tr['deal_probability']

        # X_va = hstack([csr_matrix(data_va.drop(columns_to_drop, axis=1)), va_titles, va_desc])
        # y_va = data_va['deal_probability']

        if useKfold:
            X_tr = np.concatenate((X_tr, X_va), axis=0)
            y_tr = np.concatenate((y_tr, y_va), axis=0)

        X_te = hstack([csr_matrix(data_te.drop(columns_to_drop, axis=1)), te_titles, te_desc])

        del tr_titles, tr_desc, data_tr
        # del va_titles, va_desc, data_va
        del te_titles, te_desc, data_te
        gc.collect()
    else:

        feature_names = data_tr.drop(columns_to_drop, axis=1).columns.tolist()
        X_tr = data_tr.drop(columns_to_drop, axis=1)
        y_tr = data_tr['deal_probability']
        # X_va = data_va.drop(columns_to_drop, axis=1)
        # y_va = data_va['deal_probability']

        # if useKfold:
        #     X_tr = np.concatenate((X_tr, X_va), axis=0)
        #     y_tr = np.concatenate((y_tr, y_va), axis=0)
        X_te = data_te.drop(columns_to_drop, axis=1)

        # del data_va
        gc.collect()
        del data_tr
        gc.collect()
        del data_te
        gc.collect()
        print('Finished hstacking sparse matrix with no tfidf ...')

        ################################################################################
        # if nrows is None:
        # utils.save_features(X_tr, lgbm_dir, "X_train")
        # utils.save_features(X_va, lgbm_dir, "X_val")
        # utils.save_features(X_te, lgbm_dir, "test")
        # utils.save_features(y_tr, lgbm_dir, "y_train")
        # utils.save_features(y_va, lgbm_dir, "y_val")
        ################################################################################
elif args.feature == "load":
    # print("[+] Load features ")
    # X_tr = utils.load_features(lgbm_dir, "X_train").any()
    # X_va = utils.load_features(lgbm_dir, "X_val").any()
    # X_te = utils.load_features(lgbm_dir, "test").any()
    # y_tr = utils.load_features(lgbm_dir, "y_train")
    # y_va = utils.load_features(lgbm_dir, "y_val")
    # print("[+] Done ")
    # X = vstack([X_tr, X_va])
    # y = np.concatenate((y_tr, y_va))
    #
    # X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, test_size=0.1, random_state=42)
    #
    # print(y_train)
    # print(y_val)
    print("Test size {}".format(X_te.shape[0]))


# In[ ]:
if useTFIDF:
    X_tr_lgb = lgb.Dataset(X_tr, label=y_tr)
    # X_va_lgb = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, categorical_feature=cat_cols, reference=X_tr_lgb)
else:
    X_tr_lgb = lgb.Dataset(X_tr, label=y_tr)
    # X_va_lgb = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, categorical_feature=cat_cols, reference=X_tr_lgb)

parameters = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'num_leaves': 255,
    'learning_rate': 0.02,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'num_threads': -1
}

print('++++++++++++++++++++++++++ Parameters ++++++++++++++++++++++++++ ')
print(parameters)
# In[ ]:


evals_result = {}  # to record eval results for plotting


# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
# Viz lib
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt

kf = KFold(n_splits=5, shuffle=True)
if useKfold:
    for idx, (train_index, test_index) in enumerate(kf.split(X_tr)):
        print('+++++ CV at fold number ', idx)
        X_train, X_test = X_tr[train_index], X_tr[test_index]
        y_train, y_test = y_tr[train_index], y_tr[test_index]
        X_tr_lgb = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, categorical_feature=cat_cols)
        X_va_lgb = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, categorical_feature=cat_cols, reference=X_tr_lgb)
        model = lgb.train(parameters, X_tr_lgb, valid_sets=[X_tr_lgb, X_va_lgb],
                          valid_names=['train', 'valid'], evals_result=evals_result,
                          num_boost_round=3999, early_stopping_rounds=100, verbose_eval=100)
        print("++++++ model.feature_importance ", model.feature_importance().reshape(-1,1).shape)
        if idx == 0:
            feature_important_list = model.feature_importance().reshape(1,-1)
        else:
            feature_important_list = np.concatenate((feature_important_list, model.feature_importance().reshape(1,-1)), axis=0)

        y_pred = model.predict(X_te)
        y_pred[y_pred < 0.005] = 0
        y_pred[y_pred > 0.95] = 1
        sub['deal_probability'] = y_pred
        # sub['deal_probability'].clip(0.0, 1.0, inplace=True)
        if not os.path.exists(os.path.join(root_dir + 'submission')):
            os.mkdir(os.path.join(root_dir + 'submission'))
        sub.to_csv(root_dir + '/submission/lgb_2204' + str(idx) + '.csv', index=False)
        print('Finished saving submision csv file')
        if nrows is None:
            ########################################################################
            # Viz

            # Feature Importance Plot
            f, ax = plt.subplots(figsize=[80, 80])
            lgb.plot_importance(model, max_num_features=200, ax=ax)
            plt.title("Light GBM Feature Importance")
            plt.savefig('./analysis/feature_import' + str(idx) + '.png')
    if nrows is None:
        print(feature_important_list.shape)
        pd.DataFrame(feature_important_list, columns=feature_names).to_csv('./analysis/feature_important.csv')
else:
    X_tr_lgb = lgb.Dataset(X_tr, label=y_tr)
    # X_va_lgb = lgb.Dataset(X_va, label=y_va, feature_name=feature_names, categorical_feature=cat_cols,
    #                        reference=X_tr_lgb)
    del X_tr#, X_va
    gc.collect()

    with utils.timer('Training LGBM model ...'):
        model = lgb.train(parameters, X_tr_lgb, valid_sets=[X_tr_lgb],
                          valid_names=['train'], evals_result=evals_result,
                          num_boost_round=3999, early_stopping_rounds=100, verbose_eval=100)
    if nrows is None:
        with utils.timer('Saving submission csv file'):
            y_pred = model.predict(X_te)
            y_pred[y_pred < 0.005] = 0
            y_pred[y_pred > 0.95] = 1
            sub['deal_probability'] = y_pred
            # sub['deal_probability'].clip(0.0, 1.0, inplace=True)
            if not os.path.exists(os.path.join(root_dir + 'submission')):
                os.mkdir(os.path.join(root_dir + 'submission'))
            sub.to_csv(root_dir + '/submission/lgb_2204.csv', index=False)
            print('Finished saving submission csv file')
        # with utils.timer('Saving pictorial data'):
        #     print('Plot metrics during training...')
        #     ax = lgb.plot_metric(evals_result, metric='rmse')
        #     plt.title("Light GBM RMSE Metric")
        #     plt.savefig('./analysis/RMSE_curve_no_kfolds.png')
        #     # Feature Importance Plot
        #     f, ax = plt.subplots(figsize=[80, 80])
        #     lgb.plot_importance(model, max_num_features=200, ax=ax)
        #     plt.title("Light GBM Feature Importance")
        #     plt.savefig('./analysis/feature_import_no_kfolds.png')
        #     feature_important_list = model.feature_importance().reshape(1, -1)
        #     print(feature_important_list.shape)
        #     pd.DataFrame(feature_important_list, columns=feature_names).to_csv('./analysis/feature_important_no_kfolds.csv')
print('Time to finish the whole running is {}'.format(time.time() - first_start))