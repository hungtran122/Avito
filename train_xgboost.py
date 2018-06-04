import warnings

warnings.filterwarnings('ignore')

import numpy as  np
import pandas as pd
import os
import json
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix, vstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
import utils
import argparse

import xgboost as xgb

parser = argparse.ArgumentParser()
parser.add_argument('feature', choices=['load', 'new'])
args = parser.parse_args()

# Load config
config = json.load(open("config.json"))
xgb_root = "xgb_root"

nrows = None
sub = pd.read_csv(config["sample_submission"], nrows=nrows)
len_sub = len(sub)
print("Sample submission len {}".format(len_sub))

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

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


if args.feature == "new":
    # Load csv
    print("\n[+] Load csv ")
    train_df = pd.read_csv(config["train_csv"], parse_dates=["activation_date"], index_col="item_id", nrows=nrows)
    test_df = pd.read_csv(config["test_csv"], parse_dates=["activation_date"], index_col="item_id", nrows=nrows)

    train_norm_df = pd.read_csv(config["train_norm_csv"], index_col="item_id",
                      parse_dates=["activation_date"],nrows=nrows)
    test_norm_df = pd.read_csv(config["test_norm_csv"], index_col="item_id",
                      parse_dates=["activation_date"],nrows=nrows)

    user_df = pd.read_csv('./aggregated_features.csv', nrows=nrows)

    train_df["description_norm"] = train_norm_df["description_norm"]
    test_df["description_norm"] = test_norm_df["description_norm"]

    train_df["title_norm"] = train_norm_df["title_norm"]
    test_df["title_norm"] = test_norm_df["title_norm"]

    y = train_df.deal_probability.copy()

    ntr = len(train_df)
    nte = len(test_df)

    # Merge two dataframes
    n_train = len(train_df)
    df = pd.concat([train_df, test_df])
    df = pd.merge(left=df, right=user_df, how="left", on=["user_id"])

    del train_df, test_df, train_norm_df, test_norm_df
    gc.collect()

    cols_to_fill = ['description', 'param_1', 'param_2', 'param_3', 'description_norm']
    df[cols_to_fill] = df[cols_to_fill].fillna(' ')

    ## Tramnsform log
    eps = 0.001

    df['city'] = df['city'] + '_' + df['region']
    df["price"] = np.log(df["price"] + eps)
    df["price"].fillna(df["price"].mean(), inplace=True)
    df["image_top_1"].fillna(df["image_top_1"].mean(), inplace=True)
    df['avg_days_up_user'].fillna(-1, inplace=True)
    df['avg_days_up_user'].fillna(-1, inplace=True)
    df['avg_times_up_user'].fillna(-1, inplace=True)
    df['n_user_items'].fillna(-1, inplace=True)

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

    textfeats1 = ['description', "title", 'param_1', 'param_2', 'param_3', 'description_norm', "title_norm"]
    for col in textfeats1:
        df[col] = df[col].astype(str)
        df[col] = df[col].astype(str).fillna(' ')
        df[col] = df[col].str.lower()

    textfeats = ['description', "title"]
    for col in textfeats:
        df[col + '_num_words'] = df[col].apply(lambda s: len(s.split()))
        df[col + '_num_unique_words'] = df[col].apply(lambda s: len(set(w for w in s.split())))
        df[col + '_words_vs_unique'] = df[col + '_num_unique_words'] / df[col + '_num_words'] * 100
        df[col + '_num_lowE'] = df[col].str.count("[a-z]")
        df[col + '_num_lowR'] = df[col].str.count("[а-я]")
        df[col + '_num_pun'] = df[col].str.count("[[:punct:]]")
        df[col + '_num_dig'] = df[col].str.count("[[:digit:]]")

    df['param_2'] = df['param_1'] + ' ' + df['param_2']
    df['param_3'] = df['param_2'] + ' ' + df['param_3']
    # df['params'] = df['param_3']

    ###############################################################################
    df['params'] = df['param_3'] + ' ' + df['title_norm']
    df['text'] = df['description_norm'] + ' ' + df['title_norm']
    ###############################################################################

    names = ["city", "param_1", "user_id"]
    for i in names:
        df.loc[df[i].value_counts()[df[i]].values < 100, i] = "Rare_value"
    df.loc[df["image_top_1"].value_counts()[df["image_top_1"]].values < 200, "image_top_1"] = -1
    df.loc[df["item_seq_number"].value_counts()[df["item_seq_number"]].values < 150, "item_seq_number"] = -1

    cat_cols = ['user_id', 'region', 'city', 'category_name', "parent_category_name",
            'param_1', 'param_2', 'param_3', 'user_type',
            'weekday', 'ads_count']

    # Encoder:
    for c in cat_cols:
        le = LabelEncoder()
        allvalues = np.unique(df[c].values).tolist()
        le.fit(allvalues)
        df[c] = le.transform(df[c].values)

    # print(df.head(5).T)
    X_train = df[:n_train]
    X_test = df[n_train:]

    del df
    gc.collect()


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

    tr_titles = titles_tfidf.fit_transform(X_train.text)
    te_titles = titles_tfidf.transform(X_test.text)

    desc_tfidf = TfidfVectorizer(
        stop_words=russian_stop,
        max_features=15000,
        norm='l2',
        sublinear_tf=True,
        smooth_idf=False,
        dtype=np.float32,
    )

    tr_desc = desc_tfidf.fit_transform(X_train.params)
    te_desc = desc_tfidf.transform(X_test.params)

    # params_cv = CountVectorizer(
    #     stop_words=russian_stop,
    #     max_features=5000,
    #     dtype=np.float32,
    # )
    #
    # tr_params = params_cv.fit_transform(X_train.params)
    # te_params = params_cv.transform(X_test.params)

    # from sklearn.metrics import mean_squared_error
    # from math import sqrt
    #
    # kf = KFold(NFOLDS, shuffle=True, random_state=SEED)
    #
    # ridge_params = {'alpha': 2.0, 'fit_intercept': True, 'normalize': False, 'copy_X': True,
    #                 'max_iter': None, 'tol': 0.001, 'solver': 'auto', 'random_state': SEED}
    #
    # # Ridge oof method from Faron's kernel
    #
    # ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
    # ridge_oof_train_desc, ridge_oof_test_desc = get_oof(ridge, tr_desc, y, te_desc)
    # ridge_oof_train_title, ridge_oof_test_title = get_oof(ridge, tr_titles, y, te_titles)
    # # ridge_oof_train_params, ridge_oof_test_params = get_oof(ridge, tr_params, y, te_params)
    #
    # rms = sqrt(mean_squared_error(y, ridge_oof_train_desc))
    # print('Ridge OOF RMSE: {}'.format(rms))
    #
    # print("Modeling Stage")
    #
    # X_train['ridge_oof_desc'] = ridge_oof_train_desc
    # X_test['ridge_oof_desc'] = ridge_oof_test_desc
    #
    # X_train['ridge_preds_title'] = ridge_oof_train_title
    # X_test['ridge_preds_title'] = ridge_oof_test_title
    #
    # # X_train['ridge_preds_params'] = ridge_oof_train_params
    # # X_test['ridge_preds_params'] = ridge_oof_test_params
    #
    # del ridge_oof_train_title, ridge_oof_test_title, ridge_oof_train_desc, ridge_oof_test_desc

    gc.collect()

    X_train, X_val = train_test_split(X_train, shuffle=True, test_size=0.1, random_state=42)

    fStats.fit_transform(X_train)
    fStats.transform(X_val)
    fStats.transform(X_test)

    tr_titles = titles_tfidf.fit_transform(X_train.text)
    va_titles = titles_tfidf.transform(X_val.text)
    te_titles = titles_tfidf.transform(X_test.text)

    tr_desc = desc_tfidf.fit_transform(X_train.params)
    va_desc = desc_tfidf.transform(X_val.params)
    te_desc = desc_tfidf.transform(X_test.params)

    # tr_params = params_cv.fit_transform(X_train.params)
    # va_params = params_cv.transform(X_val.params)
    # te_params = params_cv.transform(X_test.params)

    columns_to_drop = ['title', 'description', 'params', 'image',
                   'activation_date', 'deal_probability', 'title_norm', 'description_norm', 'text']

    X_tr = hstack([csr_matrix(X_train.drop(columns_to_drop, axis=1)), tr_titles, tr_desc])
    y_tr = X_train['deal_probability']
    del tr_titles, tr_desc, X_train

    gc.collect()
    X_va = hstack([csr_matrix(X_val.drop(columns_to_drop, axis=1)), va_titles, va_desc])
    y_va = X_val['deal_probability']
    del va_titles, va_desc, X_val
    gc.collect()
    X_te = hstack([csr_matrix(X_test.drop(columns_to_drop, axis=1)), te_titles, te_desc])

    del te_titles, te_desc, X_test

    gc.collect()


    if nrows is None:
        utils.save_features(X_tr, xgb_root, "X_train")
        utils.save_features(X_va, xgb_root, "X_val")
        utils.save_features(X_te, xgb_root, "test")
        utils.save_features(y_tr, xgb_root, "y_train")
        utils.save_features(y_va, xgb_root, "y_val")

elif args.feature == "load":
    print("[+] Load features ")
    X_tr = utils.load_features(xgb_root, "X_train").any()
    X_va = utils.load_features(xgb_root, "X_val").any()
    X_te = utils.load_features(xgb_root, "test").any()
    y_tr = utils.load_features(xgb_root, "y_train")
    y_va = utils.load_features(xgb_root, "y_val")
    print("[+] Done ")
    X = vstack([X_tr, X_va])
    y = np.concatenate((y_tr, y_va))

    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, test_size=0.1, random_state=42)

    print(y_train)
    print(y_val)

print("Test size {}".format(X_te.shape[0]))
# assert len_sub != test.shape[0]

# Leave most parameters as default
params = {
    'objective': 'reg:logistic',
    'booster': "gbtree",
    'eval_metric': "rmse",
    # 'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'max_depth': 21,
    'eta': 0.05,
    'min_child_weight': 11,
    'gamma': 0,
    'subsample': 0.85,
    'colsample_bytree': 0.7,
    'silent': True,
    'alpha': 2.0,
    'lambda': 0,
    'nthread': 8,
    # 'max_bin': 16,
    # 'updater': 'grow_gpu',
    # 'tree_method':'exact'
}

xg_train = xgb.DMatrix(X_tr, label=y_tr)
xg_val = xgb.DMatrix(X_va, label=y_va)
xg_test = xgb.DMatrix(X_te)
gpu_res = {}  # Store accuracy result
watchlist = [(xg_train, 'train'), (xg_val, 'val')]
num_round = 5000
bst = xgb.train(params, xg_train, num_round, evals=watchlist, early_stopping_rounds=200, evals_result=gpu_res,
                verbose_eval=50)
pred = bst.predict(xg_test)
# print(len(pred))
# sub = pd.read_csv(config["sample_submission"],  nrows=nrows)
sub['deal_probability'] = pred
sub['deal_probability'].clip(0.0, 1.0, inplace=True)
sub.to_csv('submission_xgboost.csv', index=False)