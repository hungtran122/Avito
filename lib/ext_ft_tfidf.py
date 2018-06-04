import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gc
from ft_statistic import FeaturesStatistics
from scipy.sparse import hstack, csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
import utils

# root path
root = "/home/deeplearning/Kaggle/avito/input/"

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
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# For degbug logic
nrows = None

# Load csv files
train_df = pd.read_csv(root+"train.csv",
                       parse_dates=["activation_date"],
                       index_col="item_id", nrows=nrows)
test_df = pd.read_csv(root+"test.csv",
                      parse_dates=["activation_date"],
                      index_col="item_id", nrows=nrows)
train_norm_df = pd.read_csv(root + "train_norm.csv",
                            index_col="item_id",
                            parse_dates=["activation_date"], nrows=nrows)
test_norm_df = pd.read_csv(root + "test_norm.csv",
                           index_col="item_id",
                           parse_dates=["activation_date"], nrows=nrows)
user_df = pd.read_csv(root+'aggregated_features.csv', nrows=nrows)

train_df["description_norm"] = train_norm_df["description_norm"]
test_df["description_norm"] = test_norm_df["description_norm"]

train_df["title_norm"] = train_norm_df["title_norm"]
test_df["title_norm"] = test_norm_df["title_norm"]

# Target
y = train_df.deal_probability.copy()

ntr = len(train_df)
nte = len(test_df)

# Merge two dataframes
n_train = len(train_df)
df = pd.concat([train_df, test_df])
df = pd.merge(left=df, right=user_df, how="left", on=["user_id"])

del train_df, test_df, train_norm_df, test_norm_df
gc.collect()

textfeats1 = ['description', "title", 'param_1', 'param_2', 'param_3', 'description_norm', "title_norm"]
for col in textfeats1:
    df[col] = df[col].astype(str)
    df[col] = df[col].astype(str).fillna(' ')
    df[col] = df[col].str.lower()

df['param_2'] = df['param_1'] + ' ' + df['param_2']
df['param_3'] = df['param_2'] + ' ' + df['param_3']
df['params'] = df['param_3'] + ' ' + df['title_norm']
df['text'] = df['description_norm'] + ' ' + df['title_norm']

# Split to test and train
X_train = df[:n_train]
X_test = df[n_train:]
del df
gc.collect()

# russian_stop = set(stopwords.words('russian'))

text_tfidf = TfidfVectorizer(
    # stop_words=russian_stop,
    max_features=20000,
    norm='l2',
    sublinear_tf=True,
    smooth_idf=False,
    dtype=np.float32,
)

tr_text = text_tfidf.fit_transform(X_train.text)
te_text = text_tfidf.transform(X_test.text)

params_tfidf = TfidfVectorizer(
    # stop_words=russian_stop,
    max_features=15000,
    norm='l2',
    sublinear_tf=True,
    smooth_idf=False,
    dtype=np.float32,
)

tr_params = params_tfidf.fit_transform(X_train.params)
te_params = params_tfidf.transform(X_test.params)

# Extract ridge params
kf = KFold(NFOLDS, shuffle=True, random_state=SEED)

ridge_params = {'alpha': 2.0, 'fit_intercept': True, 'normalize': False, 'copy_X': True,
                'max_iter': None, 'tol': 0.001, 'solver': 'auto', 'random_state': SEED}

# Ridge oof method from Faron's kernel
ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
ridge_oof_train_text, ridge_oof_test_text = get_oof(ridge, tr_text, y, te_text)
ridge_oof_train_params, ridge_oof_test_params = get_oof(ridge, tr_params, y, te_params)

# Save features
utils.save_bcolz(tr_text, "features", "X_train_tfidf_text")
utils.save_bcolz(te_text, "features", "X_test_tfidf_text")

utils.save_bcolz(tr_params, "features", "X_train_tfidf_params")
utils.save_bcolz(te_params, "features", "X_test_tfidf_params")

utils.save_bcolz(ridge_oof_train_text, "features", "X_train_ridge_text")
utils.save_bcolz(ridge_oof_test_text, "features", "X_test_ridge_text")

utils.save_bcolz(ridge_oof_train_params, "features", "X_train_ridge_params")
utils.save_bcolz(ridge_oof_test_params, "features", "X_test_ridge_params")

print("[+] Extract TFIDF and Ridge features done !")