# Light GBM for Avito Demand Prediction Challenge
# Uses Bag-of-Words, meta-text features, and dense features.
# NO COMPUTER VISION COMPONENT.

# https://www.kaggle.com/c/avito-demand-prediction
# By Nick Brooks, April 2018

import time

notebookstart = time.time()

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import json

# print("Data:\n", os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold

# Gradient Boosting
import lightgbm as lgb

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

import utils

skip_fold = [0, 1, 2, 3, 4, 5]

config = json.load(open("config.json"))
lgb_root = "lbg_root"

X = utils.load_bcolz(lgb_root, "lgb_X")[0].tocsr()
y = utils.load_bcolz(lgb_root, "lgb_y")
tfvocab = utils.load_bcolz(lgb_root, "lgb_tfvocab")
testing = utils.load_bcolz(lgb_root, "lgb_testing")[0]
categorical = utils.load_bcolz(lgb_root, "lgb_categorical")
testdex = utils.load_bcolz(lgb_root, "lgb_testdex")

print("\nModeling Stage")

# Training and Validation Set
"""
Using Randomized train/valid split doesn't seem to generalize LB score, so I will try time cutoff
"""

# Train with k-fold
n_folds = config["n_fold"]
skf = KFold(n_folds)

for fold, (train_index, val_index) in enumerate(skf.split(X)):
    print(f"\n[+] Fold {fold}")
    if fold in skip_fold:
        print(f"[+] Fold {fold} is skipped")
        continue

    X_train = X[train_index]
    y_train = y[train_index]

    X_valid = X[val_index]
    y_valid = y[val_index]

    # Save val index and test index to file
    utils.save_features(np.asarray(train_index), lgb_root, f"train_index_fold_{fold}")
    utils.save_features(np.asarray(val_index), lgb_root, f"val_index_fold_{fold}")

    print("Light Gradient Boosting Regressor")
    lgbm_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 15,
        'num_leaves': 35,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        # 'bagging_freq': 5,
        'learning_rate': 0.019,
        'verbose': 0
    }

    # LGBM Dataset Formatting
    lgtrain = lgb.Dataset(X_train, y_train,
                          feature_name=tfvocab,
                          categorical_feature=categorical)
    lgvalid = lgb.Dataset(X_valid, y_valid,
                          feature_name=tfvocab,
                          categorical_feature=categorical)

    # Go Go Go
    modelstart = time.time()
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=15000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train', 'valid'],
        early_stopping_rounds=200,
        verbose_eval=200
    )

    print('Save model...')
    # save model to file
    lgb_clf.save_model(f'lgb_model_fold_{fold}.txt')

    # Feature Importance Plot
    f, ax = plt.subplots(figsize=[7, 10])
    lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
    plt.title("Light GBM Feature Importance")
    plt.savefig(f'feature_import_fold_{fold}.png')

    print("Model Evaluation Stage")
    lgb_oof_pred = lgb_clf.predict(X_valid)
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_oof_pred)))
    lgpred = lgb_clf.predict(testing)
    lgsub = pd.DataFrame(lgpred, columns=["deal_probability"], index=testdex)
    lgsub['deal_probability'].clip(0.0, 1.0, inplace=True)  # Between 0 and 1
    lgsub.to_csv(f"lgb_sub_fold_{fold}.csv", index=True, header=True)

    # Save out of fold
    lgb_oof = pd.DataFrame(lgb_oof_pred, columns=["deal_probability"], index=val_index)
    lgb_oof['deal_probability'].clip(0.0, 1.0, inplace=True)  # Between 0 and 1
    lgb_oof.to_csv(f"lgb_oof_fold_{fold}.csv", index=True, header=True)

    print("Model Runtime: %0.2f Minutes" % ((time.time() - modelstart) / 60))
    print("Notebook Runtime: %0.2f Minutes" % ((time.time() - notebookstart) / 60))