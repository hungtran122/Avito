import json
import pandas as pd
import bcolz
import numpy as np
import os

import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold

import models
import datasets as d
import utils
import json
import gc

from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
stopWords_rus = stopwords.words('russian')


def load_csv(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["activation_date"])
    df = df.replace(np.nan, -1, regex=True)
    return df


def main():
    # Load json config
    config = json.load(open("config.json"))

    print("[+] Load csv ...")
    train_df = load_csv(config["train_csv"])
    test_df = load_csv(config["test_csv"])

    df = pd.concat([train_df, test_df])
    del train_df
    del test_df
    gc.collect()

    print("[+] Log price ...")
    df["price"] = df["price"].apply(np.log1p)
    df["price"] = df["price"].apply(lambda x: -1 if x == -np.inf else x)

    print("[+] Create time features ...")
    df["mon"] = df["activation_date"].dt.month
    df["mday"] = df["activation_date"].dt.day
    df["week"] = df["activation_date"].dt.week
    df["wday"] = df["activation_date"].dt.weekday

    cat_vars = [
        "category_name", "parent_category_name",
        "region", "user_type"
    ]

    print("[+] Label categories ...")
    for cat in cat_vars:
        df[cat] = LabelEncoder().fit_transform(df[cat].values)

    txt_vars = [
        "city", "param_1", "param_2",
        "param_3", "title", "description"
    ]

    print("[+] Merge text ...")
    for txt in txt_vars:
        df[txt] = df[txt].astype("str")

    df["txt"] = ""
    for txt in txt_vars:
        df["txt"] += df[txt]

    delete_columns = [
        "item_id", "user_id", "city",
        "param_1", "param_2", "param_3",
        "title", "description", "activation_date", "image"
    ]

    print("[+] Delete unused columns ...")
    for c in delete_columns:
        df = df.drop(c, axis=1)

    print("[+] Extract TFIDF  ...")
    df["txt"] = df["txt"].apply(lambda x: x.lower())
    df["txt"] = df["txt"].replace("[^[:alpha:]]", " ", regex=True)
    df["txt"] = df["txt"].replace("\\s+", " ", regex=True)
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, stop_words=stopWords_rus, max_features=5500)
    full_tfidf = tfidf_vec.fit_transform(df['txt'].values.tolist())
    # for i in range(5500):
    #     df['tfidf_' + str(i)] = full_tfidf[:, i]

    extract_columns = [
        'region', 'parent_category_name', 'category_name', 'price',
        'item_seq_number', 'user_type', 'image_top_1',
        'mon', 'mday', 'week', 'wday'
    ]

    print("[+] Stack more features  ...")
    for c in extract_columns:
        full_tfidf = hstack([full_tfidf, df[c].as_matrix()])
    full_tfidf = full_tfidf.tocsr()
    print("[+] Create y_train ...")
    y_train = train_df["deal_probability"].as_matrix()
    y_train = np.asarray(y_train)

    extracted_features_root = config["extracted_features"]
    utils.save_features(full_tfidf.tocsr(), root=extracted_features_root,
                       name="X_train_xgboost")
    utils.save_features(y_train, root=extracted_features_root,
                       name="y_train_xgboost")

if __name__ == '__main__':
    main()