import bcolz
import os
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm


def save_bcolz(feature, root, name):
    if not os.path.exists(root):
        os.mkdir(root)
    bcolz.carray(feature, chunklen=1024, mode='w', rootdir=f"{root}/{name}")

def load_bcolz(root, name):
    if not os.path.exists(root):
        print(f"[+] Feature {name} does not exists")
        return None
    return bcolz.open(f"{root}/{name}")

def save_csv(df, root, name):
    if not os.path.exists(root):
        os.mkdir(root)
    df.to_csv(f"{root}/{name}", index=False)

def load_features(train=True):
    # Load numeric features
    root = "features"

    if train:
        prefix = "train"
    else:
        prefix = "test"

    ato_prefix = [
        "num", "cat",
        "tfidf_text", "tfidf_params",
        "ridge_text", "ridge_params",
        "word"
    ]

    embedding_weights = load_bcolz(root, "embedding_weights")
    y = load_bcolz(root, "X_train_y")

    fname = [f"X_{prefix}_{x}" for x in ato_prefix]
    features = [load_bcolz(root, name) for name in fname]
    features.append(embedding_weights)
    features = [y] + features

    return features, fname

def unused_numeric(X_num, unsed_num):
    if unsed_num == []:
        return X_num

    with open("numeric_columns.txt", "r") as f:
        lines = f.readlines()
        numeric_columns = [line.rstrip('\n') for line in lines]

    index_list = []
    for i, c in enumerate(numeric_columns):
        if c not in unsed_num:
            index_list.append(i)

    X_num_new = X_num[:, index_list]
    return X_num_new

def unused_category(X_cat, unsed_num):
    if unsed_num == []:
        return X_cat

    with open("category_columns.txt", "r") as f:
        lines = f.readlines()
        cat_columns = [line.rstrip('\n') for line in lines]

    index_list = []
    for i, c in enumerate(cat_columns):
        if c not in unsed_num:
            index_list.append(i)

    X_cat_new = X_cat[:, index_list]
    return X_cat_new