import json
import pandas as pd
import bcolz
import numpy as np
import os
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
import utils
import gc
stopWords_rus = stopwords.words('russian')


tfidf_para = {
    "stop_words": stopWords_rus,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}

all_columns = [
    "item_id", "user_id", "region", "city",
    "parent_category_name", "category_name",
    "param_1", "param_2", "param_3", "title",
    "description", "price", "item_seq_number",
    "activation_date", "user_type", "image",
    "image_top_1", "deal_probability"
]

unused_columns = [
    "image", "user_id", "item_id"
]

cat_columns = [
    "region", # Importance feature, best_val: 0.507
    "city", # Importance feature, best_val: 0.510
    "parent_category_name", # This feature seems not be importance, best_val:0.0505
    "category_name", # This feature seems be importance, best_val:0.0507
    "param_1", "param_2", "param_3",
    "user_type", "image_top_1"
]

agg_columns = [
    'region', 'city', 'parent_category_name',
    'category_name', 'image_top_1', 'user_type',
    'item_seq_number','activation_date'
]

num_columns = ["item_seq_number"]

text_cols = ["description", "text_feat", "title"]

config = json.load(open("config.json"))


def load_csv(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["activation_date"])
    df = df.replace(np.nan, -1, regex=True)
    return df


def remove_unused_columns(df: pd.DataFrame, columns=unused_columns):
    for c in columns:
        df = df.drop(c, axis=1)
    return df


def date_to_dow(df: pd.DataFrame):
    df["weekday"] = df["activation_date"].dt.weekday
    df["week"] = df["activation_date"].dt.week
    df["mon"] = df["activation_date"].dt.month
    df["mday"] = df["activation_date"].dt.day
    df["year_day"] = df["activation_date"].dt.dayofyear

    num_columns.append("weekday")

    return df


def create_token(df:pd.DataFrame, columns=cat_columns):
    token = []
    for c in columns:
        t = {x: i + 1 for i, x in enumerate(df[c].unique())}
        token.append(t)
    return token


def tokenize_data(df: pd.DataFrame, token, columns=cat_columns):
    token_data = []
    for i, c in enumerate(columns):
        td = np.asarray([token[i].get(key, 0) for key in df[c]], dtype=int)
        token_data.append(td)
    return token_data


def log_prices(df: pd.DataFrame):
    df["price"] = df["price"].apply(np.log1p)
    df["price"] = df["price"].apply(lambda x: -1 if x == -np.inf else x)
    num_columns.append("price")
    return df


def write_to_bcolz(data, name, root="bcolz_data"):
    if not os.path.exists(root):
        os.mkdir(root)

    fname = f"{root}/{name}"
    # n_dim = len(data)
    # n = len(data[0])
    bcolz_data = bcolz.carray(data, chunklen=1, mode="w", rootdir=fname)
    return bcolz_data


def agg_features(train_df, test_df, columns=agg_columns):
    for c in columns:
        gp = train_df.groupby(c)['price']
        mean = gp.mean()
        train_df[c + '_price_avg'] = train_df[c].map(mean)
        test_df[c + '_price_avg'] = test_df[c].map(mean)

        num_columns.append(c + '_price_avg')

    return train_df, test_df


def title_features(df, n_comp=3):
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
    tfidf_vec.fit(df['title'].values.tolist())
    tfidf = tfidf_vec.transform(df['title'].values.tolist())


    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(tfidf)
    svd = svd_obj.transform(tfidf)
    svd_df = pd.DataFrame(svd)
    svd_df.columns = ['svd_title_' + str(i + 1) for i in range(n_comp)]
    print("Title SVD_DF")
    print(svd_df.head(5))

    for i in range(n_comp):
        num_columns.append('svd_title_' + str(i + 1))

    df = pd.concat([df, svd_df], axis=1)
    return df, tfidf


def description_features(df, n_comp=3):
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
    tfidf_vec.fit(df['description'].values.tolist())
    tfidf = tfidf_vec.transform(df['description'].values.tolist())

    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(tfidf)
    svd = svd_obj.transform(tfidf)
    svd_df = pd.DataFrame(svd)
    svd_df.columns = ['svd_desc_' + str(i + 1) for i in range(n_comp)]
    print("Description SVD_DF")
    print(svd_df.head(5))
    
    for i in range(n_comp):
        num_columns.append('svd_desc_' + str(i + 1))

    df = pd.concat([df, svd_df], axis=1)
    return df, tfidf


def extract_params_tex_features(df):
    count_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=6000)
    count_vec.fit(df["text_feat"].values.tolist())
    tfidf = count_vec.transform(df["text_feat"].values.tolist())

    return tfidf


def extract_text_features_as_numeric(df, columns=text_cols):
    for cols in columns:
        df[cols] = df[cols].astype(str)
        df[cols] = df[cols].fillna('NA')  # FILL NA
        df[cols] = df[cols].str.lower()  # Lowercase all text, so that capitalized words dont get treated differently
        df[cols + '_num_chars'] = df[cols].apply(len)  # Count number of Characters
        df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split()))  # Count number of Words
        df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        df[cols + '_words_vs_unique'] = df[cols + '_num_unique_words'] / df[cols + '_num_words'] * 100  # Count Unique Words

        num_columns.append(cols + "_num_words")

    return df


def main():
    # Load json config
    config = json.load(open("config.json"))

    with utils.timer("Load csv"):
        print("[+] Load csv ...")
        train_df = load_csv(config["train_csv"])
        test_df = load_csv(config["test_csv"])

    with utils.timer("Create token"):
        print("[+] Create token ...")
        token = create_token(train_df)

    with utils.timer("Tokenize data"):
        print("[+] Tokenize data ...")
        train_token_data = tokenize_data(train_df, token)
        test_token_data = tokenize_data(test_df, token)
        
    y_train = train_df["deal_probability"].as_matrix()
    train_df = train_df.drop("deal_probability", axis=1)

    df = pd.concat([train_df, test_df], ignore_index=True)
    n_train = len(train_df)
    del train_df
    del test_df
    gc.collect()

    with utils.timer("Extract time features"):
        print("[+] Convert date to day of week ...")
        df = date_to_dow(df)

    with utils.timer("Extract text features as numeric"):
        print("[+] Extract text features as numeric ...")
        df['text_feat'] = df.apply(lambda row: ' '.join([
            str(row['param_1']),
            str(row['param_2']),
            str(row['param_3'])]), axis=1)  # Group Param Features
        df = extract_text_features_as_numeric(df)

    with utils.timer("Extract params text features"):
        print("[+] Extract params text features ...")
        param_tfidf = extract_params_tex_features(df)

    with utils.timer("Extract title features"):
        print("[+] Extract title features ...")
        df, title_tfidf = title_features(df)

    with utils.timer("Extract description features"):
        print("[+] Extract description features ...")
        df, description_tfidf = description_features(df)

    with utils.timer("Extract price features"):
        print("[+] Extract price features ...")
        df = log_prices(df)

    X_num = []
    print("[+] Extract numerical features ...")
    for c in num_columns:
        X_num.append(df[c].as_matrix())

    # Numeric data
    X_num = np.array(X_num, dtype=np.float32).T
    X_train_num = X_num[:n_train]
    X_test_num = X_num[n_train:]
    print(f"[+] Numeric {X_train_num.shape}/{X_test_num.shape}")

    del X_num
    gc.collect()

    # Categorical data
    X_train_cat = np.array(train_token_data, dtype=np.int).T
    X_test_cat = np.array(test_token_data, dtype=np.int).T
    print(f"[+] Cat {X_train_cat.shape}/{X_test_cat.shape}")

    X_train_desc = description_tfidf[:n_train]
    X_test_desc = description_tfidf[n_train:]
    print(f"[+] Description {X_train_desc.shape}/{X_test_desc.shape}")

    X_train_title = title_tfidf[:n_train]
    X_test_title = title_tfidf[n_train:]
    print(f"[+] Title {X_train_title.shape}/{X_test_title.shape}")

    X_train_param = param_tfidf[:n_train]
    X_test_param = param_tfidf[n_train:]
    print(f"[+] Param {X_train_param.shape}/{X_test_param.shape}")

    print("[+] Save features ...")

    y_train = np.asarray(y_train)
    # Save token len
    token_len = [len(t) for t in token]
    
    extracted_features_root = config["extracted_features"]
    utils.save_features(X_train_num, root=extracted_features_root,
                       name="X_train_num")

    utils.save_features(X_test_num, root=extracted_features_root,
                       name="X_test_num")

    utils.save_features(X_train_cat, root=extracted_features_root,
                       name="X_train_cat")

    utils.save_features(X_test_cat, root=extracted_features_root,
                       name="X_test_cat")

    utils.save_features(X_train_desc, root=extracted_features_root,
                       name="X_train_desc")

    utils.save_features(X_test_desc, root=extracted_features_root,
                       name="X_test_desc")

    utils.save_features(X_train_title, root=extracted_features_root,
                       name="X_train_title")

    utils.save_features(X_test_title, root=extracted_features_root,
                       name="X_test_title")

    utils.save_features(X_train_param, root=extracted_features_root,
                       name="X_train_param")

    utils.save_features(X_test_param, root=extracted_features_root,
                       name="X_test_param")

    utils.save_features(y_train, root=extracted_features_root,
                       name="y_train")

    utils.save_features(np.asarray(token_len), 
                        root=extracted_features_root,
                        name="token_len")


if __name__ == '__main__':
    main()