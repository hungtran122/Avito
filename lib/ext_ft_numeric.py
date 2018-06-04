import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gc
from ft_statistic import FeaturesStatistics
from scipy.sparse import csr_matrix
import utils

# root path
root = "/home/deeplearning/Kaggle/avito/input/"

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

# Tramnsform log
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
df["item_seq_bin"] = df["item_seq_number"] // 100
df["ads_count"] = df.groupby("user_id", as_index=False)["user_id"].transform(lambda s: s.count())

textfeats1 = ['description', "title", 'param_1', 'param_2', 'param_3', 'description_norm', "title_norm"]
for col in textfeats1:
    df[col] = df[col].astype(str)
    df[col] = df[col].astype(str).fillna(' ')
    df[col] = df[col].str.lower()

df.loc[df["image_top_1"].value_counts()[df["image_top_1"]].values < 200, "image_top_1"] = -1
df.loc[df["item_seq_number"].value_counts()[df["item_seq_number"]].values < 150, "item_seq_number"] = -1

textfeats = ['description', "title"]
for col in textfeats:
    df[col + '_num_words'] = df[col].apply(lambda s: len(s.split()))
    df[col + '_num_unique_words'] = df[col].apply(lambda s: len(set(w for w in s.split())))
    df[col + '_words_vs_unique'] = df[col + '_num_unique_words'] / df[col + '_num_words'] * 100
    df[col + '_num_lowE'] = df[col].str.count("[a-z]")
    df[col + '_num_lowR'] = df[col].str.count("[а-я]")
    df[col + '_num_pun'] = df[col].str.count("[[:punct:]]")
    df[col + '_num_dig'] = df[col].str.count("[[:digit:]]")

fStats = FeaturesStatistics(['region', 'city', 'parent_category_name', 'category_name',
                             'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1',
                             'ads_count', 'weekday'])

# Split to test and train
X_train = df[:n_train]
X_test = df[n_train:]
del df
gc.collect()

# Do the feature statistic
fStats.fit_transform(X_train)
fStats.transform(X_test)

# Category columns
cat_cols = ['user_id', 'region', 'city', 'category_name', "parent_category_name",
            'param_1', 'param_2', 'param_3', 'user_type',
            'weekday', 'ads_count']

columns_to_drop = ['title', 'description', 'image',
                   'activation_date', 'deal_probability', 'title_norm', 'description_norm']

# Drop columns
X_train = X_train.drop(columns_to_drop + cat_cols, axis=1)
X_test = X_test.drop(columns_to_drop + cat_cols, axis=1)

# Numeric columns
numeric_columns = list(X_train.columns.values)
with open('numeric_columns.txt', 'w') as f:
    for item in numeric_columns:
        f.write("%s\n" % item)

# To csr matrix
X_train = csr_matrix(X_train)
X_test = csr_matrix(X_test)

# Save matrix
utils.save_bcolz(X_train, "features", "X_train_num")
utils.save_bcolz(X_test, "features", "X_test_num")
utils.save_bcolz(y, "features", "X_train_y")

print("[+] Extract numeric features done !")
