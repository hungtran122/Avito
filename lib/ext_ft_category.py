import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gc
import utils
from sklearn.preprocessing import LabelEncoder

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
user_df = pd.read_csv(root+'aggregated_features.csv', nrows=nrows)

# Target
y = train_df.deal_probability.copy()

ntr = len(train_df)
nte = len(test_df)

# Merge two dataframes
n_train = len(train_df)
df = pd.concat([train_df, test_df])

del train_df, test_df
gc.collect()

df["ads_count"] = df.groupby("user_id", as_index=False)["user_id"].transform(lambda s: s.count())
df['weekday'] = df['activation_date'].dt.weekday
# Category columns
cat_cols = ['user_id', 'region', 'city', 'category_name', "parent_category_name",
            'param_1', 'param_2', 'param_3', 'user_type',
            'weekday', 'ads_count']

fill_na = ['param_1', 'param_2', 'param_3']
for col in fill_na:
    df[col] = df[col].astype(str)
    df[col] = df[col].astype(str).fillna(' ')
    df[col] = df[col].str.lower()

names = ["city", "param_1", "user_id"]
for i in names:
    df.loc[df[i].value_counts()[df[i]].values < 100, i] = "Rare_value"

# Encoder:
for c in cat_cols:
    le = LabelEncoder()
    allvalues = np.unique(df[c].values).tolist()
    le.fit(allvalues)
    df[c] = le.transform(df[c].values)

# Split to test and train
X_train = df[:n_train]
X_test = df[n_train:]
del df
gc.collect()

# Category columns
with open('category_columns.txt', 'w') as f:
    for item in cat_cols:
        f.write("%s\n" % item)

# To numpy array
X_train = X_train[cat_cols].values
X_test = X_test[cat_cols].values

# Save matrix
utils.save_bcolz(X_train, "features", "X_train_cat")
utils.save_bcolz(X_test, "features", "X_test_cat")

print("[+] Extract categorical features done !")