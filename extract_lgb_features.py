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

config = json.load(open("config.json"))
lgb_root = "lbg_root"

print("\nData Load Stage")
training = pd.read_csv(config["train_csv"], index_col="item_id", parse_dates=["activation_date"])
traindex = training.index
testing = pd.read_csv(config["test_csv"], index_col="item_id", parse_dates=["activation_date"])
testdex = testing.index
y = training.deal_probability.copy()
training.drop("deal_probability", axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training, testing], axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print("Feature Engineering")
df["price"] = np.log(df["price"] + 0.001)
df["price"].fillna(-999, inplace=True)
df["image_top_1"].fillna(-999, inplace=True)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day

# Create Validation Index and Remove Dead Variables
training_index = df.loc[df.activation_date <= pd.to_datetime('2017-04-07')].index
validation_index = df.loc[df.activation_date >= pd.to_datetime('2017-04-08')].index
df.drop(["activation_date", "image"], axis=1, inplace=True)

print("\nEncode Variables")
categorical = ["user_id", "region", "city", "parent_category_name", "category_name", "user_type", "image_top_1"]
print("Encoding :", categorical)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))

print("\nText Features")

# Feature Engineering
df['text_feat'] = df.apply(lambda row: ' '.join([
    str(row['param_1']),
    str(row['param_2']),
    str(row['param_3'])]), axis=1)  # Group Param Features
df.drop(["param_1", "param_2", "param_3"], axis=1, inplace=True)

# Meta Text Features
textfeats = ["description", "text_feat", "title"]
for cols in textfeats:
    df[cols] = df[cols].astype(str)
    df[cols] = df[cols].astype(str).fillna('nicapotato')  # FILL NA
    df[cols] = df[cols].str.lower()  # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_chars'] = df[cols].apply(len)  # Count number of Characters
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split()))  # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols + '_num_unique_words'] / df[cols + '_num_words'] * 100  # Count Unique Words

print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
russian_stop = set(stopwords.words('russian'))

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    # "min_df":5,
    # "max_df":.9,
    "smooth_idf": False
}


def get_col(col_name): return lambda x: x[col_name]


vectorizer = FeatureUnion([
    ('description', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=18000,
        **tfidf_para,
        preprocessor=get_col('description'))),
    ('text_feat', CountVectorizer(
        ngram_range=(1, 2),
        # max_features=7000,
        preprocessor=get_col('text_feat'))),
    ('title', TfidfVectorizer(
        ngram_range=(1, 2),
        **tfidf_para,
        # max_features=7000,
        preprocessor=get_col('title')))
])

start_vect = time.time()
vectorizer.fit(df.loc[traindex, :].to_dict('records'))
ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes" % ((time.time() - start_vect) / 60))

# Drop Text Cols
df.drop(textfeats, axis=1, inplace=True)

# Dense Features Correlation Matrix
f, ax = plt.subplots(figsize=[10, 7])
sns.heatmap(pd.concat([df.loc[traindex, [x for x in df.columns if x not in categorical]], y], axis=1).corr(),
            annot=False, fmt=".2f", cbar_kws={'label': 'Correlation Coefficient'}, cmap="plasma", ax=ax, linewidths=.5)
ax.set_title("Dense Features Correlation Matrix")
plt.savefig('correlation_matrix.png')

print("Modeling Stage")
# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(df.loc[traindex, :].values), ready_df[0:traindex.shape[0]]])  # Sparse Matrix
testing = hstack([csr_matrix(df.loc[testdex, :].values), ready_df[traindex.shape[0]:]])
tfvocab = df.columns.tolist() + tfvocab
for shape in [X, testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ", len(tfvocab))
del df
gc.collect();

utils.save_bcolz(np.array(X), lgb_root, "lgb_X")
utils.save_bcolz(np.array(y), lgb_root, "lgb_y")
utils.save_bcolz(np.array(tfvocab), lgb_root, "lgb_tfvocab")
utils.save_bcolz(np.array(testing), lgb_root, "lgb_testing")
utils.save_bcolz(np.array(categorical), lgb_root, "lgb_categorical")
utils.save_bcolz(np.array(testdex), lgb_root, "lgb_testdex")