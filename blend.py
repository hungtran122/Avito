import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Any results you write to the current directory are saved as output.

root = "model_bag"

model_bag = [
    {
        "name": "xgboost",
        "path": f"{root}/XGB_0216192_local.csv",
        "score": 0.2208,
        "weight": 0.3,
    },
    {
        "name": "LightGBM",
        "path": f"{root}/LGBM_02209_no_ridge.csv",
        "score": 0.2207,
        "weight": 0.3,
    },
    {
        "name": "xgboost_R",
        "path": f"{root}/xgb_tfidf0.218395.csv",
        "score": 0.2234,
        "weight": 0.1,
    },
    {
        "name": "keras_capsule",
        "path": f"{root}/keras_capsule_02215_self_training.csv",
        "score": 0.2215,
        "weight": 0.3,
    },
    # {
    #     "name": "keras_capsule_2",
    #     "path": f"{root}/keras_capsule_02218_fasttext.csv",
    #     "score": 0.2218,
    #     "weight": 0.225,
    # },
]

preds = []
for model in model_bag:
    print("\n[+] Load model {}".format(model["name"]))
    print("[+] Score {}".format(model["score"]))
    print("[+] path {}".format(model["path"]))
    label  = pd.read_csv(str(model["path"]))
    weight = model["weight"]
    preds.append(weight * label["deal_probability"].values)

preds = np.array(preds)
pred_mean = np.sum(preds, axis=0)

df = pd.DataFrame()
df['item_id'] = label['item_id']
df['deal_probability'] = pred_mean
df.to_csv(f'{root}/blend_L09NR_X08_N15_R34_play.csv',index=False)