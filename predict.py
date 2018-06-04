import json
import pandas as pd
import bcolz
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import models
import datasets as d
import utils
import json
from tqdm import tqdm


def predict_fold(config, n_folds, X_num, X_cat, X_text, token_len):
    test_dataset = d.AvitoDataset(X_num, X_cat,
                                  X_text, None)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], 
                                 num_workers = 8, shuffle=False)
    preds_all = []
    predict_root = config["predict_root"]
    for fold in range(n_folds):
        print("[+] Predict fold: {}".format(fold))
        embedding_size = config["embedding_size"]
        # Category model
        cat_model = models.AvitorCat(token_len, embedding_size)
        print("[+] Category model")
        print(cat_model)

        # Numeric model
        num_model = models.AvitorNum(X_num.shape[1])
        print("[+] Numeric model")
        print(num_model)

        text_input_shapes = [text.shape[1] for text in X_text]
        dropouts = [0.5 for text in X_text]
        # Text model
        text_model = models.AvitorText(text_input_shapes,
                                       drop_outs=dropouts)
        print("[+] Text model")
        print(text_model)

        # FC model
        model = models.Avitor(num_model, cat_model, text_model)
        print("[+] Summary model")
        print(model)

        if torch.cuda.is_available():
            model.cuda()

        best_val = 0

        model_name = config["model_name"]
        best_ckp = f"ckp_best_{model_name}_{fold}.pth.tar"
        checkpoint_path = f"checkpoint/{model_name}"

        checkpoint_name = f"{checkpoint_path}/{best_ckp}"
        ckp = utils.load_checkpoint(checkpoint_name)
        if ckp:
            model.load_state_dict(ckp["state_dict"])
            best_val = ckp["best_val"]

        print("[+] Model name " + model_name)
        print("[+] Best val {}".format(best_val))

        model.eval()
        preds = []
        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for batch_id, (X_num, X_cat, X_text, _) in pbar:
            output = model(X_num, X_cat, X_text)
            preds += output.data.cpu().numpy().tolist()

        preds = [p[0] for p in preds]
        preds_all.append(preds)
        submission = pd.read_csv(config["sample_submission"])
        submission['deal_probability'] = preds
        #submission.to_csv(f"submission_{model_name}_{fold}.csv", index=False)
        utils.save_csv(submission, predict_root, f"submission_{model_name}_{fold}.csv")
    preds_all = np.array(preds_all)
    preds_avg = np.mean(preds_all, axis=0)
    submission = pd.read_csv(config["sample_submission"])
    submission['deal_probability'] = preds_avg
    #submission.to_csv(f"submission_{model_name}_avg.csv", index=False)
    utils.save_csv(submission, predict_root, f"submission_{model_name}_avg.csv")


def predict_one(config, X_num, X_cat, X_text, token_len):
    test_dataset = d.AvitoDataset(X_num, X_cat,
                                  X_text, None)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], 
                                 num_workers = 8, shuffle=False)

    embedding_size = config["embedding_size"]
    # Category model
    cat_model = models.AvitorCat(token_len, embedding_size)
    print("[+] Category model")
    print(cat_model)

    # Numeric model
    num_model = models.AvitorNum(X_num.shape[1])
    print("[+] Numeric model")
    print(num_model)

    text_input_shapes = [text.shape[1] for text in X_text]
    dropouts = [0.5 for text in X_text]
    # Text model
    text_model = models.AvitorText(text_input_shapes,
                                   drop_outs=dropouts)
    print("[+] Text model")
    print(text_model)

    # FC model
    model = models.Avitor(num_model, cat_model, text_model)
    print("[+] Summary model")
    print(model)

    if torch.cuda.is_available():
        model.cuda()

    best_val = 0

    model_name = config["model_name"]
    checkpoint_path = f"checkpoint/{model_name}"

    if config["resume"]:
        resume = config["resume"]
        checkpoint_name = f"{checkpoint_path}/{resume}"
        ckp = utils.load_checkpoint(checkpoint_name)
        if ckp:
            model.load_state_dict(ckp["state_dict"])
            start_epoch = ckp["epoch"]
            best_val = ckp["best_val"]

    print("[+] Model name " + model_name)
    print("[+] Best val {}".format(best_val))

    model.eval()

    preds = []
    predict_root = config["predict_root"]
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for batch_id, (X_num, X_cat, X_text, _) in pbar:
        output = model(X_num, X_cat, X_text)
        preds += output.data.cpu().numpy().tolist()

    preds = [p[0] for p in preds]
    submission = pd.read_csv(config["sample_submission"])
    submission['deal_probability'] = preds
    #submission.to_csv(f"submission_{model_name}_one.csv", index=False)
    utils.save_csv(submission, predict_root, f"submission_{model_name}_one.csv")

def main():
    # Load json config
    config = json.load(open("config.json"))
    extracted_features_root = config["extracted_features"]
    print("[+] Load features ...")

    X_test_num = utils.load_features(extracted_features_root, "X_test_num")
    X_test_cat = utils.load_features(extracted_features_root, "X_test_cat")
    X_test_desc = utils.load_features(extracted_features_root, "X_test_desc").any()
    X_test_title = utils.load_features(extracted_features_root, "X_test_title").any()
    #X_test_param = utils.load_features(extracted_features_root, "X_test_param").any()

    token_len = utils.load_features(extracted_features_root, "token_len")

    #X_test_text = [X_test_desc, X_test_title, X_test_param]
    X_test_text = [X_test_desc, X_test_title]

    n_folds = config["n_fold"]
    if n_folds:
        predict_fold(config, n_folds, X_test_num,
                    X_test_cat, X_test_text,
                    token_len)
    else:
        predict_one(config, X_test_num,
                    X_test_cat, X_test_text,
                    token_len)

if __name__ == '__main__':
    main()