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
from sklearn.model_selection import StratifiedKFold, KFold

import models
import datasets as d
import utils
import json


class rmse(nn.Module):
    def __init__(self):
        super(rmse, self).__init__()

    def forward(self, y, y_hat):
        return torch.sqrt(torch.mean((y-y_hat).pow(2)))


def train_normal(config, X_num, X_cat, X_text, X_word, embedding_weights, y, token_len):
    # Create train/valid dataset and dataloader
    indicates = range(X_num.shape[0])
    _, _, _, _, train_indicates, test_indicates = train_test_split(X_num, y, indicates, test_size=0.1)

    X_train_num = X_num[train_indicates]
    X_train_cat = X_cat[train_indicates]
    X_train_text = [text[train_indicates] for text in X_text]
    X_train_word = [word[train_indicates] for word in X_word]

    y_train = y[train_indicates]
    train_dataset = d.AvitoDataset(X_train_num, X_train_cat,
                                   X_train_text, X_train_word,
                                   y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], 
                                  num_workers=config["n_workers"], shuffle=True)

    X_val_num = X_num[test_indicates]
    X_val_cat = X_cat[test_indicates]
    X_val_text = [text[test_indicates] for text in X_text]
    X_val_word = [word[test_indicates] for word in X_word]
    y_valid = y[test_indicates]
    valid_dataset = d.AvitoDataset(X_val_num, X_val_cat,
                                   X_val_text, X_val_word,
                                   y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], 
                                  num_workers=config["n_workers"], shuffle=True)

    embedding_size = config["embedding_size"]
    # Category model
    cat_model = models.AvitorCat(token_len, embedding_size)
    print("[+] Category model")
    print(cat_model)

    # Numeric model
    num_model = models.AvitorNum(X_train_num.shape[1])
    print("[+] Numeric model")
    print(num_model)

    text_input_shapes = [text.shape[1] for text in X_text]
    dropouts = [0.5 for text in X_text]

    # Text model
    text_model = models.AvitorText(text_input_shapes,
                                   drop_outs=dropouts)
    print("[+] Text model")
    print(text_model)

    # Word model
    word_token_len = [config["word_input_size"] for word in X_word]
    word_model = models.AvitorWord(max_features=config["word_max_dict"],
                                   token_len=word_token_len,
                                   embedding_size=config["word_embedding_size"],
                                   weights=embedding_weights)
    print("[+] Word model")
    print(word_model)

    # FC model
    model = models.Avitor(num_model, cat_model, text_model, word_model)
    print("[+] Summary model")
    print(model)
    # print(model)

    # MSE loss and optimizer
    criterion = nn.MSELoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=config["lr"])
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    start_epoch = 0
    best_val = float("inf")
    is_best = False
    patience = 0

    model_name = config["model_name"]
    epoch_ckp = f"ckp_{model_name}.pth.tar"
    best_ckp = f"ckp_best_{model_name}.pth.tar"
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
    print("[+] Start at epoch {}".format(start_epoch))
    print("[+] Best val: {}".format(best_val))

    for epoch in range(config["epoch"]):
        utils.train(epoch, train_dataloader, model, criterion, optimizer)
        val_loss = utils.test(epoch, valid_dataloader, model, criterion)
        if val_loss < best_val:
            best_val = val_loss
            is_best = True
            patience = 0
        else:
            is_best = False
            patience += 1

        if patience >= config["patience"]:
            print("[+] Early stopping !!!")
            print("[+] Best val {}".format(best_val))
            return

        utils.save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_val": best_val
        }, is_best, model_name, epoch_ckp, best_ckp)


def train_fold(config, n_folds, X_num, X_cat, X_text, X_word, embedding_weights, y, token_len):
    skf = KFold(n_folds)
    for fold, (train_index, test_index) in enumerate(skf.split(X_num)):
        print("[+] Fold: {}".format(fold))
        X_train_num = X_num[train_index]
        X_train_cat = X_cat[train_index]
        X_train_text = [text[train_index] for text in X_text]
        X_train_word = [word[train_index] for word in X_word]

        y_train = y[train_index]
        train_dataset = d.AvitoDataset(X_train_num, X_train_cat,
                                       X_train_text, X_train_word,
                                       y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], 
                                      num_workers=config["n_workers"], shuffle=True)

        X_val_num = X_num[test_index]
        X_val_cat = X_cat[test_index]
        X_val_text = [text[test_index] for text in X_text]
        X_val_word = [word[test_index] for word in X_word]
        y_valid = y[test_index]
        valid_dataset = d.AvitoDataset(X_val_num, X_val_cat,
                                       X_val_text, X_val_word,
                                       y_valid)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], 
                                      num_workers=config["n_workers"], shuffle=True)

        embedding_size = config["embedding_size"]
        # Category model
        cat_model = models.AvitorCat(token_len, embedding_size)
        print("[+] Category model")
        print(cat_model)

        # Numeric model
        num_model = models.AvitorNum(X_train_num.shape[1])
        print("[+] Numeric model")
        print(num_model)

        text_input_shapes = [text.shape[1] for text in X_text]
        dropouts = [0.5 for text in X_text]

        # Text model
        text_model = models.AvitorText(text_input_shapes,
                                       drop_outs=dropouts)
        print("[+] Text model")
        print(text_model)

        # Word model
        word_token_len = [config["word_input_size"] for word in X_word]
        word_model = models.AvitorWord(max_features=config["word_max_dict"],
                                       token_len=word_token_len,
                                       embedding_size=config["word_embedding_size"],
                                       weights=embedding_weights)
        print("[+] Word model")
        print(word_model)

        # FC model
        model = models.Avitor(num_model, cat_model, text_model, word_model)
        print("[+] Summary model")
        print(model)

        # MSE loss and optimizer
        criterion = nn.MSELoss()
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters, lr=config["lr"])
        if torch.cuda.is_available():
            model.cuda()
            criterion.cuda()

        start_epoch = 0
        best_val = float("inf")
        is_best = False
        patience = 0

        model_name = config["model_name"]
        epoch_ckp = f"ckp_{model_name}_{fold}.pth.tar"
        best_ckp = f"ckp_best_{model_name}_{fold}.pth.tar"
        checkpoint_path = f"checkpoint/{model_name}"

        for epoch in range(config["epoch"]):
            utils.train(epoch, train_dataloader, model, criterion, optimizer)
            val_loss = utils.test(epoch, valid_dataloader, model, criterion)
            if val_loss < best_val:
                best_val = val_loss
                is_best = True
                patience = 0
            else:
                is_best = False
                patience += 1

            if patience >= config["patience"]:
                print("[+] Early stopping !!!")
                print("[+] Best val {}".format(best_val))
                break

            utils.save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_val": best_val
            }, is_best, model_name, epoch_ckp, best_ckp)


def main():
    torch.backends.cudnn.benchmark = True

    # Load json config
    config = json.load(open("config.json"))
    extracted_features_root = config["extracted_features"]
    # Load data and token len of embedding layers
    print("[+] Load features ...")
    y = utils.load_features(extracted_features_root, "y_train")
    token_len = utils.load_features(extracted_features_root, "token_len")

    X_train_num = utils.load_features(extracted_features_root, "X_train_num")
    X_train_cat = utils.load_features(extracted_features_root, "X_train_cat")
    X_train_desc = utils.load_features(extracted_features_root, "X_train_desc").any()
    X_train_title = utils.load_features(extracted_features_root, "X_train_title").any()
    # X_train_word_desc = utils.load_features(extracted_features_root, "X_train_word_description")
    # X_train_word_title = utils.load_features(extracted_features_root, "X_train_word_title")
    embedding_weights = utils.load_bcolz(extracted_features_root, "embedding_weights")
    X_train_word = [utils.load_bcolz(extracted_features_root, "X_train_word")]

    X_train_text = [X_train_desc, X_train_title]
    # X_train_word = [X_train_word_desc, X_train_word_title]

    n_folds = config["n_fold"]
    if n_folds:
        train_fold(config, n_folds,
                   X_train_num, X_train_cat,
                   X_train_text, X_train_word,
                   embedding_weights, y, token_len)
    else:
        train_normal(config,
                   X_train_num, X_train_cat,
                   X_train_text, X_train_word,
                   embedding_weights, y, token_len)

if __name__ == '__main__':
    main()