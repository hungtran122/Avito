import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')

import keras as K
from keras.layers import *
from keras.models import *
import json
import utils

import keras
from keras.callbacks import *
from keras_utils import AttentionWithContext
from keras.regularizers import l2
import gc
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import argparse
import pandas as pd
import keras_model as kmodel
import tensorflow as tf

from keras.callbacks import CSVLogger

from keras.utils import plot_model
config = json.load(open("config.json"))

# Load json config
config = json.load(open("config.json"))
extracted_features_root = config["extracted_features"]
# Load data and token len of embedding layers
print("[+] Load features ...")
y = utils.load_features(extracted_features_root, "y_train")
token_len = utils.load_features(extracted_features_root, "token_len")

X_num = utils.load_features(extracted_features_root, "X_train_num")
X_cat = utils.load_features(extracted_features_root, "X_train_cat")
X_desc = utils.load_features(extracted_features_root, "X_train_desc").any()
X_title = utils.load_features(extracted_features_root, "X_train_title").any()

embedding_weights = utils.load_bcolz(extracted_features_root, "embedding_weights")
X_word = utils.load_bcolz(extracted_features_root, "X_train_word")

X_text = [X_desc, X_title]

del X_desc, X_title
gc.collect()

cat_columns = [
    "region", # Importance feature, best_val: 0.507
    "city", # Importance feature, best_val: 0.510
    "parent_category_name", # This feature seems not be importance, best_val:0.0505
    "category_name", # This feature seems be importance, best_val:0.0507
    "param_1", "param_2", "param_3",
    "user_type", "image_top_1"
]

cat_embedding_size = [
    3,
    5,
    7,
    9,
    7,
    5,
    3,
    5,
    7,
]

text_columns = [
    "description",
    "title"
]

model_name = None

train_nlp = 1

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def get_model(args):
    input_num = Input(shape=(X_num.shape[1],), name="numeric")

    input_cat = []
    for cat, name in zip(range(X_cat.shape[1]), cat_columns):
        input_cat.append(Input(shape=(1,), name="cat_" + name))

    input_text = []
    for text, name in zip(X_text, text_columns):
        input_text.append(Input(shape=(text.shape[1],), name="text_" + name))

    if train_nlp:
        input_words = Input((config["word_input_size"],), name="word")

    out_num = BatchNormalization()(input_num)
    out_num = Dense(50, activation="relu", kernel_initializer="glorot_normal")(out_num)
    out_num = BatchNormalization()(out_num)
    out_num = Dropout(0.5)(out_num)

    out_cat = []
    for x, tkl, embed_size in zip(input_cat, token_len, cat_embedding_size):
        x = Embedding(tkl + 1, embed_size, embeddings_initializer="glorot_normal")(x)
        x = SpatialDropout1D(0.25)(x)
        x1 = GlobalAveragePooling1D()(x)
        x2 = GlobalMaxPooling1D()(x)
        x = Concatenate()([x1, x2])
        # x = Flatten()(x)
        out_cat.append(x)

    out_text = []
    text_input_size = [txt.shape[1] for txt in X_text]
    for x, text_size in zip(input_text, text_input_size):
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(32, activation="linear", kernel_initializer="glorot_normal")(x)
        x = Dropout(0.5)(x)
        out_text.append(x)


    if train_nlp:
        if args.architecture == "bilstm_amp":
            x_words = kmodel.BidLstmAmp(input_words, config["word_max_dict"], config["word_embedding_size"], embedding_weights)
        elif args.architecture == "bilstm_ap":
            x_words = kmodel.BidLstmAmp(input_words, config["word_max_dict"], config["word_embedding_size"],
                                        embedding_weights)
        elif args.architecture == "bilstm_mp":
            x_words = kmodel.BidLstmMp(input_words, config["word_max_dict"], config["word_embedding_size"],
                                        embedding_weights)
        elif args.architecture == "bilstm_mpatn":
            x_words = kmodel.BidLstmMpAtn(input_words, config["word_input_size"], config["word_max_dict"],
                                          config["word_embedding_size"],
                                          embedding_weights)
        elif args.architecture == "bigru":
            x_words = kmodel.BidGRU(input_words, config["word_input_size"], config["word_max_dict"],
                                          config["word_embedding_size"],
                                          embedding_weights)
        elif args.architecture == "rnnv2":
            x_words = kmodel.RNNV2(input_words, config["word_max_dict"],
                                          config["word_embedding_size"],
                                          embedding_weights)
        elif args.architecture == "capsule":
            x_words = kmodel.CapsuleNet(input_words, config["word_max_dict"],
                                          config["word_embedding_size"],
                                          embedding_weights)
        elif args.architecture == "cnn":
            x_words = kmodel.CNN(input_words, config["word_max_dict"],
                                          config["word_embedding_size"],
                                          embedding_weights)

    merg_out_text = concatenate(out_text)
    merg_out_text = BatchNormalization()(merg_out_text)
    merg_out_text = Dense(128, activation="relu", kernel_initializer="glorot_normal")(merg_out_text)
    merg_out_text = Dropout(0.5)(merg_out_text)

    if train_nlp:
        merge_out = concatenate([out_num, input_num, *out_cat, merg_out_text, x_words])
    else:
        merge_out = concatenate([out_num, input_num, *out_cat, merg_out_text])

    merge_out = BatchNormalization()(merge_out)
    merge_out = Dense(50, activation="relu", kernel_initializer="glorot_normal")(merge_out)
    merge_out = BatchNormalization()(merge_out)
    merge_out = Dense(1, activation="sigmoid", kernel_initializer="glorot_normal")(merge_out)

    if train_nlp:
        model = Model(inputs=[input_num, *input_cat, *input_text, input_words], outputs=merge_out)
    else:
        model = Model(inputs=[input_num, *input_cat, *input_text], outputs=merge_out)
    model.compile(optimizer=optimizers.Adam(lr=config["lr"]), loss="mean_squared_error", metrics=[rmse])

    plot_model(model, to_file='model.png')
    return model


def train(args):
    checkpoint_path = f"checkpoint/{model_name}/"

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    file_path = f"{checkpoint_path}/keras_best.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True,
                                 mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=1e-4)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=2,
                                   verbose=1,
                                   epsilon=1e-4,
                                   min_lr=1e-5,
                                   mode='min')

    list_train_cat = [X_cat[:, i] for i in range(X_cat.shape[1])]

    n_folds = config["n_fold"]

    if n_folds:
        # Train with k-fold
        skf = KFold(n_folds, shuffle=True, random_state=2018)
        for fold, (train_index, val_index) in enumerate(skf.split(X_num)):

            model = get_model(args)
            # model.summary()

            print(f"\n[+] Fold {fold}")

            file_path = f"{checkpoint_path}/keras_{model_name}_best_{fold}.h5"
            checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=2, save_best_only=True,
                                         save_weights_only=True,
                                         mode='min')
            csv_logger = CSVLogger(f'{checkpoint_path}/log_{model_name}_{fold}.csv', append=True, separator=',')
            callbacks_list = [checkpoint, early, lr_reduced, csv_logger]

            X_tr_fold_num = X_num[train_index]
            X_tr_fold_cat = [cat[train_index] for cat in list_train_cat]
            X_tr_fold_text = [text[train_index] for text in X_text]
            X_tr_fold_word = X_word[train_index]
            y_tr_fold = y[train_index]

            X_val_fold_num = X_num[val_index]
            X_val_fold_cat = [cat[val_index] for cat in list_train_cat]
            X_val_fold_text = [text[val_index] for text in X_text]
            X_val_fold_word = X_word[val_index]
            y_val_fold = y[val_index]

            if train_nlp:
                print("[+] Train with NLP")
                train_params = [X_tr_fold_num, *X_tr_fold_cat, *X_tr_fold_text, X_tr_fold_word]
                val_params = [X_val_fold_num, *X_val_fold_cat, *X_val_fold_text, X_val_fold_word]
            else:
                print("[+] Train without NLP")
                train_params = [X_tr_fold_num, *X_tr_fold_cat, *X_tr_fold_text]
                val_params = [X_val_fold_num, *X_val_fold_cat, *X_val_fold_text]

            history = model.fit(train_params, y_tr_fold,
                                validation_data=(val_params, y_val_fold),
                                verbose=1, callbacks=callbacks_list,
                                epochs=config["epoch"], batch_size=config["batch_size"])
    else:
        model = get_model(args)
        model.summary()

        csv_logger = CSVLogger(f'{checkpoint_path}/log_{model_name}_one.csv', append=True, separator=',')
        callbacks_list = [checkpoint, early, lr_reduced, csv_logger]

        if train_nlp:
            train_params = [X_num, *list_train_cat, *X_text, X_word]
        else:
            train_params = [X_num, *list_train_cat, *X_text]

        history = model.fit(train_params, y, validation_split=0.1,
                            verbose=1, callbacks=callbacks_list,
                            epochs=config["epoch"], batch_size=config["batch_size"])


def test(args):
    predict_root = config["predict_root"]
    checkpoint_path = f"checkpoint/{model_name}/"
    file_path = f"{checkpoint_path}/keras_best.h5"
    list_cat = [X_cat[:, i] for i in range(X_cat.shape[1])]

    n_folds = config["n_fold"]

    if train_nlp:
        test_params = [X_num, *list_cat, *X_text, X_word]
    else:
        test_params = [X_num, *list_cat, *X_text]

    if n_folds:
        # Test with k-fold
        preds_all = []
        for fold in range(n_folds):
            model = get_model(args)
            print(f"\n[+] Test Fold {fold}")
            file_path = f"{checkpoint_path}/keras_{model_name}_best_{fold}.h5"
            model.load_weights(file_path)
            pred = model.predict(test_params, batch_size=512)
            submission = pd.read_csv(config["sample_submission"])
            submission['deal_probability'] = pred
            utils.save_csv(submission, predict_root, f"keras_{model_name}_{fold}.csv")
            preds_all.append(pred)
        preds_all = np.array(preds_all)
        preds_avg = np.mean(preds_all, axis=0)
        submission = pd.read_csv(config["sample_submission"])
        submission['deal_probability'] = preds_avg
        # submission.to_csv(f"submission_{model_name}_avg.csv", index=False)
        utils.save_csv(submission, predict_root, f"keras_{model_name}_avg.csv")
    else:
        model = get_model(args)
        model.summary()
        model.load_weights(file_path)
        pred = model.predict([X_num, *list_cat, *X_text, X_word], batch_size=512)
        submission = pd.read_csv(config["sample_submission"])
        submission['deal_probability'] = pred
        utils.save_csv(submission, predict_root, f"keras_{model_name}_one.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('architecture', choices = [
        "bilstm_amp", "bilstm_ap", "bilstm_mp",
        "bilstm_mpatn", "bigru", "rnnv2", "capsule",
        "cnn"
    ])
    parser.add_argument('train_nlp', choices = [
        "0", "1"
    ])

    args = parser.parse_args()

    print(f"[+] Start {args.mode}")

    model_name = args.architecture
    print(f"[+] Model {model_name}")

    train_nlp = int(args.train_nlp)
    print(f"[+] Train nlp {train_nlp}")

    if args.mode == "test":
        X_num = utils.load_features(extracted_features_root, "X_test_num")
        X_cat = utils.load_features(extracted_features_root, "X_test_cat")
        X_desc = utils.load_features(extracted_features_root, "X_test_desc").any()
        X_title = utils.load_features(extracted_features_root, "X_test_title").any()
        X_text = [X_desc, X_title]
        X_word = utils.load_bcolz(extracted_features_root, "X_test_word")

    # model = get_model()
    # model.summary()
    # plot_model(model, to_file='model.png')

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)