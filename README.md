*** Originated from https://github.com/ngxbac/Avito with great contribution from my teammate nxbac
# 1. Overall
This repository includes the code of the competition "_Avito Demand Prediction Challenge_" hosted by Kaggle.
We built three models for this competition:
- LightGBM
- Xgboost
- Deep neural network (DNN)

# 2. config.json
Include settings as following:
- train_csv: path of train.csv
- test_csv: path of test.csv
- train_norm_csv: path of train_norm.csv (after text normalization)
- test_norm_csv: path of test_norm.csv (after text normalization)
- sample_submission: path of sample submission
- fasttext_vec: pretrained embedding model
- extracted_features: folder to save the features
- predict_root: folder to save prediction of DNN
- word_embedding_size: word embedding size
- word_max_dict: maximum number of dictionary
- word_input_size: number of word in an input of NLP part
- word_max_sent: Not use
- lr: learning rate
- epoch: number of running epoch
- batch_size: batch size (depends on your GPU)
- embedding_size: embedding size of categorical. Currently it is unused
- n_workers: number of thread used for DNN training
- model_name: prefix of model
- patience: patience for early stopping
- n_fold: number of fold
- resume: unused

# 3. Deep Neural Network
To run deep neural network, you should following two parts.
- Feature extraction  
  - _extract_features.py_ : extract category and numeric (normal numeric data + TFIDF of title and description, then save them to numpy and bcolz files. 
  - _extract_word.py_ : extract word features for NLP part in Deep neural network (DNN)
- How to add more features  
  - Numeric features
The numeric features are saved in the list _num_columns_ in the file _extract_features.py_
If you define new columns as a new feature in dataframe, please append new features to _num_columns_ list.
At the end of script, the features will be auto extracted to numpy.
Ex: 
    ```python
    df["new_num_feature"] = new_feature_data
    num_columns.append("new_num_feature")
    ```  
  - The category features  
    Similar as numeric features.
  - Word feature for NLP  
    TBD
- How to run
First, you should to change all the paths in the _config.json_ to be suiatable for your environment.
Second, run the following scripts to start extracting features and training DNN
     ```sh
    python extract_features.py
    python extract_word.py
    python keras_nlp train capsule 1
    python keras_nlp test capsule 1
    ```  
# 4. LightGBM
TBD

# 5. Xgboost
TBD
