# 1. Overall
- This folder is used for develop new Neural Network architecture which will use same (or almost same) features as LGBM and XGBOOST.
- This folder is totally seperated as previous version. It means that you can run the code and dont mind about previous code before.

# 2. Baseline
- I found the feature extraction really wastes of time (around 40 min). So, the idea is we will extract full feature into numpy/bcolz first. 
Then model will load those features again.
- I recommend to extract all features as much as possible. To avoid doing feature extraction many time, I implemented a method that can remove
ununsed features at the first stage of running model. Those methods are also able to apply to LGBM or XGB.
- Whenever you want to change/add/modify features, I would not take time to extract unchanged features as LGB and XGB do.

## 2.1 Prepare
Make sure *checkpoint* and *features* folder are created
```
cd scripts
mkdir checkpoint
mkdir features
```  
## 2.2 Extract features
Please make sure you change the *root* path in each file corretly

- Extract numeric features
  ```
  python ext_ft_numeric.py
  ```
  Output:
  - features/X_train_num
  - features/X_test_num
  - features/X_train_y
 
- Extract category features
  ```
  python ext_ft_category.py
  ```
  Output:
  - features/X_train_cat
  - features/X_test_cat
  
- Extract TFIDF and ridge features
  ```
  python ext_ft_tfidf.py
  ```
  Output:
  - features/X_train_tfidf_text
  - features/X_test_tfidf_text
  - features/X_train_tfidf_params
  - features/X_test_tfidf_params
  - features/X_train_ridge_text
  - features/X_test_ridge_text
  - features/X_train_ridge_params
  - features/X_test_ridge_params
  
 - Extract word features
 TBD
 
 # 3. Runs the models
 LBM, XGB and NN can use the extracted features above.
 
 ## 3.1 Neural Network
 Neural network supports arguments to config and run the model.
 
 - To train
 ```
 python nn.py --mode train --batch_size 1024 --epochs 100 --lr 0.001 --kfold 10 --model_name avito_model
 ```
 - To test
  ```
 python nn.py --mode test --batch_size 1024 --epochs 100 --lr 0.001 --kfold 10 --model_name avito_model
 ```
 - To remove features  
 Check variable: *unused_num* and *unused_cat* in *nn.py*
 
 ## 3.2 LGBM
 TBD
 
 ## 3.3 XGB
 TBD
