import pandas as pd
import numpy as np
import xgboost as xgb
from itertools import product

from utils import surv_grid, survival_preprocess, res_cov
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import mean_absolute_error, mean_squared_error
from SurvivalEVAL.Evaluator import LifelinesEvaluator
from lifelines.utils import concordance_index
datapath = 'E:/GitHub/SNGCCA/SNGCCA/RealData/'
method = 'resdg/'

filter_data, y = survival_preprocess(datapath)
data = res_cov(datapath, method)

filter_data = pd.concat([data, filter_data], axis=1)
# split data into training and testing sets
c_index_list = []
mse_list = []
mae_list = []

for i in range(100):
  print(i)
  random_state = np.random.randint(0, 1000)
  X_train, X_test, y_train, y_test = train_test_split(filter_data, y, test_size=0.2, random_state=random_state)

  y_lower = np.where(X_train['vital_status'] == 1, X_train['time_to_event'], -np.inf)  # For uncensored: observed time
  y_upper = np.where(X_train['vital_status'] == 1, X_train['time_to_event'], np.inf)  # For censored: [time, +inf]

  # cross validation
  # Define parameter grid
  #param_grid = {
  #    'learning_rate': [0.01, 0.1],
  #    'max_depth': [3, 6],
  #    'aft_loss_distribution': ['normal', 'logistic'],
  #    'aft_loss_distribution_scale': [1.0, 2.0],
  #    'lambda': [0.1, 1.0],
  #    'alpha': [0.1, 1.0]
  #}
  param_grid = {
      'learning_rate': [0.01, 0.1],
      'max_depth': [3, 5],
      'aft_loss_distribution_scale': [0.5, 1.0],
      'alpha': [0.1, 0.5, 1.0],
  }

  param_combinations = list(product(
      param_grid['learning_rate'],
      param_grid['max_depth'],
      param_grid['aft_loss_distribution_scale'],
      param_grid['alpha']
  ))


  # 100 epochs random selection
  # create DMatrix for training
  dmat = xgb.DMatrix(
    # Features
    data = X_train.iloc[:, :-2],
    # Label
    label_lower_bound = y_lower, label_upper_bound = y_upper,
    missing = float("NaN"), 
    enable_categorical = True
  )
  if i == 0:
    best_param = surv_grid(param_combinations, dmat)

  params = {'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': 'normal',
            'aft_loss_distribution_scale': best_param[2],
            'tree_method': 'hist', 'learning_rate': best_param[0], 
            'max_depth': best_param[1], 'alpha': best_param[3]}
  bst = xgb.train(params, dmat, num_boost_round=1000)

  y_lower_test = np.where(X_test['vital_status'] == 1, X_test['time_to_event'], -np.inf)  # For uncensored: observed time
  y_upper_test = np.where(X_test['vital_status'] == 1, X_test['time_to_event'], np.inf)  # For censored: [time, +inf]

  dmat_test = xgb.DMatrix(
    # Features
    data = X_test.iloc[:, :-2],
    # Label
    label_lower_bound = y_lower_test, label_upper_bound = y_upper_test,
    missing = float("NaN"), 
    enable_categorical = True
  )
  pred = bst.predict(dmat_test)

  # Make the evaluation
  cindex = concordance_index(X_test['time_to_event'], pred, X_test['vital_status'])
  #cindex = concordance_index_censored(X_test['vital_status'].astype(bool), X_test['time_to_event'], -pred)
  c_index_list.append(cindex)
  #y_true = X_test[X_test["vital_status"] == 1]["time_to_event"]
  #y_pred = pred[X_test["vital_status"] == 1]
  mse = mean_squared_error(X_test['time_to_event'], pred)
  mse_list.append(mse)
  mae = mean_absolute_error(X_test['time_to_event'], pred)
  mae_list.append(mae) 

print(f"Concordance index: {np.mean(c_index_list):.4f}", f"MSE: {np.mean(mse_list):.4f}", f"MAE: {np.mean(mae_list):.4f}")
print(f"Concordance index: {np.std(c_index_list)/100:.4f}", f"MSE: {np.std(mse_list)/100:.4f}", f"MAE: {np.std(mae_list)/100:.4f}")