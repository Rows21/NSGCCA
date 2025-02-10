import pandas as pd
import numpy as np
import xgboost as xgb
from itertools import product
from tqdm import tqdm
from NSGCCA.networks.utils import surv_grid_xgb, survival_preprocess, res_cov
from SurvivalEVAL.Evaluator import PointEvaluator
from sksurv.metrics import concordance_index_ipcw
from sklearn.model_selection import train_test_split
import sys

datapath = '/scratch/rw2867/projects/SNGCCA/SNGCCA/RealData/'
#datapath = 'E:/res/SNGCCA/SNGCCA/RealData/'
#methods = ['ressgccaadmm/','ressng/','restsk/','respdd/','ressak/','resdg/']'resdg/','ressak/','restsk/', 'respdd/', 
methods = ['ressng/', 'ressgccaadmm/','resdg/','ressak/','restsk/', 'respdd/', None]

random_state = sys.argv[1]

print(random_state)
res = pd.DataFrame(columns=['method', 'cindex1', "cindex2", 'rmse', 'mae'])

for method in methods:
  filter_data, y = survival_preprocess(datapath)
  data = res_cov(datapath + 'newData/', method)
  print(method)
  filter_data = pd.concat([pd.DataFrame(data), filter_data], axis=1)
  # split data into training and testing sets
  #c_index_list = []
  #rmse_list = []
  #mae_list = []

  #for i in tqdm(range(10)):
    #print(i)
  #random_state = np.random.randint(0, 1000)
  X_train, X_test, y_train, y_test = train_test_split(filter_data, y, test_size=0.2, random_state=int(random_state), stratify=y)

  y_lower = X_train['time_to_event'].values#np.where(X_train['vital_status'] == 1, X_train['time_to_event'], -np.inf)  # For uncensored: observed time
  y_upper = np.where(X_train['vital_status'] == 1, X_train['time_to_event'], +np.inf)  # For censored: [time, +inf]
    #X_train['vital_status'] = 1 - X_train['vital_status']
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
        'max_depth': [3, 5, 7],
        'aft_loss_distribution_scale': [0.5, 1.1, 1.7],
        'alpha': [0.1, 1.0, 10.0],
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
      label_lower_bound = y_lower, 
      label_upper_bound = y_upper,
      missing = float("NaN"), 
      enable_categorical = True
    )
    #if i == 0:
  best_param = surv_grid_xgb(param_combinations, dmat)
    #best_param = [0.1,3,1.1,1]
    #print(best_param)
  params = {'objective': 'survival:aft',
              'eval_metric': 'aft-nloglik',
              'aft_loss_distribution': 'normal',
              'aft_loss_distribution_scale': best_param[2],
              'learning_rate': best_param[0], 
              'max_depth': best_param[1], 'alpha': best_param[3]}
  bst = xgb.train(params, dmat, num_boost_round=1000)

  y_lower_test = X_test['time_to_event'].values#np.where(X_test['vital_status'] == 1, X_test['time_to_event'], -np.inf)  # For uncensored: observed time
  y_upper_test = np.where(X_test['vital_status'] == 1, X_test['time_to_event'], +np.inf) # For censored: [time, +inf]
    #X_train['vital_status'] = 1 - X_train['vital_status']
  dmat_test = xgb.DMatrix(
      # Features
      data = X_test.iloc[:, :-2],
      # Label
      label_lower_bound = y_lower_test, 
      label_upper_bound = y_upper_test,
      missing = float("NaN"), 
      enable_categorical = True
    )
  pred = bst.predict(dmat_test)
  eval = PointEvaluator(pred, X_test['time_to_event'],X_test['vital_status'], X_train['time_to_event'],X_train['vital_status']) 
  # Make the evaluation
  
  cindex1, _, _ = eval.concordance()
  cindex2, _, _ = eval.concordance(pair_method="Margin")
  #train_eval = np.array(
  #      [(e, t) for e, t in zip(X_train['vital_status'], X_train['time_to_event'])],
  #      dtype=[("event", "bool"), ("time", "float")]
  #  )
    
  #test_eval = np.array(
  #      [(e, t) for e, t in zip(X_test['vital_status'], X_test['time_to_event'])],
  #      dtype=[("event", "bool"), ("time", "float")]
  #  )
  #cindex = concordance_index_ipcw(train_eval, test_eval, -pred)[0]
  mae = eval.mae(method="Hinge")
  rmse = eval.rmse(method="Hinge")
  X_test['pred'] = pred
  if method is None:
    X_test.to_csv('/scratch/rw2867/projects/SNGCCA/SNGCCA/RealData2/survival_results' + sys.argv[1] + 'None.csv', index=False)
  else:
    X_test.to_csv('/scratch/rw2867/projects/SNGCCA/SNGCCA/RealData2/survival_results' + sys.argv[1] + method.replace("/","") + '.csv', index=False)
  res.loc[len(res)] = [method, cindex1, cindex2, rmse, mae]
print(res)
res.to_csv('/scratch/rw2867/projects/SNGCCA/SNGCCA/RealData2/survival_results' + sys.argv[1] + '.csv', index=False)
    #cindex = concordance_index(X_test['time_to_event'], pred, X_test['vital_status'])
    #c_index_list.append(cindex)
    #print(cindex)
    #mse = mean_squared_error(X_test['time_to_event'], pred)
    #rmse_list.append(rmse)
    #mae = mean_absolute_error(X_test['time_to_event'], pred)
    #mae_list.append(mae)

  #print(f"Concordance index: {np.mean(c_index_list):.4f}", 
  #      f"MSE: {np.mean(rmse_list):.4f}", 
  #      f"MAE: {np.mean(mae_list):.4f}")

  #print(f"Concordance index: {np.std(c_index_list)/np.sqrt(10):.4f}", 
  #      f"MSE: {np.std(rmse_list)/np.sqrt(10):.4f}", 
  #      f"MAE: {np.std(mae_list)/np.sqrt(10):.4f}")