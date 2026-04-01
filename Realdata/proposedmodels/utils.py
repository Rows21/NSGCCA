#import torch
import numba
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
#from DeepQuantreg import deep_quantreg as dq
#from DeepQuantreg import utils as utils
from tqdm import tqdm
from sklearn.model_selection import train_test_split
#from SurvivalEVAL.Evaluator import PointEvaluator
from scipy.stats import ttest_rel
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print("Using", torch.cuda.device_count(), "GPUs")
device = 'cpu'
#if device == 'cuda':
#    import cupy as cp
#def sqdist(X1, X2):
#        n1 = X1.shape[1]
#        n2 = X2.shape[1]
#        D = torch.sum(X1 ** 2, dim=0).reshape(-1, 1).repeat(1, n2) + torch.sum(X2 ** 2, dim=0).reshape(1, -1).repeat(n1,
#            1) - 2 * torch.mm(X1.T, X2)
#        return D

#def rbf_kernel(self, X, sigma=None):
        # dist
#        D = torch.sqrt(torch.abs(self.sqdist(X.t(), X.t())))

#        if sigma is None:
            # median sigma
#            sigma = torch.median(D)

        # kernel
#        K = torch.exp(- (D ** 2) / (2 * sigma ** 2))
#        return K, sigma
    
#def rbf_kx(x, Pi, sigma=None):
#    n = x.shape[0]
#    Kx = np.zeros((n, n))
#    for i in range(n):
#        for j in range(n):
#            Kx[i, j] = np.trace(Pi @ np.outer(x[i] - x[j], x[i] - x[j]))
#                
#    Kx = np.exp(- Kx / 2)
#    return Kx

def is_invertible(matrix):
  try:
    np.linalg.inv(matrix)
    return True
  except np.linalg.LinAlgError:
    return False

def rbf_kx(x:np.ndarray, Pi:np.ndarray, sigma=None):
    # Calculate pairwise differences
    n = x.shape[0]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    Kx = np.einsum('ijk,kl,ijl->ij', diff, Pi, diff)
    Kx = np.exp(- Kx / 2)
    return Kx

def dF1_vectorized(view: np.ndarray, Pi: np.ndarray, K: np.ndarray, i_mask: int = None):
    """
    view: (n, p)
    Pi  : (p, p)
    K   : (n, n)   # 样本对权重（例如 rbf 核或别的系数矩阵）
    i_mask: 可选，对样本对 (a,b) 做上三角遮罩时的偏移（通常不需要，保留接口）
    return:
        dF1: (p, p)
    计算公式（向量化）：
        diff[a,b,:] = view[a,:] - view[b,:]                  # (n,n,p)
        quad[a,b]   = diff[a,b,:]^T Pi diff[a,b,:]           # (n,n)
        W[a,b]      = K[a,b] * exp(-quad[a,b]/2)             # (n,n)
        dF1         = sum_{a,b} W[a,b] * diff[a,b,:] ⊗ diff[a,b,:]   # (p,p)
    """
    x = view  # (n,p)

    # 所有样本对的差: (n, n, p)
    diff = x[:, None, :] - x[None, :, :]

    # 每个样本对的二次型标量: (n, n)
    quad = np.einsum('ijp,pq,ijq->ij', diff, Pi, diff)

    # 权重矩阵：K * exp(-quad/2)  (n, n)
    W = K * np.exp(-0.5 * quad)

    # 如需只累加上三角可加遮罩（一般不需要）
    if i_mask is not None:
        mask = np.triu(np.ones_like(W, dtype=bool), k=i_mask+1)
        W = np.where(mask, W, 0.0)

    # 合成 dF1：把每个 (a,b) 的外积加权求和  → (p, p)
    dF1 = np.einsum('ij,ijp,ijq->pq', W, diff, diff)

    # 与你原式一致的 1/n^2 归一化（如果需要）
    n = view.shape[0]
    dF1 = dF1 / (n ** 2)

    return dF1

def rbf_kl(sum_K):
    n = len(sum_K)
    I_n = np.eye(n)
    H = I_n - np.outer(np.ones(n), np.ones(n)) / n
    return H @ sum_K @ H
    
#def z(x, p):
#    n = x.shape[0]
#    Z_F2 = np.zeros((n, n))
#    for i in range(n):
#        for j in range(n):
#            Z_F2[i, j] = np.linalg.norm(np.outer(x[i] - x[j], x[i] - x[j]), 'fro') ** 2
#    return Z_F2
def z(x):
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    Z_F2 = np.einsum('ijk,ijk->ij', diff, diff) ** 2
    return Z_F2

def z_r(x, R):
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    diff = np.einsum('ab,ijb,bc->ijc', R, diff, R)
    Z_F2 = np.einsum('ijk,ijk->ij', diff, diff) ** 2
    #Z_F2 = R @ Z_F2 @ R
    return Z_F2

@numba.njit(parallel=True)
def delta_Pi(x:np.ndarray, Coeft:np.ndarray):
    n = Coeft.shape[0]
    p = x.shape[-1]
    temp = np.zeros((p, p))
    for i in range(n):
        for j in range(n):
            temp += Coeft[i,j] * np.outer(x[i] - x[j], x[i] - x[j])        
    return temp /(2 * n ** 2)

@numba.njit(parallel=True)
def delta_PiH(x:np.ndarray, Coeft:np.ndarray):
    n, m = x.shape
    weighted_sum = np.zeros((x.shape[1], x.shape[1])) 

    for i in range(n):
        for j in range(i+1, n): 
            diff = x[i] - x[j]  
            outer_prod = np.outer(diff, diff)  
            weighted_sum += Coeft[i, j] * outer_prod

    return weighted_sum / (n * (n - 1))


def delta_PiL(x, Coeft):
    n = Coeft.shape[0]
    diffs = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    outer_prods = diffs[..., np.newaxis] * diffs[..., np.newaxis, :]
    weighted_sum = np.sum(Coeft[..., np.newaxis, np.newaxis] * outer_prods, axis=(0, 1))
    return weighted_sum / (2 * n ** 2)
    
def projL1(v, b):
    u = v
    sv = np.cumsum(u, axis=0)
    rho = np.maximum(u - (sv - b) / np.arange(1, len(u) + 1), np.zeros_like(sv))
    rho = np.nonzero(rho > 0)[0].max() + 1
    theta = (sv[rho - 1] - b) / rho
    w = np.maximum(u - theta, np.zeros_like(v))
    return w
    
def FantopeProjection(W):
    temp = (W + W.T)/2
        
    D, V = np.linalg.eigh(temp)
    d = np.flip(D, axis=0)
    V = np.flip(V, axis=1)
    d_final = projL1(d, 1)
        
    H = V @ np.diag(d_final) @ V.T
    return H

def calculate_mmse(y_time, y_event, y_pred):
    log_y_time = np.log(y_time) 
    log_y_pred = np.log(y_pred)
    mmse = np.sum(y_event * (log_y_time - log_y_pred) ** 2) 
    return mmse

def surv_grid_xgb(param_combinations, dmat):
    
    # document best result
    best_nloglik = float('inf')
    best_params = None
    
    # 5-Fold CV

    # walk through each parameter combination
    for params in param_combinations:
        #print(f"Testing parameters: {params}")
        param_dict = {
            'objective': 'survival:aft',
            'aft_loss_distribution': 'normal',  # 此处可换为 'logistic' 或其他分布
            'eval_metric': 'aft-nloglik',
            'learning_rate': params[0],
            'max_depth': params[1],
            'aft_loss_distribution_scale': params[2],
            'alpha': params[3],  # 设置 alpha
            "verbosity": 0
        }
        #dtrain = xgb.DMatrix(X_train, label_lower_bound=y_lower, label_upper_bound=y_upper)
        # 使用 xgboost.cv 进行交叉验证
        cv_results = xgb.cv(
            params=param_dict,
            dtrain=dmat,
            num_boost_round=100,
            nfold=5,
            metrics='aft-nloglik',
            early_stopping_rounds=10,
            verbose_eval=False
        )

        # 获取验证集的最小 aft-nloglik
        mean_nloglik = cv_results['test-aft-nloglik-mean'].min()
        #print(f"Params: {params}, Mean aft-nloglik: {mean_nloglik:.4f}")

        # 更新最优超参数
        if mean_nloglik < best_nloglik:
            best_nloglik = mean_nloglik
            best_params = params
            
    return best_params

def surv_grid(param_combinations, X, metric):
    
    # document best result
    best_score = float('inf')
    best_params = None
    
    # 5-Fold CV

    # walk through each parameter combination
    
        #print(f"Testing parameters: {params}")
        #param_dict = {
        #    'objective': 'survival:aft',
        #    'aft_loss_distribution': 'normal',  # 此处可换为 'logistic' 或其他分布
        #    'eval_metric': 'aft-nloglik',
        #    'learning_rate': params[0],
        #    'max_depth': params[1],
        #    'aft_loss_distribution_scale': params[2],
        #    'alpha': params[3],  # 设置 alpha
        #    "verbosity": 0
        #}
        #dtrain = xgb.DMatrix(X_train, label_lower_bound=y_lower, label_upper_bound=y_upper)
        
        # xgboost.cv 
        #cv_results = xgb.cv(
        #    params=param_dict,
        #    dtrain=dmat,
        #    num_boost_round=100,
        #    nfold=5,
        #    metrics='aft-nloglik',
        #    early_stopping_rounds=10,
        #    verbose_eval=False
        #)


    mse_scores = []
    for params in tqdm(param_combinations):

        
        X_train, X_test = train_test_split(X)
        #y_train, y_test = y[train_index], y[test_index]
        train_df = utils.organize_data(X_train,time="time_to_event",event="vital_status")
        test_df = utils.organize_data(X_test,time="time_to_event",event="vital_status")
        # Train the model
        result = dq.deep_quantreg(train_df,test_df,layer=3,node=300, learning_rate=params[0], metric=metric, acfn='relu',bsize=64,n_epoch=100,uncertainty=False) #, acfn=params[2], opt=params[3],bsize=params[4],n_epoch=params[5]
        eval = PointEvaluator(result.predQ, X_test['time_to_event'],X_test['vital_status'], X_train['time_to_event'],X_train['vital_status']) 
        # Evaluate
        #mse = mean_squared_error(y_test, y_pred)
        mae = eval.mae(method="Hinge")
        #mse_scores.append(mse)

        #mean_nloglik = cv_results['test-aft-nloglik-mean'].min()
        #mean_nloglik = np.mean(mse_scores)
        #print(f"Params: {params}, Mean aft-nloglik: {mean_nloglik:.4f}")

        # 更新最优超参数
        if mae < best_score:
            best_score = mae
            best_params = params
            
    return best_params

def survival_preprocess(datapath):
    data = pd.read_csv(datapath + 'clin_subtype1057.txt', sep='\t',header = 0)
    #covnames = ["age_at_initial_pathologic_diagnosis", "pathologic_stage", "Tumor_Grade", "BRCA_Pathology", "BRCA_Subtype_PAM50", 
                #"CESC_Pathology", "OV_Subtype", "UCS_Histology", "UCEC_Histology", "MSI_status", "HPV_Status", "tobacco_smoking_history",
    #            "race_indian","race_white","race_asian","race_black","race_na","ethnicity_hispanic","ethnicity_non_hispanic","ethnicity_na",
    #            "Chemotherapy","Hormone Therapy","No Therapy","Targeted Molecular therapy","Immunotherapy"]

    covnames = ['paper_age_at_initial_pathologic_diagnosis', 
                'paper_pathologic_stage', 'paper_BRCA_Pathology', 'paper_BRCA_Subtype_PAM50', 
                'race_white', 'race_asian', 'race_black', 'race_na', 'ethnicity_hispanic', 'ethnicity_non_hispanic', 'ethnicity_na', 
                'pharmaceutical_treatment_yes', 'pharmaceutical_treatment_no', 'pharmaceutical_treatment_na', 'radiation_treatment_yes', 'radiation_treatment_no', 'radiation_treatment_na']
    vitalstatus = data['vital_status']
    time_to_event = []
    for i in range(len(vitalstatus)):
        if vitalstatus[i] == 'Alive':
            time_to_event.append(data['days_to_last_follow_up'][i])
        else:
            time_to_event.append(data['days_to_death'][i])
    covariates = pd.concat([data[covnames], pd.DataFrame({'time_to_event': time_to_event, "vital_status": vitalstatus.map({'Alive': 0, 'Dead': 1})})], axis=1)
    covariates['time_to_event'][covariates['time_to_event'] <= 0] = 1
    
    # log transform
    #covariates['time_to_event'] = np.log10(covariates['time_to_event'])
    
    filter_data = covariates[covariates.columns[covariates.isnull().mean() != 1]]
    filter_data = filter_data.fillna("NA")
    #y = pd.read_csv(datapath + 'PAM50label664.txt',header = None)
    #filter_data['BRCA_Subtype_PAM50'] = y
    y = filter_data['paper_BRCA_Subtype_PAM50']
    # age
    filter_data['paper_age_at_initial_pathologic_diagnosis'] = (filter_data['paper_age_at_initial_pathologic_diagnosis'] - filter_data['paper_age_at_initial_pathologic_diagnosis'].mean()) / filter_data['paper_age_at_initial_pathologic_diagnosis'].std()

    dict_stage = {'Stage_I': 1, 'Stage_II': 2, 'Stage_III': 3, 'Stage_IV': 4} #, 'NA': 5
    filter_data['paper_pathologic_stage'] = filter_data['paper_pathologic_stage'].map(dict_stage)
    dummy_stage = pd.get_dummies(filter_data['paper_pathologic_stage'], prefix='paper_pathologic_stage')
    filter_data = pd.concat([filter_data, dummy_stage], axis=1)
    
    dict_path = {'Mixed':0,'IDC':1,'ILC':2,'Other':3, 'NA':4}
    filter_data['paper_BRCA_Pathology'] = filter_data['paper_BRCA_Pathology'].map(dict_path)
    dummy_BRCA = pd.get_dummies(filter_data['paper_BRCA_Pathology'], prefix='paper_BRCA_Pathology')
    filter_data = pd.concat([filter_data, dummy_BRCA], axis=1)
    
    dummy_Subtype = pd.get_dummies(filter_data['paper_BRCA_Subtype_PAM50'], prefix='paper_BRCA_Subtype_PAM50')
    filter_data = pd.concat([filter_data, dummy_Subtype], axis=1)
    
    filter_data = filter_data.drop(['paper_pathologic_stage', 'paper_BRCA_Pathology', 'paper_BRCA_Subtype_PAM50'], axis=1)

    return filter_data, covariates['vital_status']

def res_cov(datapath, method=None):
        Labels = ['mRNA_expression_standardized.xlsx', 'DNA_methylation_standardized.xlsx', 'microRNA_expression_standardized.xlsx']
        #Datas = ['Exp664.txt', 'Meth664.txt', 'miRNA664.txt']
        Scores = ['Exp_score.csv', 'Meth_score.csv', 'miRNA_score.csv']

        res = []
        for i in range(3):
            Labelpath = datapath + Labels[i]
            Exp_label = pd.read_excel(Labelpath)
            Exp_list = Exp_label.iloc[:, 0].values.tolist()
            Exp = Exp_label.iloc[:,1:]
            #Datapath = datapath + Datas[i]
            if method is not None:
                respath = datapath.rsplit('/',2)[0] + '/' + method
                Respath = respath + Scores[i]

                #X = Exp
                ExpRes = pd.read_csv(Respath).values
                listname = ExpRes[:,0]
                FilterRes = []
                for i in range(len(listname)):
                    list_index: int = Exp_list.index(listname[i])
                    FilterRes.append(list_index)

                ExpFilter = Exp.T.iloc[:,FilterRes].values
            else:
                ExpFilter = Exp.T.values
            res.append(ExpFilter)
    
        return np.concatenate((res[0], res[1], res[2]), axis=1)

def _get_tcga0(root):
    Exp_value = np.loadtxt(root+"/SNGCCA/RealData/Exp664.txt")
    Meth_value = np.loadtxt(root+"/SNGCCA/RealData/Meth664.txt")
    miRNA_value = np.loadtxt(root+"/SNGCCA/RealData/miRNA664.txt")
    views = [Meth_value.T,
             Exp_value.T,
             miRNA_value.T
             ]
    # stadardize
    #for i, view in enumerate(views):
    #    views[i] = (view - np.mean(view, axis=0)) / np.std(view, axis=0)
    return views

def _get_tcga(root):
    Exp_value = np.loadtxt(root+"/SNGCCA/RealData/newData/Exp1057.txt")
    Meth_value = np.loadtxt(root+"/SNGCCA/RealData/newData/Meth1057.txt")
    miRNA_value = np.loadtxt(root+"/SNGCCA/RealData/newData/miRNA1057.txt")
    views = [Meth_value,
             Exp_value,
             miRNA_value
             ]
    # stadardize
    #for i, view in enumerate(views):
    #    views[i] = (view - np.mean(view, axis=0)) / np.std(view, axis=0)
    return views

def _get_tcga_new(root, label = False):
    Exp = pd.read_excel(root+'/SNGCCA/RealData/newData/mRNA_expression_standardized.xlsx',sheet_name="Sheet 1")
    Meth = pd.read_excel(root+'/SNGCCA/RealData/newData/DNA_methylation_standardized.xlsx',sheet_name="Sheet 1")
    miRNA = pd.read_excel(root+'/SNGCCA/RealData/newData/microRNA_expression_standardized.xlsx',sheet_name="Sheet 1")

    Exp_data = Exp.values[:,1:].astype(float)
    Meth_data = Meth.values[:,1:].astype(float)
    miRNA_data = miRNA.values[:,1:].astype(float)
    views = [Exp_data.T,Meth_data.T,miRNA_data.T]
    
    if label == True:
        Exp_label = list(Exp.iloc[:,0])
        Meth_label = list(Meth.iloc[:,0])
        miRNA_label = list(miRNA.iloc[:,0])
        return views, Exp_label, Meth_label, miRNA_label
    else:
        return views
    #y = pd.read_csv(root+'/SNGCCA/RealData/PAM50label664.txt',header = None)
    
    