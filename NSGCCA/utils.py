#import torch
import numba
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from sksurv.metrics import concordance_index_censored
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print("Using", torch.cuda.device_count(), "GPUs")
device = 'cpu'
if device == 'cuda':
    import cupy as cp
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
                
    #Kx = torch.exp(- (Kx ** 2) / 2)
#    return np.exp(- (Kx ** 2) / 2)

def is_invertible(matrix):
  try:
    np.linalg.inv(matrix)
    return True
  except np.linalg.LinAlgError:
    return False

def rbf_kx(x:np.ndarray, Pi:np.ndarray, sigma=None):
    # Calculate pairwise differences
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    Kx = np.einsum('ijk,kl,ijl->ij', diff, Pi, diff)
    Kx = np.exp(- Kx / 2)
    return Kx

def rbf_kx_cp(x, Pi, sigma=None):
    diff = x[:, cp.newaxis, :] - x[cp.newaxis, :, :]
    Kx = cp.einsum('ijk,kl,ijl->ij', diff, Pi, diff)
    Kx = cp.exp(- (Kx ** 2) / 2)
    return cp.exp(- (Kx ** 2) / 2)
    
def rbf_kl(sum_K):
    n = len(sum_K)
    I_n = np.eye(n)
    H = I_n - np.outer(np.ones(n), np.ones(n)) / n
    return H @ sum_K @ H

def rbf_kl_cp(sum_K):
    n = len(sum_K)
    I_n = cp.eye(n)
    H = I_n - cp.outer(cp.ones(n), cp.ones(n)) / n
    return H @ sum_K @ H
    
#def z(x, p):
#    n = x.shape[0]
#    Z_F2 = np.zeros((n, n))
#    for i in range(n):
#        for j in range(n):
#            Z_F2[i, j] = np.linalg.norm(np.outer(x[i] - x[j], x[i] - x[j]), 'fro') ** 2
#    return Z_F2
def z(x, p):
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    Z_F2 = np.einsum('ijk,ijk->ij', diff, diff) ** 2
    return Z_F2

def z_cp(x, p):
    diff = x[:, cp.newaxis, :] - x[cp.newaxis, :, :]
    Z_F2 = cp.einsum('ijk,ijk->ij', diff, diff) ** 2
        
    return Z_F2
    
@numba.njit(parallel=True)
def delta_Pi(x, Coeft):
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

def delta_Pi_cp(x, Coeft):
    n = Coeft.shape[0]
    p = x.shape[-1]
    temp = cp.zeros((p, p))
    for i in range(n):
        for j in range(n):
            temp += Coeft[i,j] * cp.outer(x[i] - x[j], x[i] - x[j])
        
    return temp /(2 * n ** 2)
    
def rbf_kernel(X, sigma=None):
    # dist
    D = np.sqrt(np.abs(sqdist(X.t(), X.t())))

    if sigma is None:
        # median sigma
        sigma = torch.median(D)

    # kernel
    K = torch.exp(- (D ** 2) / (2 * sigma ** 2))
    return K, sigma

def centre_kernel(K):
    return K + torch.mean(K) - (torch.mean(K, dim=0).reshape((1, -1)) + torch.mean(K, dim=1).reshape((-1, 1)))
    
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

def projL1_cp(v, b):
    u = v
    sv = cp.cumsum(u, axis=0)
    rho = cp.maximum(u - (sv - b) / cp.arange(1, len(u) + 1), cp.zeros_like(sv))
    rho = cp.nonzero(rho > 0)[0].max() + 1
    theta = (sv[rho - 1] - b) / rho
    w = cp.maximum(u - theta, cp.zeros_like(v))
    return w

def FP_cp(W):
    temp = (W + W.T)/2
        
    D, V = cp.linalg.eigh(temp)
    d = cp.flip(D, axis=0)
    V = cp.flip(V, axis=1)
    d_final = projL1_cp(d, 1)
        
    H = V @ cp.diag(d_final) @ V.T
    return H

def calculate_mmse(y_time, y_event, y_pred):
    """
    计算 MMSE
    :param y_time: true time (array)
    :param y_event: event status (1=censored, 0=uncensored) (array)
    :param y_pred: predict time (array)
    :return: MMSE
    """
    log_y_time = np.log(y_time) 
    log_y_pred = np.log(y_pred)
    mmse = np.sum(y_event * (log_y_time - log_y_pred) ** 2) 
    return mmse

def surv_grid(param_combinations, dmat):
    
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
        print(f"Params: {params}, Mean aft-nloglik: {mean_nloglik:.4f}")

        # 更新最优超参数
        if mean_nloglik < best_nloglik:
            best_nloglik = mean_nloglik
            best_params = params
            
    return best_params

def survival_preprocess(datapath):
    data = pd.read_csv(datapath + 'subtype664.txt', sep=' ',header = 0)
    covnames = ["age_at_initial_pathologic_diagnosis", "pathologic_stage", "Tumor_Grade", "BRCA_Pathology", "BRCA_Subtype_PAM50", 
                "CESC_Pathology", "OV_Subtype", "UCS_Histology", "UCEC_Histology", "MSI_status", "HPV_Status", "tobacco_smoking_history"]

    vitalstatus = data['vital_status']
    time_to_event = []
    for i in range(len(vitalstatus)):
        if vitalstatus[i] == 'Alive':
            time_to_event.append(data['days_to_last_followup'][i])
        else:
            time_to_event.append(data['days_to_death'][i])
    covariates = pd.concat([data[covnames], pd.DataFrame({'time_to_event': time_to_event, "vital_status": vitalstatus.map({'Alive': 0, 'Dead': 1})})], axis=1)
    filter_data = covariates[covariates.columns[covariates.isnull().mean() != 1]]

    y = pd.read_csv(datapath + 'PAM50label664.txt',header = None)
    filter_data['BRCA_Subtype_PAM50'] = y

    dict_stage = {'Stage_I': 1, 'Stage_II': 2, 'Stage_III': 3, 'Stage_IV': 4}
    filter_data['pathologic_stage'] = filter_data['pathologic_stage'].map(dict_stage)

    dict_path = {'Mixed':0,'IDC':1,'ILC':2,'Other':3}
    filter_data['BRCA_Pathology'] = filter_data['BRCA_Pathology'].map(dict_path)

    return filter_data, y

def res_cov(datapath, method=None):
    if method is None:
        return None
    else:
        respath = datapath + method

        Labels = ['Exp664_genes.txt', 'Meth664_probes.txt', 'miRNA664_miRNA.txt']
        Datas = ['Exp664.txt', 'Meth664.txt', 'miRNA664.txt']
        Scores = ['Exp_score.csv', 'Meth_score.csv', 'miRNA_score.csv']

        res = []
        for i in range(3):
            Labelpath = datapath + Labels[i]
            Datapath = datapath + Datas[i]
            Respath = respath + Scores[i]
            Exp_label = pd.read_csv(Labelpath, sep='\t',header = None)
            Exp_list = Exp_label.iloc[:, 0].values.tolist()
            Exp = pd.DataFrame(np.loadtxt(Datapath).T,columns = Exp_label)

            #X = Exp
            ExpRes = pd.read_csv(Respath).values
            listname = ExpRes[:,0]
            FilterRes = []
            for i in range(len(listname)):
                list_index: int = Exp_list.index(listname[i])
                FilterRes.append(list_index)

            ExpFilter = Exp.iloc[:,FilterRes]
            res.append(ExpFilter)
    
    return pd.concat(res, axis=1)

def _get_tcga(root):
    Exp_value = np.loadtxt(root+"/SNGCCA/RealData/Exp664.txt")
    Meth_value = np.loadtxt(root+"/SNGCCA/RealData/Meth664.txt")
    miRNA_value = np.loadtxt(root+"/SNGCCA/RealData/miRNA664.txt")
    views = [Meth_value.T,
             Exp_value.T,
             miRNA_value.T
             ]
    # stadardize
    for i, view in enumerate(views):
        views[i] = (view - np.mean(view, axis=0)) / np.std(view, axis=0)
    return views

def _get_tcga_new(root):
    Exp = pd.read_excel(root+'/SNGCCA/RealData/newData/mRNA_expression_standardized.xlsx',sheet_name="Sheet 1").values[:,1:].astype(float)
    Meth = pd.read_excel(root+'/SNGCCA/RealData/newData/DNA_methylation_standardized.xlsx',sheet_name="Sheet 1").values[:,1:].astype(float)
    miRNA = pd.read_excel(root+'/SNGCCA/RealData/newData/microRNA_expression_standardized.xlsx',sheet_name="Sheet 1").values[:,1:].astype(float)

    #y = pd.read_csv(root+'/SNGCCA/RealData/PAM50label664.txt',header = None)
    views = [Exp.T,Meth.T,miRNA.T]
    return views