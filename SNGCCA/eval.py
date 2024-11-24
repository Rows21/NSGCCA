import os
import numpy as np
import random
from validation_method import eval, eval_plot
import pandas as pd
from tqdm import tqdm

def mean_std(data):
    return np.mean([np.mean([spe[0],spe[1],spe[2]]) for spe in data]), np.std([np.mean([spe[0],spe[1],spe[2]]) for spe in data])

def mean_std2(data):
    return np.mean(data), np.std(data)/10 * 1.96

def eval_linear(root, method, N, tol, num, scen, plot = False):
    print(N, tol, num, scen)
    #matrix = np.zeros((11, 6))
    Label = np.concatenate([np.ones(num, dtype=bool), np.zeros(tol - num, dtype=bool)])
    path = root + scen + '/' + str(N) + '_' + str(tol) + '_' + str(num) + '/' + method
    
    if method == 'sgcca_' or method == 'rgcca_':
        u1 = np.genfromtxt(path + 'u1.csv', delimiter=',')[1:,:]
        u2 = np.genfromtxt(path + 'u2.csv', delimiter=',')[1:,:]
        u3 = np.genfromtxt(path + 'u3.csv', delimiter=',')[1:,:]
        t = np.genfromtxt(path + 't.csv', delimiter=',')[1:]
    else:
        u1 = np.genfromtxt(path + 'u1.csv', delimiter=',')
        u2 = np.genfromtxt(path + 'u2.csv', delimiter=',')
        u3 = np.genfromtxt(path + 'u3.csv', delimiter=',')
        
        if method == 'K_':
            t = np.genfromtxt(path.replace('K_','t_k.csv'), delimiter=',')
        else:
            t = np.genfromtxt(path + 't.csv', delimiter=',')
    FS = []
    MCC = []
    PRE = []
    REC = []
    SPE = []
    SR = []
    for i in range(100):
        if method == 'K_':
            u = [np.flip(u1[i])/np.linalg.norm(u1[i]),np.flip(u2[i])/np.linalg.norm(u2[i]),u3[i]/np.linalg.norm(u3[i])]
        else:
            u = [u1[i]/np.linalg.norm(u1[i]),u2[i]/np.linalg.norm(u2[i]),u3[i]/np.linalg.norm(u3[i])]
        
        spe, pre, rec, f1, mcc, success = eval(u, Label, num)
        SPE.append(spe)
        PRE.append(pre)
        REC.append(rec)
        FS.append(f1)
        MCC.append(mcc)
        SR.append(success)
        #if num == 20:
            #print(mcc)
    if not plot:                
        print("Specificity:",np.mean([spe[0] for spe in SPE]),np.std([spe[0] for spe in SPE])/10,np.mean([spe[1] for spe in SPE]),np.std([spe[1] for spe in SPE])/10,np.mean([spe[2] for spe in SPE]),np.std([spe[2] for spe in SPE])/10)
        print("Precision:",np.mean([spe[0] for spe in PRE]),np.std([spe[0] for spe in PRE])/10,np.mean([spe[1] for spe in PRE]),np.std([spe[1] for spe in PRE])/10,np.mean([spe[2] for spe in PRE]),np.std([spe[2] for spe in PRE])/10)
        print("Recall:",np.mean([spe[0] for spe in REC]),np.std([spe[0] for spe in REC])/10,np.mean([spe[1] for spe in REC]),np.std([spe[1] for spe in REC])/10,np.mean([spe[2] for spe in REC]),np.std([spe[2] for spe in REC])/10)
        print("FS:",np.mean([spe[0] for spe in FS]),np.std([spe[0] for spe in FS])/10,np.mean([spe[1] for spe in FS]),np.std([spe[1] for spe in FS])/10,np.mean([spe[2] for spe in FS]),np.std([spe[2] for spe in FS])/10)
        print("MCC:",np.mean([spe[0] for spe in MCC]),np.std([spe[0] for spe in MCC])/10,np.mean([spe[1] for spe in MCC]),np.std([spe[1] for spe in MCC])/10,np.mean([spe[2] for spe in MCC]),np.std([spe[2] for spe in MCC])/10)
        print("Success Rate:", sum([s[0] for s in SR])/100, sum([s[1] for s in SR])/100, sum([s[2] for s in SR])/100)
                
        return mean_std(SPE), mean_std(PRE), mean_std(REC), mean_std(FS), mean_std(MCC), mean_std(SR), mean_std2(t)
    else:
        return mean_std2(SPE), mean_std2(PRE), mean_std2(REC), mean_std2(FS), mean_std2(MCC), mean_std2(SR), mean_std2(t)

def combine(root, combinations, method, linear=True, plot = False):
    if linear:
        scen = "Linear"
    else:
        scen = "Nonlinear"
        
    if method == 'KSSHIBA':
        prefix = 'K_'
    elif method == 'DGCCA':
        prefix = 'dgcca_'
    elif method == 'SNGCCA':
        prefix = ''
    elif method == 'RGCCA':
        prefix = 'rgcca_'
    elif method == 'SGCCA':
        prefix = 'sgcca_'
        
    df = pd.DataFrame()
    for i in combinations:
        #print(i)
        
        temp = pd.DataFrame()
        spe, pre, rec, f1, mcc, success, t = eval_linear(root, prefix, i[0], i[1], i[2], scen, plot = plot)
        temp = pd.DataFrame([[method, scen, i[0], i[1], i[2], 'Specificity', spe[0], spe[1]], 
                             [method, scen, i[0], i[1], i[2], 'Precision', pre[0], pre[1]], 
                             [method, scen, i[0], i[1], i[2], 'Recall', rec[0], rec[1]], 
                             [method, scen, i[0], i[1], i[2], 'F1-Score', f1[0], f1[1]], 
                             [method, scen, i[0], i[1], i[2], 'MCC', mcc[0], mcc[1]],
                             [method, scen, i[0], i[1], i[2], 'SR', success[0], success[1]],
                             [method, scen, i[0], i[1], i[2], 'Time', t[0], t[1]]], columns=['Methods', 'Scenario', 'n', 'p', 's', 'type', 'Values', 'std'])#, ['KSSHIBA', 'Linear', i[0], i[1], 'SR', success[0]]
        df = pd.concat([df, temp], axis=0)
    return df

def build_data(root, method, plot=False):
    print(method)
    df_list = pd.DataFrame()
    combinations = [[
        #[100, 20, 5],
        [100, 30, 5],
        [100, 50, 5],
        #[100, 80, 5],
        [100, 100, 5],
        #[100, 150, 5],
        [100, 200, 5]
    ],
    [
        [100, 100, 5],
        [200, 100, 5],
        [400, 100, 5]
    ],
    [
        [100, 100, 5],
        [100, 100, 10],
        [100, 100, 20]
    ]]
    for i in range(len(combinations)):
        df = pd.concat([combine(root, combinations[i], method, linear=True, plot = plot), combine(root, combinations[i], method, linear=False, plot = plot)], axis=0)
        df_list = pd.concat([df_list, df], axis=0)
    if plot:
        df_list.to_csv(os.path.join(root, method + '.csv'), index=False)
        
if __name__ == '__main__':
    root = 'E:/GitHub/SNGCCA/SNGCCA/Simulation/'
    #build_data(root, 'SNGCCA', plot = True)
    #build_data(root, 'DGCCA', plot = True)
    build_data(root, 'RGCCA', plot = True)
    build_data(root, 'SGCCA', plot = True)
    #build_data(root, 'KSSHIBA', plot = True)
