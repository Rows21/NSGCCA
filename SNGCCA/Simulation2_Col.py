import torch
import numpy as np
from validation_method import eval, eval_topk
import pandas as pd

def res(num_total, sample_total, tol_total):
    df = None
    methods = ['rgcca_', 'sgcca_', 'dgcca_', 'K_', '']
    iflinear = ['Linear', 'Nonlinear']
    root = 'D:/GitHub/SNGCCA/SNGCCA/Simulation/'
    for num in num_total:
        for sample in sample_total:
            for tol in tol_total:
                for linear in iflinear:
                    dir_sce = linear + '/' + str(sample) + '_' + str(tol) + '_' + str(num) + '/'
                    print(dir_sce)
                    path = root + dir_sce
                    for method in methods:
                        
                        if method == 'dgcca_' and linear == 'Nonlinear':
                            a=1
                        Label = torch.cat([torch.ones(num, dtype=torch.bool), torch.zeros(tol-num, dtype=torch.bool)])

                        # load data
                        u1 = np.genfromtxt(path + method + 'u1.csv', delimiter=',')
                        u2 = np.genfromtxt(path + method + 'u2.csv', delimiter=',')
                        u3 = np.genfromtxt(path + method + 'u3.csv', delimiter=',')

                        FS = []
                        MCC = []
                        PRE = []
                        REC = []
                        SPE = []
                        for i in range(100):
                            if method == 'rgcca_' or method == 'sgcca_':
                                j = i+1
                            else:
                                j=i

                            u = [torch.tensor(u1[j]),torch.tensor(u2[j]),torch.tensor(u3[j])]
                            u = [u[i]/torch.norm(u[i]) for i in range(3)]
                            
                            spe, pre, rec, acc, f1, mcc = eval(u, Label)
                            spe, pre, rec, acc, f1, mcc = eval_topk(u, Label, num)
                            SPE.append(spe)
                            PRE.append(pre)
                            REC.append(rec)
                            FS.append(f1)
                            MCC.append(mcc)

                        if len(tol_total) > 1:
                            param = tol
                        elif len(sample_total) > 1:
                            param = sample
                        elif len(num_total) > 1:
                            param = num
                        if method == '':
                            methname = 'SNGCCA'
                        elif method == 'K_':
                            methname = 'KSSHIBA'
                        else:
                            methname = method.replace("_","").upper()
                        new_spe = [methname, linear,param,'Specificity',np.mean(SPE),np.std(SPE)]
                        if df is None:
                            df = pd.DataFrame([new_spe], columns=['Methods', 'Scenario', 'n', 'type', 'Values', 'std'])
                        else:
                            df.loc[len(df)] = new_spe
                        
                        df.loc[len(df)] = [methname, linear,param,'Precision',np.mean(PRE),np.std(PRE)]
                        df.loc[len(df)] = [methname, linear,param,'Recall',np.mean(REC),np.std(REC)]
                        df.loc[len(df)] = [methname, linear,param,'FS',np.mean(FS),np.std(FS)]
                        df.loc[len(df)] = [methname, linear,param,'MCC',np.mean(MCC),np.std(MCC)]

    return df

if __name__ == '__main__':
    # 
    num_total = [5]
    sample_total = [100]
    tol_total = [30,100]
    df = res(num_total, sample_total, tol_total)
    df.to_csv('D:/GitHub/SNGCCA/SNGCCA/Simulation/simu1_p.csv')

    num_total = [5]
    sample_total = [100,200,400]
    tol_total = [100]
    df = res(num_total, sample_total, tol_total)
    df.to_csv('D:/GitHub/SNGCCA/SNGCCA/Simulation/simu1_s.csv')

    num_total = [5,10,20]
    sample_total = [100]
    tol_total = [100]
    df = res(num_total, sample_total, tol_total)
    df.to_csv('D:/GitHub/SNGCCA/SNGCCA/Simulation/simu1_v.csv')