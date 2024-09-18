import torch
import numpy as np

from validation_method import eval, eval_topk

num = 5
tol = 100
N = 100

print('100*100*5/Linear')
path = 'E:/GitHub/SNGCCA/SNGCCA/Simulation/' + 'Linear/' + str(N) + '_' + str(tol) + '_' + str(num) + '/' #+ 'rgcca_'
u1 = np.genfromtxt(path + 'u1.csv', delimiter=',')#[1:,:]
u2 = np.genfromtxt(path + 'u2.csv', delimiter=',')#[1:,:]
u3 = np.genfromtxt(path + 'u3.csv', delimiter=',')#[1:,:]
Label = torch.cat([torch.ones(num, dtype=torch.bool), torch.zeros(tol-num, dtype=torch.bool)])

FS = []
MCC = []
PRE = []
REC = []
SPE = []
SR = []
for i in range(100):
    u = [torch.tensor(u1[i]),torch.tensor(u2[i]),torch.tensor(u3[i])]
    spe, pre, rec, acc, f1, mcc, success = eval(u, Label, num)
    #spe, pre, rec, acc, f1, mcc = eval_topk(u, Label, num)
    SPE.append(spe)
    PRE.append(pre)
    REC.append(rec)
    FS.append(f1)
    MCC.append(mcc)
    SR.append(success)
    

print("Specificity:",np.mean([spe[0] for spe in SPE]),np.std([spe[0] for spe in SPE])/10,np.mean([spe[1] for spe in SPE]),np.std([spe[1] for spe in SPE])/10,np.mean([spe[2] for spe in SPE]),np.std([spe[2] for spe in SPE])/10)
print("Precision:",np.mean([spe[0] for spe in PRE]),np.std([spe[0] for spe in PRE])/10,np.mean([spe[1] for spe in PRE]),np.std([spe[1] for spe in PRE])/10,np.mean([spe[2] for spe in PRE]),np.std([spe[2] for spe in PRE])/10)
print("Recall:",np.mean([spe[0] for spe in REC]),np.std([spe[0] for spe in REC])/10,np.mean([spe[1] for spe in REC]),np.std([spe[1] for spe in REC])/10,np.mean([spe[2] for spe in REC]),np.std([spe[2] for spe in REC])/10)
print("FS:",np.mean([spe[0] for spe in FS]),np.std([spe[0] for spe in FS])/10,np.mean([spe[1] for spe in FS]),np.std([spe[1] for spe in FS])/10,np.mean([spe[2] for spe in FS]),np.std([spe[2] for spe in FS])/10)
print("MCC:",np.mean([spe[0] for spe in MCC]),np.std([spe[0] for spe in MCC])/10,np.mean([spe[1] for spe in MCC]),np.std([spe[1] for spe in MCC])/10,np.mean([spe[2] for spe in MCC]),np.std([spe[2] for spe in MCC])/10)
print("Success Rate:", sum([s[0] for s in SR])/100, sum([s[1] for s in SR])/100, sum([s[2] for s in SR])/100)

print('100*100*5/NonLinear')
path = 'E:/GitHub/SNGCCA/SNGCCA/Simulation/' + 'Nonlinear/' + str(N) + '_' + str(tol) + '_' + str(num) + '/'# + 'rgcca_'
u1 = np.genfromtxt(path + 'u1.csv', delimiter=',')#[1:,:]
u2 = np.genfromtxt(path + 'u2.csv', delimiter=',')#[1:,:]
u3 = np.genfromtxt(path + 'u3.csv', delimiter=',')#[1:,:]

FS = []
MCC = []
PRE = []
REC = []
SPE = []
SR = []
for i in range(100):
    u = [torch.tensor(u1[i]),torch.tensor(u2[i]),torch.tensor(u3[i])]
    spe, pre, rec, acc, f1, mcc, success = eval(u, Label, num)
    #spe, pre, rec, acc, f1, mcc = eval_topk(u, Label,num)
    print(success)
    SPE.append(spe)
    PRE.append(pre)
    REC.append(rec)
    FS.append(f1)
    MCC.append(mcc)
    SR.append(success)

print("Specificity:",np.mean([spe[0] for spe in SPE]),np.std([spe[0] for spe in SPE])/10,np.mean([spe[1] for spe in SPE]),np.std([spe[1] for spe in SPE])/10,np.mean([spe[2] for spe in SPE]),np.std([spe[2] for spe in SPE])/10)
print("Precision:",np.mean([spe[0] for spe in PRE]),np.std([spe[0] for spe in PRE])/10,np.mean([spe[1] for spe in PRE]),np.std([spe[1] for spe in PRE])/10,np.mean([spe[2] for spe in PRE]),np.std([spe[2] for spe in PRE])/10)
print("Recall:",np.mean([spe[0] for spe in REC]),np.std([spe[0] for spe in REC])/10,np.mean([spe[1] for spe in REC]),np.std([spe[1] for spe in REC])/10,np.mean([spe[2] for spe in REC]),np.std([spe[2] for spe in REC])/10)
print("FS:",np.mean([spe[0] for spe in FS]),np.std([spe[0] for spe in FS])/10,np.mean([spe[1] for spe in FS]),np.std([spe[1] for spe in FS])/10,np.mean([spe[2] for spe in FS]),np.std([spe[2] for spe in FS])/10)
print("MCC:",np.mean([spe[0] for spe in MCC]),np.std([spe[0] for spe in MCC])/10,np.mean([spe[1] for spe in MCC]),np.std([spe[1] for spe in MCC])/10,np.mean([spe[2] for spe in MCC]),np.std([spe[2] for spe in MCC])/10)
print("Success Rate:", sum([s[0] for s in SR])/100, sum([s[1] for s in SR])/100, sum([s[2] for s in SR])/100)
