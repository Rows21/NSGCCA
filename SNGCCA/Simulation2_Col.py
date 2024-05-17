import torch
import numpy as np
from validation_method import eval

num = 20
sample = 100
tol = 100
root = 'D:/GitHub/SNGCCA/SNGCCA/Simulation/'
dir_sce = 'Linear/' + str(sample) + '_' + str(tol) + '_' + str(num) + '/'
print(dir_sce)
path = root + dir_sce
u1 = np.genfromtxt(path + 'u1.csv', delimiter=',')
u2 = np.genfromtxt(path + 'u2.csv', delimiter=',')
u3 = np.genfromtxt(path + 'u3.csv', delimiter=',')
Label = torch.cat([torch.ones(num, dtype=torch.bool), torch.zeros(tol-num, dtype=torch.bool)])

FS = []
MCC = []
PRE = []
REC = []
SPE = []
for i in range(100):
    u = [torch.tensor(u1[i]),torch.tensor(u2[i]),torch.tensor(u3[i])]
    spe, pre, rec, acc, f1, mcc = eval(u, Label)
    SPE.append(spe)
    PRE.append(pre)
    REC.append(rec)
    FS.append(f1)
    MCC.append(mcc)

print("Specificity:",np.mean(SPE),np.std(SPE))
print("Precision:",np.mean(PRE),np.std(PRE))
print("Recall:",np.mean(REC),np.std(REC))
print("FS:",np.mean(FS),np.std(FS))
print("MCC:",np.mean(MCC),np.std(MCC))

dir_sce = 'Nonlinear/' + str(sample) + '_' + str(tol) + '_' + str(num) + '/'
print(dir_sce)
path = root + dir_sce
u1 = np.genfromtxt(path + 'u1.csv', delimiter=',')
u2 = np.genfromtxt(path + 'u2.csv', delimiter=',')
u3 = np.genfromtxt(path + 'u3.csv', delimiter=',')
Label = torch.cat([torch.ones(num, dtype=torch.bool), torch.zeros(tol-num, dtype=torch.bool)])

FS = []
MCC = []
PRE = []
REC = []
SPE = []
for i in range(100):
    u = [torch.tensor(u1[i]),torch.tensor(u2[i]),torch.tensor(u3[i])]
    spe, pre, rec, acc, f1, mcc = eval(u, Label)
    SPE.append(spe)
    PRE.append(pre)
    REC.append(rec)
    FS.append(f1)
    MCC.append(mcc)

print("Specificity:",np.mean(SPE),np.std(SPE))
print("Precision:",np.mean(PRE),np.std(PRE))
print("Recall:",np.mean(REC),np.std(REC))
print("FS:",np.mean(FS),np.std(FS))
print("MCC:",np.mean(MCC),np.std(MCC))
