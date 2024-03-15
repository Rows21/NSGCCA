import torch
import numpy as np

from validation_method import eval
num = 10
tol = 100
path = 'C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/Simulation/Linear/100_100_20/'


u1 = np.genfromtxt(path + 'u1.csv', delimiter=',')
u2 = np.genfromtxt(path + 'u2.csv', delimiter=',')
u3 = np.genfromtxt(path + 'u3.csv', delimiter=',')
Label = torch.cat([torch.ones(num, dtype=torch.bool), torch.zeros(tol-num, dtype=torch.bool)])

FS = []
MCC = []
PRE = []
REC = []
for i in range(50):
    u = [torch.tensor(u1[i]),torch.tensor(u2[i]),torch.tensor(u3[i])]
    pre, rec, acc, f1, mcc = eval(u, Label)
    PRE.append(pre)
    REC.append(rec)
    FS.append(f1)
    MCC.append(mcc)

print("Precision:",np.mean(PRE),np.std(PRE))
print("Recall:",np.mean(REC),np.std(REC))
print("FS:",np.mean(FS),np.std(FS))
print("MCC:",np.mean(MCC),np.std(MCC))