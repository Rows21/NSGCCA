import torch
import numpy as np
import pandas as pd

# Hyper Params Section
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")

# Result Data
ExpRes = pd.read_csv("./Results/Exp_score.csv").values
MethRes = pd.read_csv("./Results/Meth_score.csv").values
miRNARes = pd.read_csv("./Results/miRNA_score.csv").values

# Original Data
Exp_label = pd.read_csv('RealData/Exp664_genes.txt', sep='\t',header = None)
Exp_list = Exp_label.iloc[:, 0].values.tolist()
Exp = pd.DataFrame(np.loadtxt("RealData/Exp664.txt").T,columns = Exp_label)

Meth_label = pd.read_csv('RealData/Meth664_probes.txt', sep='\t',header = None)
Meth_list = Meth_label.iloc[:, 0].values.tolist()
Meth = pd.DataFrame(np.loadtxt("RealData/Meth664.txt").T,columns = Meth_label)

miRNA_label = pd.read_csv('RealData/miRNA664_miRNA.txt', sep='\t',header = None)
miRNA_list = miRNA_label.iloc[:, 0].values.tolist()
miRNA = pd.DataFrame(np.loadtxt("RealData/miRNA664.txt").T,columns = miRNA_label)

# labels
y = pd.read_csv('RealData/PAM50label664.txt',header = None).values

## K-Mean
a = 1
## SVM


