from main import Solver
import torch

import numpy as np

# Hyper Params Section
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")



Exp = np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SGCCA_HSIC/RealData/Exp664.txt")
Meth = np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SGCCA_HSIC/RealData/Meth664.txt")
miRNA = np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SGCCA_HSIC/RealData/miRNA664.txt")

views = [Exp.T,Meth.T,miRNA.T]
views = [torch.tensor(view).to(device) for view in views]

## Analysis
Solver = Solver()
print(f'input views shape :')
for i, view in enumerate(views):
    print(f'view_{i} :  {view.shape}')
    view = view.to("cpu")

u = Solver._get_outputs(views, 1e-7, 100, (1,1,1))


