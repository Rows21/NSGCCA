from main import Solver
import pandas as pd
import torch
from synth_data import create_synthData_new
from validation_method import FS_MCC
import numpy as np
from sgcca_hsic_adam import SNGCCA_ADAM
torch.set_default_tensor_type(torch.DoubleTensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")

SNGCCA_ADAM = SNGCCA_ADAM(device)
x = pd.read_csv("x.csv")
y = pd.read_csv("y.csv")
z = pd.read_csv("z.csv")

v = torch.arange(0, 1, 0.05)
umr = SNGCCA_ADAM.projL1(v,3)
n_view = len(x)
#ind = torch.randperm(n_view)[:n_view//10]
ind = np.arange(0, 30)

Xu = torch.tensor(x @ umr).reshape(-1,1)
Yu = y @ umr
Zu = z @ umr

phiu,a = SNGCCA_ADAM.rbf_approx(Xu,ind)
K = phiu.t() @ phiu
phic = SNGCCA_ADAM.centre_nystrom_kernel(phiu)
cK = phic.t() @ phic

aaa = torch.tensor(x.iloc[ind,:].values)

#grad = SNGCCA_ADAM.gradf_gauss_SGD(K, cK, aaa, a, umr)
N = 400
views = create_synthData_new(N, mode=1, F=20)

print(f'input views shape :')
for i, view in enumerate(views):
    print(f'view_{i} :  {view.shape}')
    view = view.to("cpu")

phic = SNGCCA_ADAM.set_init(views,ind,(1,1,1))

a = SNGCCA_ADAM.fit(views, eps=1e-5, maxit=10, b=(1,1,1),early_stopping=True, patience=10, logging=1)
print(a)