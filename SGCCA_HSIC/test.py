from main import Solver
import pandas as pd
import torch
from synth_data import create_synthData_new
from validation_method import FS_MCC
import numpy as np
from sgcca_hsic_adam import SNGCCA_ADAM
import scipy
from sgcca_hsic import SGCCA_HSIC
torch.set_default_tensor_type(torch.DoubleTensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")

SNGCCA_ADAM = SNGCCA_ADAM(device)

x = pd.read_csv("x.csv").values
y = pd.read_csv("y.csv").values
z = pd.read_csv("z.csv").values

views = [torch.tensor(x),torch.tensor(y)]
a = SNGCCA_ADAM.fit(views, eps=1e-5, maxit=10, b=(2,2,2),early_stopping=True, patience=10, logging=1)


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

grad = SNGCCA_ADAM.gradf_gauss_SGD(K, cK, aaa, a, umr)
'''
N = 400
views = create_synthData_new(N, mode=1, F=20)

print(f'input views shape :')
for i, view in enumerate(views):
    print(f'view_{i} :  {view.shape}')
    view = view.to("cpu")

a = SNGCCA_ADAM.fit(views, eps=1e-5, maxit=10, b=(2,2,2),early_stopping=True, patience=10, logging=1)
print(a)

import scipy.linalg as linalg
import numpy as np




def nystrom_kernel_svd(samples, kernel_fn, top_q):
    """Compute top eigensystem of kernel matrix using Nystrom method.

    Arguments:
        samples: data matrix of shape (n_sample, n_feature).
        kernel_fn: tensor function k(X, Y) that returns kernel matrix.
        top_q: top-q eigensystem.

    Returns:
        eigvals: top eigenvalues of shape (top_q).
        eigvecs: (rescaled) top eigenvectors of shape (n_sample, top_q).
    """

    n_sample, _ = samples.shape
    kmat = kernel_fn(samples, samples).cpu().data.numpy()
    scaled_kmat = kmat / n_sample
    vals, vecs = linalg.eigh(scaled_kmat,
                             eigvals=(n_sample - top_q, n_sample - 1))
    eigvals = vals[::-1][:top_q]
    eigvecs = vecs[:, ::-1][:, :top_q] / np.sqrt(n_sample)
    beta = np.diag(kmat).max()

    return eigvals, eigvecs, beta
'''

