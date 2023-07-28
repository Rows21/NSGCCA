#from main import Solver
import pandas as pd
import torch
from synth_data import create_synthData_new
from validation_method import FS_MCC
import numpy as np
from sngcca_approx import SNGCCA_APPROX
import scipy

import torch.optim as optim
import math
torch.set_default_tensor_type(torch.DoubleTensor)
a = (np.exp(np.linspace(0, math.log(2), num=10)) - 1) * 49 + 1

for aa in a:
    print(aa)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")

from synth_data import create_synthData_new
from main import Solver


views = create_synthData_new(400, mode=1, F=20)
Solver = Solver(device="cpu")
u = Solver._get_outputs(views, 1e-5, 100, (1,1,1))

from sklearn.cross_decomposition import CCA
import numpy as np
print(u)
# 执行CCA分析
cca = CCA(n_components=1)
cca.x_weights_ = u[0]
cca.y_weights_ = u[1]
cca.fit(views[0], views[1])


# 输出结果
print(f"Canonical Correlation Coefficient: {cca.score(views[0], views[1]):.2f}")


# Get the canonical correlation coefficient


corr = np.sqrt(sorted(np.linalg.eigvals(M))[::-1])

from gcca.gcca import GCCA
import numpy as np


# create data in advance
a = np.random.rand(400, 20)
b = np.random.rand(400, 20)
c = np.random.rand(400, 20)
print(a.shape)
# create instance of GCCA
gcca = GCCA()
# calculate GCCA
gcca.fit(a, b, c)
# transform
res = gcca.transform(a, b, c)


print(gcca.z_list[0].shape)
gcca.plot_result()


#SNGCCA_APPROX = SNGCCA_APPROX(device,batch_size=100)

#N = 400
#views = create_synthData_new(N, mode=1, F=20)

#a = SNGCCA_APPROX.fit(views, eps=1e-7, maxit=50, b=(10,10,10),early_stopping=True, patience=5, logging=1)
#print(a)

