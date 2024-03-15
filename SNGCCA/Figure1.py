from SNGCCA.main import Solver
import torch
from synth_data import create_synthData_new
from validation_method import FS_MCC
import numpy as np
import pandas as pd

# Hyper Params Section
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")

solver = Solver(device)
## Evaluation params

## Scenario 2
#x = np.genfromtxt('xdata.csv', delimiter=',')
seed = 0
torch.manual_seed(seed)

## Scenario 1
#x = np.genfromtxt('xdata.csv', delimiter=',')
seed = 0
torch.manual_seed(seed)

# LD
N = 100
views = create_synthData_new(10,N, mode=1, F=100)


b = [0.015,0.04,0.03]
u11 = solver.SNGCCA.fit_admm2(views, lamb=b, logging=1)
print(u11)

gtx = views[0][:,0] + views[0][:,1]
gty = views[1][:,0] + views[1][:,1]
gtz = views[2][:,0] + views[2][:,1]

x1 = views[0] @ u11[0]
y1 = views[1] @ u11[1]
z1 = views[2] @ u11[2]

# HD
views1 = create_synthData_new(2,N, mode=1, F=100)
b = [0.016,0.03,0.03]
u12 = solver.SNGCCA.fit_admm2(views1, lamb=b, logging=1)
print(u12)

x1h = views1[0] @ u12[0]
y1h = views1[1] @ u12[1]
z1h = views1[2] @ u12[2]

df = pd.DataFrame({'GTX': gtx.reshape(-1).numpy(),
                   'GTY': gty.reshape(-1).numpy(),
                   'GTZ': gtz.reshape(-1).numpy(),
                   'XL': x1.reshape(-1).numpy(),
                   'YL': y1.reshape(-1).numpy(),
                   'ZL': z1.reshape(-1).numpy(),
                   'XH': x1h.reshape(-1).numpy(),
                   'YH': y1h.reshape(-1).numpy(),
                   'ZH': z1h.reshape(-1).numpy()})

df.to_csv('.\SNGCCA\Results\Figure2\Fig1a.csv')

# LD
N = 100
views = create_synthData_new(2,N, mode=2, F=20)
b = [0.015,0.065,0.03]
u21 = solver.SNGCCA.fit_admm2(views, lamb=b, logging=1)
print(u21)

gtx = views[0][:,0] + views[0][:,1]
gty = views[1][:,0] + views[1][:,1]
gtz = views[2][:,0] + views[2][:,1]

x1 = views[0] @ u21[0]
y1 = views[1] @ u21[1]
z1 = views[2] @ u21[2]

# HD
views2 = create_synthData_new(2,N, mode=2, F=100)
b = [0.015,0.065,0.03]
u22 = solver.SNGCCA.fit_admm2(views1, lamb=b, logging=1)
print(u22)

x1h = views2[0] @ u22[0]
y1h = views2[1] @ u22[1]
z1h = views2[2] @ u22[2]

df = pd.DataFrame({'GTX': gtx.reshape(-1).numpy(),
                   'GTY': gty.reshape(-1).numpy(),
                   'GTZ': gtz.reshape(-1).numpy(),
                   'XL': x1.reshape(-1).numpy(),
                   'YL': y1.reshape(-1).numpy(),
                   'ZL': z1.reshape(-1).numpy(),
                   'XH': x1h.reshape(-1).numpy(),
                   'YH': y1h.reshape(-1).numpy(),
                   'ZH': z1h.reshape(-1).numpy()})

df.to_csv('.\SNGCCA\Results\Figure2\Fig1b.csv')