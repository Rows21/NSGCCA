from main import Solver
import torch
from synth_data import create_synthData_new
from validation_method import FS_MCC
import numpy as np

# Hyper Params Section
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")

Solver = Solver()

## Scenario 2
FS = []
MCC = []
N = 400
views = create_synthData_new(N, mode=1, F=20)

print(f'input views shape :')
for i, view in enumerate(views):
    print(f'view_{i} :  {view.shape}')
    view = view.to("cpu")

for rep in range(20):
    print("REP=", rep + 1)
    u = []
    ## train hyper
    b0, obj = Solver.tune_hyper(x_list=views, set_params=3,iter=100)

    ## fit results
    u = Solver._get_outputs(views, 1e-7, 100, b0)
    print(u)
    Label = torch.cat([torch.ones(2, dtype=torch.bool), torch.zeros(18, dtype=torch.bool)])
    res = FS_MCC(u, Label)
    FS.append(res[0])
    MCC.append(res[1])

mf = np.mean(FS)
sdf = np.std(FS)
print(mf, sdf)

mmcc = np.mean(MCC)
sdmcc = np.std(MCC)
print(mmcc, sdmcc)