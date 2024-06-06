import torch
from synth_data import create_synthData_new
import pandas as pd
import os

# Hyper Params Section
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")

import numpy as np

num = 5
sample = 400
tol = 100
root = 'D:/GitHub/SNGCCA/SNGCCA/Data/'
dir_sce1 = 'Linear/' + str(sample) + '_' + str(tol) + '_' + str(num) + '/'
dir_sce2 = 'Nonlinear/' + str(sample) + '_' + str(tol) + '_' + str(num) + '/'

N = 100
rep = 0
u1 = []
u2 = []
u3 = []

for rep in range(100):
    #print("REP=",rep)
    folder_path = root+dir_sce1
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = root+dir_sce2
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    views = create_synthData_new(num,sample, mode=1, F=tol)
    pd.DataFrame(views[0]).to_csv(root+dir_sce1+'data1_'+str(rep)+'.csv', index=False, header=False)
    pd.DataFrame(views[1]).to_csv(root+dir_sce1+'data2_'+str(rep)+'.csv', index=False, header=False)
    pd.DataFrame(views[2]).to_csv(root+dir_sce1+'data3_'+str(rep)+'.csv', index=False, header=False)

    views = create_synthData_new(num,sample, mode=2, F=tol)
    pd.DataFrame(views[0]).to_csv(root+dir_sce2+'data1_'+str(rep)+'.csv', index=False, header=False)
    pd.DataFrame(views[1]).to_csv(root+dir_sce2+'data2_'+str(rep)+'.csv', index=False, header=False)
    pd.DataFrame(views[2]).to_csv(root+dir_sce2+'data3_'+str(rep)+'.csv', index=False, header=False)
    