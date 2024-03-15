import torch
import numpy as np
import pandas as pd

from SNGCCA.main import Solver
torch.set_default_tensor_type(torch.DoubleTensor)

from synth_data import create_synthData_new

from validation_method import FS_MCC

seed = 0
torch.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")
N=100
views = create_synthData_new(5,N, mode=2, F=30)
solver = Solver(device)
b = [0.05,0.01,0.03]
#try:
u1 = solver.SNGCCA.fit_admm2(views, lamb=b,logging=1) 

x = views[0].numpy()[:,0:5]
gtx = np.sum(x, axis=1)
y = views[1].numpy()[:,0:5]
gty = np.sum(y, axis=1)
z = views[2].numpy()[:,0:5]
gtz = np.sum(y, axis=1)

x1 = views[0] @ u1[0]
y1 = views[1] @ u1[1]
z1 = views[2] @ u1[2]

import matplotlib.pyplot as plt
plt.plot(x1, y1, 'bo', label='Data 1')
plt.show()
plt.plot(x1, z1, 'bo', label='Data 2')
plt.show()
plt.plot(y1, z1, 'bo', label='Data 3')
plt.show()

df = pd.DataFrame({'GTX': gtx,
                   'GTY': gty,
                   'GTZ': gtz,
                   'XL': x1.reshape(-1).numpy(),
                   'YL': y1.reshape(-1).numpy(),
                   'ZL': z1.reshape(-1).numpy(),
                   'XH': x1.reshape(-1).numpy(),
                   'YH': x1.reshape(-1).numpy(),
                   'ZH': x1.reshape(-1).numpy()})

df.to_csv('.\SNGCCA\Results\Figure2\Fig1b.csv')
