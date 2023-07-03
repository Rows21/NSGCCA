from main import Solver
import pandas as pd
import torch
from synth_data import create_synthData_new
from validation_method import FS_MCC
import numpy as np
from sngcca_approx import SNGCCA_APPROX
import scipy

import torch.optim as optim
torch.set_default_tensor_type(torch.DoubleTensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")

SNGCCA_APPROX = SNGCCA_APPROX(device,batch_size=100)

N = 400
views = create_synthData_new(N, mode=1, F=20)

a = SNGCCA_APPROX.fit(views, eps=1e-7, maxit=50, b=(10,10,10),early_stopping=True, patience=5, logging=1)
print(a)
