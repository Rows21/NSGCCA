from main import Solver
import torch
from synth_data import create_synthData_new

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")

## import TCGA Cancer dataset
N = 400
views = create_synthData_new(N, mode=1, F=20)

## start CCA process
Solver = Solver()
b0,obj = Solver.tune_hyper(x_list=views,set_params=5)

## fit results
u = Solver._get_outputs(views,1e-5,20,b0)