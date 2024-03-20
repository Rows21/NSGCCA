from main import Solver
import torch
from synth_data import create_synthData_new
from validation_method import eval

import os
import time
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # Slurm Object note
    #import sys
    #rs = os.environ.get('SLURM_JOB_ID')
    #torch.manual_seed(rs)  # random seed

    # Hyper Params Section
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using", torch.cuda.device_count(), "GPUs")

    # views = create_synthData_hd()
    '''
    N = 100
    views = create_synthData_new(N, mode=1, F=60)

    print(f'input views shape :')
    for i, view in enumerate(views):
        print(f'view_{i} :  {view.shape}')
        view = view.to("cpu")
    Solver = Solver(device)
    u = Solver._get_outputs(views, 1e-7, 300, (10, 10, 10)) 
    #train,test,u = Solver.fit(views)
    print(u)
    #b0, obj = Solver.tune_hyper(x_list=views, set_params=5, max_params=50, iters=100)
    #print(b0)
    '''
    #seed = 0
    #torch.manual_seed(seed)
    
    N = 100
    rep = 0
    u1 = []
    u2 = []
    u3 = []
    FS = []
    MCC = []
    ACC = []
    PRE = []
    REC = []
    start_time = time.time()

    for i in range(10):
        views = create_synthData_new(5,N, mode=2, F=30)
        solver = Solver(device)
        b = [0.006,0.008,0.008]
        try:
            u = solver.SNGCCA.fit_admm2(views, lamb=b,logging=0)  
        except:
            continue
        
    end_time = time.time()
    time_diff = end_time - start_time
    print("time diff:", time_diff/20, "s")
