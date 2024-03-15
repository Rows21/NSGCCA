from SNGCCA.main import Solver
import torch
from synth_data import create_synthData_new, create_synthData_multi
from validation_method import FS_MCC

import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    import sys

    #rs = os.environ.get('SLURM_JOB_ID')
    #torch.manual_seed(rs)  # random seed

    # Hyper Params Section
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using", torch.cuda.device_count(), "GPUs")

    for scenario in range(3):
    ## Scenario
        ## Evaluation params
        ACC_list = []
        FS_list = []
        MCC_list = []

        for batch_size in [0.1,0.2,0.5,0.8]:
            print("Scenario=",scenario+1,"kernel size=",batch_size)

            FS = []
            MCC = []
            ACC = []
            views = create_synthData_new(400, mode=scenario+1, F=20)
            print(f'input views shape :')
            for j, view in enumerate(views):
                print(f'view_{j} :  {view.shape}')
                view = view.to("cpu")

            ## train hyper
            solver = Solver(device,batch_size=batch_size)
            b0, obj = solver.tune_hyper(x_list=views, set_params=6, max_params=50, iters=100, approx=True, logging=0)
            print(b0)

            print("SNGCCA Started!")
            for rep in range(100):
                if (rep + 1) % 20 == 0:
                    print("REP=", rep + 1)

                ## fit results
                u = solver._get_outputs(views=views, eps=1e-7,maxit=200, b=(b0,b0,b0), approx=True)

                Label = torch.cat([torch.ones(2, dtype=torch.bool), torch.zeros(18, dtype=torch.bool)])
                acc, f1, mcc = FS_MCC(u, Label)
                ACC.append(acc)
                FS.append(f1)
                MCC.append(mcc)

            df = pd.DataFrame({'Accuracy': ACC,
                               'F-Score': FS,
                               'MCC': MCC})

            dir_path = "./Simulation"
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            df.to_csv('./Simulation/Scenario' + str(scenario) + 'Kernel' + str(batch_size) + '.csv')
            macc = np.mean(ACC)
            sdacc = np.std(ACC)
            print(macc, sdacc)
            ACC_list.append([macc, sdacc])

            mf = np.mean(FS)
            sdf = np.std(FS)
            print(mf, sdf)
            FS_list.append([mf, sdf])

            mmcc = np.mean(MCC)
            sdmcc = np.std(MCC)
            print(mmcc, sdmcc)
            MCC_list.append([mmcc, sdmcc])
