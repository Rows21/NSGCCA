from main import Solver
import torch
from synth_data import create_synthData_new, create_synthData_multi
from validation_method import FS_MCC

import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    import sys

    rs = os.environ.get('SLURM_JOB_ID')
    torch.manual_seed(rs)  # random seed

    # Hyper Params Section
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using", torch.cuda.device_count(), "GPUs")

    Solver = Solver(device)

    for scenario in range(3):
    ## Scenario
        ## Evaluation params
        ACC_list = []
        FS_list = []
        MCC_list = []

        for N in [10,30,40]:
            print("Scenario=",scenario+1,"Noise=",N)

            FS = []
            MCC = []
            ACC = []
            views = create_synthData_multi(i=2, data_type=scenario+1, N=400, p=N, q=N, r=N)
            print(f'input views shape :')
            for j, view in enumerate(views):
                print(f'view_{j} :  {view.shape}')
                view = view.to("cpu")

            ## train hyper
            b0, obj = Solver.tune_hyper(x_list=views, set_params=20,max_params=100, iter=100)
            print(b0)

            print("SNGCCA Started!")
            for rep in range(100):
                if (rep + 1) % 100 == 0:
                    print("REP=", rep + 1)

                ## fit results
                u = Solver._get_outputs(views, 1e-7, 300, (b0,b0,b0))

                Label = torch.cat([torch.ones(2, dtype=torch.bool), torch.zeros(N-2, dtype=torch.bool)])
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

            df.to_csv('./Simulation/Scenario' + str(scenario+1) + 'Noise' + str(N) + '.csv')
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
