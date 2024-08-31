from main import Solver
import torch
import numpy as np
import pandas as pd
import os
#from synth_data import create_synthData_new, create_synthData_multi

if __name__ == '__main__':

    #rs = os.environ.get('SLURM_JOB_ID')
    #torch.manual_seed(rs)  # random seed

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using", torch.cuda.device_count(), "GPUs")

    Exp_label = pd.read_csv('./SNGCCA/RealData/Exp664_genes.txt', sep='\t',header = None)
    Exp_list = Exp_label.iloc[:, 0].values.tolist()
    Exp = pd.DataFrame(np.loadtxt("./SNGCCA/RealData/Exp664.txt").T,columns = Exp_label)

    Meth_label = pd.read_csv('./SNGCCA/RealData/Meth664_probes.txt', sep='\t',header = None)
    Meth_list = Meth_label.iloc[:, 0].values.tolist()
    Meth = pd.DataFrame(np.loadtxt("./SNGCCA/RealData/Meth664.txt").T,columns = Meth_label)

    miRNA_label = pd.read_csv('./SNGCCA/RealData/miRNA664_miRNA.txt', sep='\t',header = None)
    miRNA_list = miRNA_label.iloc[:, 0].values.tolist()
    miRNA = pd.DataFrame(np.loadtxt("./SNGCCA/RealData/miRNA664.txt").T,columns = miRNA_label)

    y = pd.read_csv('./SNGCCA/RealData/PAM50label664.txt',header = None)

    Exp_value = np.loadtxt("./SNGCCA/RealData/Exp664.txt")
    Meth_value = np.loadtxt("./SNGCCA/RealData/Meth664.txt")
    miRNA_value = np.loadtxt("./SNGCCA/RealData/miRNA664.txt")
    views = [torch.tensor(Exp_value).T,torch.tensor(Meth_value).T,torch.tensor(miRNA_value).T]

    print(f'input views shape :')
    for i, view in enumerate(views):
        print(f'view_{i} :  {view.shape}')
        view = view.to("cpu")

    solver = Solver(device)

    # Start Training
    u_list = []
    obj_temp = []
    test_total = []

    df_u1_total = pd.DataFrame()
    df_u2_total = pd.DataFrame()
    df_u3_total = pd.DataFrame()

    for rep in range(1):
        ## split
        print("REP=", rep + 1)
        shuffled_index = np.random.permutation(len(views[0]))
        split_index = int(len(views[0]) * 1 / 4)
        data_size = views[0].size(0)
        start_index = split_index
        fold_index = shuffled_index[start_index:data_size]

        test_list = []
        train_list = []
        '''
        print(f'input views shape :')
        for i, view in enumerate(test_list):
            print(f'view_{i} :  {view.shape}')
            view = view.to("cpu")
        '''

        # Hyper Params Section
        #b0, obj = Solver.tune_hyper(x_list=train_list, set_params=5,max_params=200,iter=100)
        #print(b0)

        ## fit results
        b = [0.025,0.0005,0.01]
        u = solver.SNGCCA.fit_admm2(views, lamb=b, logging=0)

        df_u1 = pd.DataFrame(u[0], columns=['u1_' + str(rep + 1)])
        df_u1_total = pd.concat([df_u1_total, df_u1], axis=1)
        df_u2 = pd.DataFrame(u[1], columns=['u2_' + str(rep + 1)])
        df_u2_total = pd.concat([df_u2_total, df_u2], axis=1)
        df_u3 = pd.DataFrame(u[2], columns=['u3_' + str(rep + 1)])
        df_u3_total = pd.concat([df_u3_total, df_u3], axis=1)

        dir_path = "./RealData"
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        df_u1_total.to_csv('./RealData/U1.csv')
        df_u2_total.to_csv('./RealData/U2.csv')
        df_u3_total.to_csv('./RealData/U3.csv')

        df_obj = pd.DataFrame(obj_temp)
        df_obj.to_csv('./RealData/OBJ.csv')
