from main import Solver
#import torch
import numpy as np
import pandas as pd
import os
#from synth_data import create_synthData_new, create_synthData_multi

if __name__ == '__main__':

    #rs = os.environ.get('SLURM_JOB_ID')
    #torch.manual_seed(rs)  # random seed
    #print("PyTorch Version: ", torch.__version__)
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    #print("Using", torch.cuda.device_count(), "GPUs")
    root = 'E:/Github/SNGCCA'
    #root = '/scratch/rw2867/projects/SNGCCA'
    Exp_label = pd.read_csv(root+'/SNGCCA/RealData/Exp664_genes.txt', sep='\t',header = None)
    Exp_list = Exp_label.iloc[:, 0].values.tolist()
    Exp = pd.DataFrame(np.loadtxt(root+"/SNGCCA/RealData/Exp664.txt").T,columns = Exp_label)

    Meth_label = pd.read_csv(root+'/SNGCCA/RealData/Meth664_probes.txt', sep='\t',header = None)
    Meth_list = Meth_label.iloc[:, 0].values.tolist()
    Meth = pd.DataFrame(np.loadtxt(root+"/SNGCCA/RealData/Meth664.txt").T,columns = Meth_label)

    miRNA_label = pd.read_csv(root+'/SNGCCA/RealData/miRNA664_miRNA.txt', sep='\t',header = None)
    miRNA_list = miRNA_label.iloc[:, 0].values.tolist()
    miRNA = pd.DataFrame(np.loadtxt(root+"/SNGCCA/RealData/miRNA664.txt").T,columns = miRNA_label)

    y = pd.read_csv(root+'/SNGCCA/RealData/PAM50label664.txt',header = None)

    Exp_value = np.loadtxt(root+"/SNGCCA/RealData/Exp664.txt")
    Meth_value = np.loadtxt(root+"/SNGCCA/RealData/Meth664.txt")
    miRNA_value = np.loadtxt(root+"/SNGCCA/RealData/miRNA664.txt")
    views = [Meth_value.T,
             Exp_value.T,
             miRNA_value.T
             ]
    # stadardize
    for i, view in enumerate(views):
        views[i] = (view - np.mean(view, axis=0)) / np.std(view, axis=0)
        
    print(f'input views shape :')
    for i, view in enumerate(views):
        print(f'view_{i} :  {view.shape}')

    solver = Solver(device)
    #solver = Solver(mode='cv', device=device)
    #b = solver.tune_hyper(views, k=5)
    
    # Start Training
    u_list = []  
    obj_temp = []
    test_total = []

    df_u1_total = pd.DataFrame()
    df_u2_total = pd.DataFrame()
    df_u3_total = pd.DataFrame()

    for rep in range(1):
        ## split
        '''
        print(f'input views shape :')
        for i, view in enumerate(test_list):
            print(f'view_{i} :  {view.shape}')
            view = view.to("cpu")
        '''

        # Hyper Params Section
        #b0, obj = Solver.tune_hyper(x_list=train_list, set_params=5,max_params=200,iter=100)
        #print(b0)
        print("Start",rep)
        ## fit results
        #b = [0.0001, 0.01, 0.005]
        b = [0.0055,0.0025,0.0035]
        print(b)
        if device == 'cuda':
            u = solver.SNGCCA.fit_admm(views, constraint=b, criterion=5e-6, logging=1)
        else:
            u = solver.SNGCCA.fit_admm(views, constraint=b, criterion=5e-6, logging=1)

        df_u1 = pd.DataFrame(u[0], columns=['u1_' + str(rep + 1)])
        df_u1_total = pd.concat([df_u1_total, df_u1], axis=1)
        df_u2 = pd.DataFrame(u[1], columns=['u2_' + str(rep + 1)])
        df_u2_total = pd.concat([df_u2_total, df_u2], axis=1)
        df_u3 = pd.DataFrame(u[2], columns=['u3_' + str(rep + 1)]) 
        df_u3_total = pd.concat([df_u3_total, df_u3], axis=1)

        dir_path = "./RealData"
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        df_u1_total.to_csv(root+'/RealData/U1.csv')
        df_u2_total.to_csv(root+'/RealData/U2.csv')
        df_u3_total.to_csv(root+'/RealData/U3.csv')

        df_obj = pd.DataFrame(obj_temp)
        df_obj.to_csv(root+'/RealData/OBJ.csv')
