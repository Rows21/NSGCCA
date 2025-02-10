from gcca_admm_new import gcca_admm
from utils import _get_tcga
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':

    device = 'cpu'
    root = 'E:/res/SNGCCA'

    views = _get_tcga(root)
    views = [view for view in views]
    print(f'input views shape :')
    for i, view in enumerate(views):
        print(f'view_{i} :  {view.shape}')
    
    # Start Training
    u_list = []  
    obj_temp = []
    test_total = []

    df_u1_total = pd.DataFrame()
    df_u2_total = pd.DataFrame()
    df_u3_total = pd.DataFrame()

    model = gcca_admm(views,1)
    model.admm()
    df_u1 = pd.DataFrame(np.real(model.list_U[0]), columns=['u1'])
    df_u1_total = pd.concat([df_u1_total, df_u1], axis=1)
    df_u2 = pd.DataFrame(np.real(model.list_U[1]), columns=['u2'])
    df_u2_total = pd.concat([df_u2_total, df_u2], axis=1)
    df_u3 = pd.DataFrame(np.real(model.list_U[2]), columns=['u3']) 
    df_u3_total = pd.concat([df_u3_total, df_u3], axis=1)

    dir_path = "./RealData"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    df_u1_total.to_csv(root+'/RealData/sgadmm_u1.csv')
    df_u2_total.to_csv(root+'/RealData/sgadmm_u2.csv')
    df_u3_total.to_csv(root+'/RealData/sgadmm_u3.csv')

