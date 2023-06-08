import torch
import numpy as np
import math
from itertools import combinations_with_replacement
torch.set_default_tensor_type(torch.DoubleTensor)

from synth_data import create_synthData_new
from sgcca_hsic import SGCCA_HSIC

from validation_method import FS_MCC

class Solver():
    def __init__(self,device):
        self.SGCCA_HSIC = SGCCA_HSIC(device)
        self.device = device

    def fit(self, x_list, vx_list=None, tx_list=None, checkpoint='checkpoint.model'):
        x_list = [x.to(device) for x in x_list]
        data_size = x_list[0].size(0)

    def tune_hyper(self,x_list,set_params,eps = 1e-5,iter = 20):
        ## set hyperparams set
        a = np.exp(np.linspace(0, math.log(5), num=set_params))
        print("Start Hyperparams Tuning")
        ## fixed folds number
        folds = 3
        ## split
        shuffled_index = np.random.permutation(len(x_list[0]))
        split_index = int(len(x_list[0]) * 1/folds)
        train_index = shuffled_index[:split_index]
        test_index = shuffled_index[split_index:]

        train_data = []
        test_data = []
        for i, view in enumerate(x_list):
            train_data.append(view[train_index, :])
            test_data.append(view[test_index, :])

        ## start validation
        b0 = a[0]
        obj_validate = 0
        count = 0
        for aa in combinations_with_replacement(a, 3):
            count +=1
            u = self.SGCCA_HSIC.fit(train_data,eps,iter,aa,logging=2)

            # Save iterations
            ## calculate K,cK for validation set
            K_test = []
            cK_test = []
            for i,view in enumerate(test_data):
                Xu = view.to(self.device) @ u[i]
                sigma = None
                if sigma is None:
                    K, a = self.SGCCA_HSIC.rbf_kernel(Xu)
                else:
                    K, a = self.SGCCA_HSIC.rbf_kernel(Xu, sigma)
                cK = self.SGCCA_HSIC.centre_kernel(K)
                K_test.append(K)
                cK_test.append(cK)
            ## get obj
            obj_temp = self.SGCCA_HSIC.ff(K_test,cK_test)

            #print("Sparsity selection number=", count, "hyperparams=", aa,"obj=",obj_temp)
            if obj_temp > obj_validate:
                b0 = aa
                obj_validate = obj_temp
            else:
                continue
        print("Finish Tuning!")
        return b0,obj_validate

    def _get_outputs(self,views,eps,maxit,b):
        print("SNGCCA Started!")
        u = self.SGCCA_HSIC.fit(views, eps, maxit,b,patience=10,logging=1)
        return u

    def early_stop(self):
        pass

    def test(self):
        pass


if __name__ == '__main__':
    ############
    # Hyper Params Section
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using", torch.cuda.device_count(), "GPUs")

    N = 400
    views = create_synthData_new(N, mode=1, F=20)

    print(f'input views shape :')
    for i, view in enumerate(views):
        print(f'view_{i} :  {view.shape}')
        view = view.to("cpu")

    u = []
    a = SGCCA_HSIC(device)

    ## fit results
    u = a.fit(views,1e-7,50,(1,1,1),logging=1)
    print(u)











