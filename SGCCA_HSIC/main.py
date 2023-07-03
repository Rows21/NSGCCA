import torch
import numpy as np
import math
from itertools import combinations_with_replacement
torch.set_default_tensor_type(torch.DoubleTensor)

from synth_data import create_synthData_new
from sngcca import SNGCCA
from sngcca_approx import SNGCCA_APPROX

from validation_method import FS_MCC

class Solver():
    def __init__(self,device,batch_size=100):
        self.SNGCCA = SNGCCA(device)
        self.SNGCCA_APPROX = SNGCCA_APPROX(device,batch_size=batch_size)
        self.device = device

    def fit(self, x_list, vx_list=None, tx_list=None, method='checkpoint.model'):
        x_list = [x.to(device) for x in x_list]
        data_size = x_list[0].size(0)

    def tune_hyper(self,x_list,set_params,eps = 1e-6,iter = 20,k=5):
        ## set hyperparams set
        a = np.exp(np.linspace(0, math.log(50), num=set_params))
        print("Start Hyperparams Tuning")
        ## fixed folds number

        ## split
        shuffled_index = np.random.permutation(len(x_list[0]))
        split_index = int(len(x_list[0]) * 1/k)
        fold_index = []
        for i in range(k):
            start_index = i * split_index
            end_index = (i + 1) * split_index
            fold_index.append(shuffled_index[start_index:end_index])

        # start cross validation
        b0 = a[0]
        obj_validate = 0
        count = 0

        for aa in combinations_with_replacement(a, 3):

            count +=1
            obj_temp = []
            for i in range(k):

                fold_mask = np.zeros_like(x_list[0], dtype=bool)
                fold_mask[fold_index[i]] = True

                train_data = []
                test_data = []

                for _, view in enumerate(x_list):
                    a = fold_index[i]

                    test_data.append(view[fold_index[i], :])
                    non_fold_index = [num for num in shuffled_index if num not in fold_index[i]]
                    train_data.append(view[non_fold_index, :])

                u = self.SNGCCA.fit(train_data,eps,iter,aa,logging=0)

                # Save iterations
                ## calculate K,cK for validation set
                K_test = []
                cK_test = []
                for j,view in enumerate(test_data):
                    Xu = view.to(self.device) @ u[j]
                    sigma = None
                    if sigma is None:
                        K, a = self.SNGCCA.rbf_kernel(Xu)
                    else:
                        K, a = self.SNGCCA.rbf_kernel(Xu, sigma)
                    cK = self.SNGCCA.centre_kernel(K)
                    K_test.append(K)
                    cK_test.append(cK)
                ## get obj
                obj_temp.append(self.SNGCCA.ff(K_test,cK_test))

            mean_obj = sum(obj_temp)/len(obj_temp)
            print("Sparsity selection number=", count, "hyperparams=", aa, "obj=", mean_obj)
            if mean_obj > obj_validate:
                b0 = aa
                obj_validate = mean_obj
            else:
                continue
        print("Finish Tuning!")
        return b0,obj_validate

    def _get_outputs(self,views,eps,maxit,b,logging=0):
        #print("SNGCCA Started!")
        u = self.SNGCCA.fit(views, eps, maxit,b,loss="SGD",patience=10,logging=logging)
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
    Solver = Solver(device)

    b0, obj = Solver.tune_hyper(x_list=views, set_params=5, iter=50)
    print(b0)











