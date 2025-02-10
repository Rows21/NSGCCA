#import torch
import numpy as np
from tqdm import tqdm
#torch.set_default_tensor_type(torch.DoubleTensor)

from synth_data import create_synthData
from networks.sgcca_hsic import SGCCA_HSIC
from sklearn.model_selection import KFold
from validation_method import eval, eval_plot
from networks.utils import rbf_kl, rbf_kx
import itertools

import time
#torch.manual_seed(0)
class Solver():
    def __init__(self, device='cpu'):
        self.SNGCCA = SGCCA_HSIC(device)
        self.device = device

    def fit(self, x_list, test_list=None, train_list=None, eps=1e-7, maxit=100, b=(100,100,100), k=3):
        x_list = [x.to(device) for x in x_list]
        # split
        shuffled_index = np.random.permutation(len(x_list[0]))
        split_index = int(len(x_list[0]) * 1 / k)
        data_size = x_list[0].size(0)
        start_index = i * split_index
        end_index = (i + 1) * split_index
        fold_index = shuffled_index[start_index:end_index]

        if test_list is None:
            test_list = []
        if train_list is None:
            train_list = []

        for _, view in enumerate(x_list):
            test_list.append(view[fold_index, :])
            non_fold_index = [num for num in shuffled_index if num not in fold_index]
            train_list.append(view[non_fold_index, :])

        u = Solver._get_outputs(train_list, eps, maxit, b)

        return train_list, test_list, u

    def tune_hyper(self, x_list, k=5, mode = 'cv', a=[1e-5, 1e-5, 1e-5]):
        # split
        shuffled_index = np.random.permutation(len(x_list[0]))
        split_index = int(len(x_list[0]) * 1/k)
        fold_index = []
        for j in range(k):
            start_index = j * split_index
            end_index = (j + 1) * split_index
            fold_index.append(shuffled_index[start_index:end_index])
            
        obj_validate = 999999
        if mode == 'cv':
            # set hyperparams set
            #a = [1e-4, 1e-4, 1e-4]
            print("Start Hyperparams Tuning")
            # fixed folds number
            # start cross validation
            b0 = a
            count = 0

            # for a in combinations_with_replacement(a, 3):
            while max(a) < 1e-1:
                a = [i * 10 for i in a]
                count +=1
                o_list = [0] * 3
                obj_temp = np.zeros((k, len(x_list)))
                min_index, mean_obj = self._cv(x_list, fold_index, shuffled_index, obj_temp, o_list, a, mode, k)
                print("Sparsity selection number=", count, "hyperparams=", a, "obj=", mean_obj)
                #a[min_index] = a[min_index] * 10
                
                if mean_obj < obj_validate:
                    b0 = a
                    #print(b0)
                    obj_validate = mean_obj
                else:
                    continue
                
            print("Finish Tuning!")
            return b0, obj_validate
        elif mode == 'multi_start':
            a = a
            b0 = None
            print("Start Multi-start Tuning")
            for i in tqdm(range(k)):
                o_list = [0] * 3
                obj_temp = np.zeros((k, len(x_list)))
                if i == 0:
                    Pi0_list = [np.eye(x_list[j].shape[1])/np.trace(np.cov(x_list[j].T)) for j in range(3)]
                Pi_list, u_list, obj = self.SNGCCA.fit_admm(x_list, constraint=a, criterion = 5e-5, logging=2, mode=mode, Pi0_list=Pi0_list)
                if obj > obj_validate:
                    b0 = self.SNGCCA.Pi0_list
                    obj_validate = obj
                else:
                    continue
            
            return b0
    
    def _cv(self, x_list, fold_index, shuffled_index, obj_temp, o_list, a, mode, k=5):
        for fold in tqdm(range(k)):

            fold_mask = np.zeros_like(x_list[0], dtype=bool)
            fold_mask[fold_index[fold]] = True

            train_data = []
            test_data = []

            for _, view in enumerate(x_list):

                test_data.append(view[fold_index[fold], :])
                non_fold_index = [num for num in shuffled_index if num not in fold_index[fold]]
                train_data.append(view[non_fold_index, :])
                
                
            Pi_list, u_list = self.SNGCCA.fit_admm(train_data, constraint=a, criterion = 1e-4, logging=2, mode=mode)
            K_list = [rbf_kx(test_data[i], Pi_list[i]) for i in range(len(test_data))]
            obj_k = []
            for items in itertools.combinations(range(len(K_list)), 2):
                obj_k.append(np.trace(K_list[items[0]] @ rbf_kl(K_list[items[1]])))
            obj_temp[fold] = np.stack(obj_k)
            o_list = [np.sum(abs(u_list[i]) < 0.05) + o_list[i] for i in range(3)]
        len_u = [len(u) for u in u_list]
        mean_obj = np.mean(np.sum(obj_temp, axis=1))
        min_index = np.argmin(np.stack(o_list)/np.stack(len_u)/k)#np.argmax([np.mean(np.abs(i)) for i in u_list])
        
        return min_index, mean_obj
    

    def _get_outputs(self, b, logging=1):
        u = self.SNGCCA.fit_admm(b, logging)
        return u


if __name__ == '__main__':
    ############
    # Hyper Params Section
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print("Using", torch.cuda.device_count(), "GPUs")
    device='cpu'
    import numpy as np
    mode = 1
    N = 100
    num = 5
    tol = 100
    
    Pi = None

    views = create_synthData(num,N, mode=mode, F=tol)
    solver = Solver(device=device)
    
    constraint, _ = solver.tune_hyper(views, k=5, mode='cv')
    print(constraint)
    #Pi = solver.tune_hyper(views, k=1, mode='multi_start', a=constraint)
    criterion = 5e-3
    u = solver.SNGCCA.fit_admm(views, constraint=constraint, criterion=criterion, logging=1)
    print(u[0])
    #print(u[1])
    #print(u[2])

 