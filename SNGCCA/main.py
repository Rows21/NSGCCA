#import torch
import numpy as np
from tqdm import tqdm
#torch.set_default_tensor_type(torch.DoubleTensor)

from synth_data import create_synthData_new
from sngcca import SNGCCA
from sklearn.model_selection import KFold
from validation_method import eval, eval_plot
from utils import rbf_kl, rbf_kx
import itertools

import time
#torch.manual_seed(0)
class Solver():
    def __init__(self, device='cpu'):
        self.SNGCCA = SNGCCA(device)
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

    def tune_hyper(self, x_list, k=5, mode = 'cv', a=[1e-4, 1e-4, 1e-4]):
        # split
        shuffled_index = np.random.permutation(len(x_list[0]))
        split_index = int(len(x_list[0]) * 1/k)
        fold_index = []
        for j in range(k):
            start_index = j * split_index
            end_index = (j + 1) * split_index
            fold_index.append(shuffled_index[start_index:end_index])
            
        obj_validate = 0
        if mode == 'cv':
            # set hyperparams set
            a = [1e-4, 1e-4, 1e-4]
            print("Start Hyperparams Tuning")
            # fixed folds number
            # start cross validation
            b0 = a
            count = 0

            # for a in combinations_with_replacement(a, 3):
            while max(a) < 1e-1:
                count +=1
                o_list = [0] * 3
                obj_temp = np.zeros((k, len(x_list)))
                min_index, mean_obj = self._cv(x_list, fold_index, shuffled_index, obj_temp, o_list, a, mode, k)
                a[min_index] = a[min_index] * 10
                #a = [i * 10 for i in a]

                print("Sparsity selection number=", count, "hyperparams=", a, "obj=", mean_obj)
                if mean_obj > obj_validate:
                    b0 = a
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
    root = 'E:/GitHub'
    mode = 2
    N = 100
    num = 5
    tol = 150
    if mode == 1:
        folder = 'Linear/'
    else:
        folder = 'Nonlinear/'
    data_path = root + '/SNGCCA/SNGCCA/Data/' + folder + '/' + str(N) + '_' + str(tol) + '_' + str(num) + '/'
    
    rep = 0
    u1 = []
    u2 = []
    u3 = []
    FS = []
    MCC = []
    ACC = []
    PRE = []
    REC = []
    t = []
    Pi = None
    for rep in range(100):
        print("REP=",rep)
        #views = create_synthData_new(num,N, mode=mode, F=tol)
        view1 = np.genfromtxt(data_path + 'data' + str(1) + '_' + str(rep) + '.csv', delimiter=',')
        view2 = np.genfromtxt(data_path + 'data' + str(2) + '_' + str(rep) + '.csv', delimiter=',')
        view3 = np.genfromtxt(data_path + 'data' + str(3) + '_' + str(rep) + '.csv', delimiter=',')
        views = [view1, view2, view3]
        #for i in range(len(views)):
        #    views[i].tofile(data_path + 'data' + str(i+1) + '_' + str(rep) + '.csv', sep=',')
        #    np.savetxt(data_path + 'data' + str(i+1) + '_' + str(rep) + '.csv', views[i], delimiter=',')
        
        solver = Solver(device=device)
        
        #constrant = [0.3,1,0.3]
        #constrant = [0.1,0.1,0.1]
        constrant = [0.07,1,0.2]
        #Pi = solver.tune_hyper(views, k=5, mode='multi_start', a=constrant)
        criterion = 5e-3
        s = time.time()
        u = solver.SNGCCA.fit_admm(views, constraint=constrant, criterion=criterion, logging=0)
        e = time.time()

        Label = np.concatenate([np.ones(num, dtype=bool), np.zeros(tol - num, dtype=bool)])
        spe, pre, rec, f1, mcc, sr = eval(u, Label, num)
        #spe, pre, rec, acc, f1, mcc = eval_topk(u, Label, 5)
        print(mcc)
        t.append(e - s)
        #PRE.append(pre)
        #REC.append(rec)
        #ACC.append(acc)
        #FS.append(f1)
        #MCC.append(mcc)
        u1.append(u[0])
        u2.append(u[1])
        u3.append(u[2])
    
    merged_array = merged_array = np.empty((100,tol))
    
    # Save results
    # Save data
    path = 'E:/GitHub/SNGCCA/SNGCCA/Simulation/' + folder + '/' + str(N) + '_' + str(tol) + '_' + str(num) + '/'
        
    for i, arr in enumerate(u1):
        merged_array[i] = u1[i].flatten()
    np.savetxt(path + 'u1.csv', merged_array, delimiter=',')
    for i, arr in enumerate(u2):
        merged_array[i] = u2[i].flatten()
    np.savetxt(path + 'u2.csv', merged_array, delimiter=',')
    for i, arr in enumerate(u3):
        merged_array[i] = u3[i].flatten()
    np.savetxt(path + 'u3.csv', merged_array, delimiter=',')
    np.savetxt(path + 't.csv', t, delimiter=',')
    macc = np.mean(ACC)
    sdacc = np.std(ACC)
    print(macc, sdacc)

    mf = np.mean(FS)
    sdf = np.std(FS)
    print(mf, sdf)

    mmcc = np.mean(MCC)
    sdmcc = np.std(MCC)
    print(mmcc, sdmcc)
 