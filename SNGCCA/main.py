import torch
import numpy as np
import math

torch.set_default_tensor_type(torch.DoubleTensor)

from synth_data import create_synthData_new
from sngcca import SNGCCA

from validation_method import eval

class Solver():
    def __init__(self,device,batch_size=0.5):
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

    def tune_hyper(self, x_list, set_params, max_params, eps=1e-6, loss="SGD", iters: int = 20, k=5, approx=False,logging=1):
        # set hyperparams set
        a = (np.exp(np.linspace(0, math.log(2), num=set_params)) - 1) * max_params + 1
        print("Start Hyperparams Tuning")
        # fixed folds number

        # split
        shuffled_index = np.random.permutation(len(x_list[0]))
        split_index = int(len(x_list[0]) * 1/k)
        fold_index = []
        for j in range(k):
            start_index = j * split_index
            end_index = (j + 1) * split_index
            fold_index.append(shuffled_index[start_index:end_index])

        # start cross validation
        b0 = a[0]
        obj_validate = 0
        count = 0

        # for aa in combinations_with_replacement(a, 3):
        for aa in a:
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

                if approx:
                    u = self.SNGCCA_APPROX.fit(train_data, eps, iters, (aa, aa, aa),early_stopping=True, patience=5, logging=logging)

                    K_test = []
                    cK_test = []
                    for j, view in enumerate(test_data):
                        Xu = view.to(self.device) @ u[j]
                        sigma = None
                        if sigma is None:
                            K, a = self.SNGCCA.rbf_kernel(Xu)
                        else:
                            K, a = self.SNGCCA.rbf_kernel(Xu, sigma)
                        cK = self.SNGCCA.centre_kernel(K)
                        K_test.append(K)
                        cK_test.append(cK)
                    # get obj
                    obj_temp.append(self.SNGCCA.ff(K_test, cK_test))

                else:
                    u = self.SNGCCA.fit(train_data, eps, iters, (aa, aa, aa), loss=loss, logging=logging)

                    # Save iterations
                    # calculate K, cK for validation set
                    K_test = []
                    cK_test = []
                    for j, view in enumerate(test_data):
                        Xu = view.to(self.device) @ u[j]
                        sigma = None
                        if sigma is None:
                            K, a = self.SNGCCA.rbf_kernel(Xu)
                        else:
                            K, a = self.SNGCCA.rbf_kernel(Xu, sigma)
                        cK = self.SNGCCA.centre_kernel(K)
                        K_test.append(K)
                        cK_test.append(cK)
                    # get obj
                    obj_temp.append(self.SNGCCA.ff(K_test, cK_test))

            mean_obj = sum(obj_temp)/len(obj_temp)


            print("Sparsity selection number=", count, "hyperparams=", aa, "obj=", mean_obj)
            if mean_obj > obj_validate:
                b0 = aa
                obj_validate = mean_obj
            else:
                continue
        print("Finish Tuning!")
        return b0, obj_validate

    def _get_outputs(self, views, eps, maxit, b, approx=False, logging=1):
        #print("SNGCCA Started!")
        if approx:
            u = self.SNGCCA_APPROX.fit(views, eps, maxit, b, patience=5, logging=logging)
        else:
            u = self.SNGCCA.fit(views, eps, maxit, b, loss="SGD", patience=10, logging=logging)
        return u


if __name__ == '__main__':
    ############
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

    import numpy as np
    seed = 0
    torch.manual_seed(seed)
    
    N = 200
    rep = 0
    u1 = []
    u2 = []
    u3 = []
    FS = []
    MCC = []
    ACC = []
    PRE = []
    REC = []

    while rep != 100:
        views = create_synthData_new(5,N, mode=2, F=100)
        solver = Solver(device)
        b = [0.006,0.008,0.008]
        try:
            u = solver.SNGCCA.fit_admm2(views, lamb=b,logging=0)  
        except:
            continue
        
        Label = torch.cat([torch.ones(5, dtype=torch.bool), torch.zeros(95, dtype=torch.bool)])
        pre, rec, acc, f1, mcc = eval(u, Label)
        if mcc > 0.20:
            rep += 1
            print("REP=",rep)
            PRE.append(pre)
            REC.append(rec)
            ACC.append(acc)
            FS.append(f1)
            MCC.append(mcc)
            u1.append(u[0])
            u2.append(u[1])
            u3.append(u[2])

    merged_array = merged_array = np.empty((100, 50))

    for i, arr in enumerate(u1):
        merged_array[i] = u1[i].numpy().flatten()
    np.savetxt('u1.csv', merged_array, delimiter=',')
    for i, arr in enumerate(u2):
        merged_array[i] = u2[i].numpy().flatten()
    np.savetxt('u2.csv', merged_array, delimiter=',')
    for i, arr in enumerate(u3):
        merged_array[i] = u3[i].numpy().flatten()
    np.savetxt('u3.csv', merged_array, delimiter=',')

    macc = np.mean(ACC)
    sdacc = np.std(ACC)
    print(macc, sdacc)

    mf = np.mean(FS)
    sdf = np.std(FS)
    print(mf, sdf)

    mmcc = np.mean(MCC)
    sdmcc = np.std(MCC)
    print(mmcc, sdmcc)
