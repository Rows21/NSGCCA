from gcca_admm_new import gcca_admm
import numpy as np
import time
from tqdm import tqdm

def cv(x_list, k=5, a = 1e-4):
    # split
    shuffled_index = np.random.permutation(len(x_list[0]))
    split_index = int(len(x_list[0]) * 1/k)
    fold_index = []
    obj_validate = 0
    for j in range(k):
        start_index = j * split_index
        end_index = (j + 1) * split_index
        fold_index.append(shuffled_index[start_index:end_index])
        obj_temp = np.zeros((k, len(x_list)))
    while a < 100:
        
        mean_obj = _cv(x_list, fold_index, shuffled_index, obj_temp, a, k)
        if mean_obj > obj_validate:
            b0 = a
            obj_validate = mean_obj
            a = a * 10
        else:
            a = a * 10
            continue
            
    return b0, obj_validate
    
def _cv(x_list, fold_index, shuffled_index, obj_temp, a, k=5):
    for fold in tqdm(range(k)):

        fold_mask = np.zeros_like(x_list[0], dtype=bool)
        fold_mask[fold_index[fold]] = True

        train_data = []
        test_data = []

        for _, view in enumerate(x_list):
            test = view[fold_index[fold], :]
            test = test - np.mean(test, axis=0)
            test_data.append(test)
            non_fold_index = [num for num in shuffled_index if num not in fold_index[fold]]
            
            train = view[non_fold_index, :]
            train = train - np.mean(train, axis=0)
            train_data.append(train)
                
        model = gcca_admm(train_data,1,mu_x= [a,a,a])
        model.admm()
        
        for i in range(3):
            obj_k = []
            model_test = gcca_admm(test_data,1)
            G = model_test.solve_g()
            for i in range(3):
                obj_k.append(np.linalg.norm(test_data[i] @ model.list_U[i] - G, ord='fro'))
        obj_temp[fold] = np.stack(obj_k)
    mean_obj = np.mean(np.sum(obj_temp, axis=1))
    return mean_obj

if __name__ == "__main__":
    combinations = [
        [100, 30, 5],
        [100, 50, 5],
        [100, 100, 5],
        [100, 200, 5],
        [200, 100, 5],
        [400, 100, 5],
        [100, 100, 10],
        [100, 100, 20]
    ]
    
    for mode in [1, 2]:
        for params in combinations:
            print(params)
            t = []
            u1 = []
            u2 = []
            u3 = []
            
            N = params[0]
            P = params[1]
            S = params[2]
            root = 'E:/res/SNGCCA/SNGCCA/'
            if mode == 1:
                    folder = 'Linear/'
            else:
                    folder = 'Nonlinear/'
            data_path = root + 'Data/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/'

            for r in range(100):
                print(f'Iteration : {r}')
                #views = create_synthData_new(v=5,N=N,mode=1,F=30)
                view1 = np.genfromtxt(data_path + 'data' + str(1) + '_' + str(r) + '.csv', delimiter=',')
                view2 = np.genfromtxt(data_path + 'data' + str(2) + '_' + str(r) + '.csv', delimiter=',')
                view3 = np.genfromtxt(data_path + 'data' + str(3) + '_' + str(r) + '.csv', delimiter=',')
                views = [view1, view2, view3]
                
                if r == 0:
                    constraint,_ = cv(views, k=5)
                #print(f'input views shape :')
                #for i, view in enumerate(views):
                #    print(f'view_{i} :  {view.shape}')
                s = time.time()
                # standardization
                for i, view in enumerate(views):
                    views[i] = view - np.mean(view, axis=0)
                print(f'constraint : {constraint}')
                model = gcca_admm(views,1,mu_x=[constraint,constraint,constraint])
                model.admm()
                e = time.time()
                t.append(e-s)
                u1.append(np.real(model.list_U[0]))
                u2.append(np.real(model.list_U[1]))
                u3.append(np.real(model.list_U[2]))
                    
            merged_array = merged_array = np.empty((100,P))
            path = 'E:/res/SNGCCA/SNGCCA/Simulation/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/'
            
            for i, arr in enumerate(u1):
                merged_array[i] = u1[i].flatten()
            np.savetxt(path + 'sgcca_admm_u1.csv', merged_array, delimiter=',')
            for i, arr in enumerate(u2):
                merged_array[i] = u2[i].flatten()
            np.savetxt(path + 'sgcca_admm_u2.csv', merged_array, delimiter=',')
            for i, arr in enumerate(u3):
                merged_array[i] = u3[i].flatten()
            np.savetxt(path + 'sgcca_admm_u3.csv', merged_array, delimiter=',')
            #np.savetxt(path + 'sgcca_admm_t.csv', t, delimiter=',')