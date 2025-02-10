from gcca_admm_new import gcca_admm
import numpy as np
import time

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
            root = 'F:/SNGCCA/SNGCCA/'
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
                #print(f'input views shape :')
                #for i, view in enumerate(views):
                #    print(f'view_{i} :  {view.shape}')
                s = time.time()
                model = gcca_admm(views,1)
                model.admm()
                e = time.time()
                t.append(e-s)
                u1.append(np.real(model.list_U[0]))
                u2.append(np.real(model.list_U[1]))
                u3.append(np.real(model.list_U[2]))
                    
            merged_array = merged_array = np.empty((100,P))
            path = 'F:/SNGCCA/SNGCCA/Simulation/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/'
            
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