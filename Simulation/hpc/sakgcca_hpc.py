import numpy as np
import os
import sys
from proposedmodels.sa_kgcca import sakgcca

if __name__ == '__main__':
        print("START")
        mode = int(sys.argv[2])
        N = sys.argv[3]
        num = sys.argv[4]
        tol = sys.argv[5]
        root = '/scratch/rw2867/projects/SNGCCA/SNGCCA/'
        if mode == 1:
                folder = 'Linear/'
        else:
                folder = 'Nonlinear/'
        data_path = root + 'Data/' + folder + '/' + str(N) + '_' + str(tol) + '_' + str(num) + '/'

        U_sum = []
        t = []
        outputs_sum = []

        view1 = np.genfromtxt(data_path + 'data' + str(1) + '_' + str(int(sys.argv[1])-1) + '.csv', delimiter=',')
        view2 = np.genfromtxt(data_path + 'data' + str(2) + '_' + str(int(sys.argv[1])-1) + '.csv', delimiter=',')
        view3 = np.genfromtxt(data_path + 'data' + str(3) + '_' + str(int(sys.argv[1])-1) + '.csv', delimiter=',')
        views = [view1, view2, view3]

        print(f'input views shape :')
        for i, view in enumerate(views):
                print(f'view_{i} :  {view.shape}')

        u, delta, best_alpha, best_s_k = sakgcca(views)

        root = '/scratch/rw2867/projects/SNGCCA/baselines/'
        res = root+'hpcres/sakgcca/' + folder + '/' + str(N) + '_' + str(tol) + '_' + str(num) + '/'
        os.makedirs(res, exist_ok=True)

        name = res+'/t'+str(sys.argv[1])+'.csv'
        np.savetxt(name, np.array([delta]), delimiter=',')

        np.savetxt(res+'/dgcca_u1'+str(sys.argv[1])+'.csv', u[0], delimiter=',')
        np.savetxt(res+'/dgcca_u2'+str(sys.argv[1])+'.csv', u[1], delimiter=',')
        np.savetxt(res+'/dgcca_u3'+str(sys.argv[1])+'.csv', u[2], delimiter=',')
