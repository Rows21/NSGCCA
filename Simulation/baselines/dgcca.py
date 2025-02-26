## similar to github.com/Michaelvll/DeepCCA main
import torch
import torch.nn as nn
import numpy as np
from linear_gcca import linear_gcca
#from Simulation.baselines.synth_data import create_synthData_new
from models import DeepGCCA
# from utils import load_data, svm_classify
import time

torch.set_default_tensor_type(torch.DoubleTensor)
from main import Solver

############
# Parameters Section

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")

# the path to save the final learned features
save_name = './DGCCA.model'

# the size of the new space learned by the model (number of the new features)
outdim_size = 1

# number of layers with nodes in each one
layer_sizes1 = [256, 512, 128, outdim_size]
layer_sizes2 = [256, 512, 128, outdim_size]
layer_sizes3 = [256, 512, 128, outdim_size]
layer_sizes_list = [layer_sizes1, layer_sizes2, layer_sizes3] 

# the parameters for training the network
learning_rate = 10
epoch_num = 200
batch_size = 100

# the regularization parameter of the network
# seems necessary to avoid the gradient exploding especially when non-saturating activations are used
reg_par = 1e-4

# specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
# if one option does not work for a network or dataset, try the other one
use_all_singular_values = False

apply_linear_gcca = True
# end of parameters section

# 1
mode = 2
N = 100
num = 20
tol = 100
root = 'E:/res/SNGCCA/SNGCCA/'
if mode == 1:
        folder = 'Linear/'
else:
        folder = 'Nonlinear/'
data_path = root + 'Data/' + folder + '/' + str(N) + '_' + str(tol) + '_' + str(num) + '/'
U_sum = []
u1 = np.zeros((N, tol))
u2 = np.zeros((N, tol))
u3 = np.zeros((N, tol))
t = []
outputs_sum = []

import pandas as pd

for r in range(100):
    #views = create_synthData_new(v=5,N=N,mode=1,F=30)
    view1 = np.genfromtxt(data_path + 'data' + str(1) + '_' + str(r) + '.csv', delimiter=',')
    view2 = np.genfromtxt(data_path + 'data' + str(2) + '_' + str(r) + '.csv', delimiter=',')
    view3 = np.genfromtxt(data_path + 'data' + str(3) + '_' + str(r) + '.csv', delimiter=',')
    views = [torch.tensor(view1), torch.tensor(view2), torch.tensor(view3)]
    print(f'input views shape :')
    for i, view in enumerate(views):
        print(f'view_{i} :  {view.shape}')
        view = view.to(device)
    start_time = time.time()
    # size of the input for view 1 and view 2
    input_shape_list = [view.shape[-1] for view in views]

    # Building, training, and producing the new features by DCCA
    model = DeepGCCA(layer_sizes_list, input_shape_list, outdim_size,
                             use_all_singular_values, device=device).double()
    l_gcca = None
    if apply_linear_gcca:
        l_gcca = linear_gcca
    solver = Solver(model, l_gcca, outdim_size, epoch_num, batch_size,
                    learning_rate, reg_par, device=device)
    # train1, train2 = data1[0][0], data2[0][0]
    # val1, val2 = data1[1][0], data2[1][0]
    # test1, test2 = data1[2][0], data2[2][0]

    solver.fit(views, checkpoint=save_name)
    
    # TODO: Save l_gcca model if needed
    A = np.zeros((int(tol),1))
    B = np.zeros((int(tol),1))
    C = np.zeros((int(tol),1))
    loss0 = solver.test([torch.tensor(view1), torch.tensor(view2), torch.tensor(view3)], apply_linear_gcca)[1]
    for j in range(int(tol)):
        view_res1 = view1.copy()
        view_res2 = view2.copy()
        view_res3 = view3.copy()
        view_res1[:,j] = np.mean(view_res1[:,j])
        test = [torch.tensor(view_res1), torch.tensor(view2), torch.tensor(view3)]
        _ , loss, outputs_def = solver.test(test, apply_linear_gcca)
        A[j] = loss - loss0
        view_res2[:,j] = np.mean(view_res2[:,j])
        test = [torch.tensor(view1), torch.tensor(view_res2), torch.tensor(view3)]
        _ , loss, outputs_def = solver.test(test, apply_linear_gcca)
        B[j] = loss - loss0
        view_res3[:,i] = np.mean(view_res3[:,j])
        test = [torch.tensor(view1), torch.tensor(view2), torch.tensor(view_res3)]
        _ , loss, outputs_def = solver.test(test, apply_linear_gcca)
        C[j] = loss - loss0
    u1[r] = A.reshape(-1)
    u2[r] = B.reshape(-1)
    u3[r] = C.reshape(-1)
    end_time = time.time()
    t.append(end_time - start_time)
    


np.savetxt(data_path.replace("Data","Simulation")+'dgcca_u1.csv', u1, delimiter=',')
np.savetxt(data_path.replace("Data","Simulation")+'dgcca_u2.csv', u2, delimiter=',')
np.savetxt(data_path.replace("Data","Simulation")+'dgcca_u3.csv', u3, delimiter=',')
np.savetxt(data_path.replace("Data","Simulation")+'dgcca_t.csv', t, delimiter=',')
#variables = pd.DataFrame(U_sum)
#results = pd.DataFrame(results_sum)