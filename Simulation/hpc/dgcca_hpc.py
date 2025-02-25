## similar to github.com/Michaelvll/DeepCCA main
import os
import sys
import numpy as np
import torch.utils
import torch.utils.data
from linear_gcca import linear_gcca
from models import DeepGCCA
# from utils import load_data, svm_classify
import time
import numpy as np

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
learning_rate = 1e-4
#[1e-5,1e-4,1e-3,1e-2,1e-1]
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
mode = sys.argv[2]
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
for i in range(int(tol)):
    view_res1 = view1.copy()
    view_res2 = view2.copy()
    view_res3 = view3.copy()
    view_res1[:,i] = np.mean(view_res1[:,i])
    test = [torch.tensor(view_res1), torch.tensor(view2), torch.tensor(view3)]
    _ , loss, outputs_def = solver.test(test, apply_linear_gcca)
    A[i] = loss - loss0
    view_res2[:,i] = np.mean(view_res2[:,i])
    test = [torch.tensor(view1), torch.tensor(view_res2), torch.tensor(view3)]
    _ , loss, outputs_def = solver.test(test, apply_linear_gcca)
    B[i] = loss - loss0
    view_res3[:,i] = np.mean(view_res3[:,i])
    test = [torch.tensor(view1), torch.tensor(view2), torch.tensor(view_res3)]
    _ , loss, outputs_def = solver.test(test, apply_linear_gcca)
    C[i] = loss - loss0
    
end_time = time.time()
t = end_time - start_time


res = '/scratch/rw2867/projects/SNGCCA/baselines/'
os.makedirs(res+'hpcres/u/', exist_ok=True)
name = res+'hpcres/u/t'+str(sys.argv[1])+'.csv'
np.savetxt(name, np.array([t]), delimiter=',')

np.savetxt(res+'hpcres/u/'+'dgcca_u1'+str(sys.argv[1])+'.csv', A, delimiter=',')
np.savetxt(res+'hpcres/u/'+'dgcca_u2'+str(sys.argv[1])+'.csv', B, delimiter=',')
np.savetxt(res+'hpcres/u/'+'dgcca_u3'+str(sys.argv[1])+'.csv', C, delimiter=',')