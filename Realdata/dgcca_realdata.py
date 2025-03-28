## similar to github.com/Michaelvll/DeepCCA main
import torch
import torch.nn as nn
import numpy as np
from linear_gcca import linear_gcca
#from synth_data import create_synthData_new
from models import DeepGCCA
# from utils import load_data, svm_classify

from utils import _get_tcga_new

import numpy as np

import pandas as pd

torch.set_default_tensor_type(torch.DoubleTensor)
from main import Solver
from loss_objectives import new_loss

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
learning_rate = 5
epoch_num = 200
batch_size = 400

# the regularization parameter of the network
# seems necessary to avoid the gradient exploding especially when non-saturating activations are used
reg_par = 1e-5

# specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
# if one option does not work for a network or dataset, try the other one
use_all_singular_values = False

apply_linear_gcca = True
# end of parameters section

# Hyper Params Section
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")

root = 'E:/res/SNGCCA/'
views = _get_tcga_new(root)

view1 = torch.tensor(views[0])
view2 = torch.tensor(views[1])
view3 = torch.tensor(views[2])

views = [view1,view2,view3]

U_sum = []
outputs_sum = []

A = np.zeros((view1.shape[-1],1))
B = np.zeros((view2.shape[-1],1))
C = np.zeros((view3.shape[-1],1))

print(f'input views shape :')
for i, view in enumerate(views):
    print(f'view_{i} :  {view.shape}')
    view = view.to("cpu")

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

solver.fit(views, checkpoint=save_name)

loss0 = solver.test([torch.tensor(view1), torch.tensor(view2), torch.tensor(view3)], apply_linear_gcca)[1]
for j in range(view1.shape[-1]):
    view_res1 = view1.numpy().copy()
    view_res1[:,j] = np.mean(view_res1[:,j])
    test = [torch.tensor(view_res1), torch.tensor(view2), torch.tensor(view3)]
    _ , loss, outputs_def = solver.test(test, apply_linear_gcca)
    A[j] = loss - loss0
    
for j in range(view2.shape[-1]):
    view_res2 = view2.numpy().copy()
    view_res2[:,j] = np.mean(view_res2[:,j])
    test = [torch.tensor(view1), torch.tensor(view_res2), torch.tensor(view3)]
    _ , loss, outputs_def = solver.test(test, apply_linear_gcca)
    B[j] = loss - loss0
    
for j in range(view3.shape[-1]):
    view_res3 = view3.numpy().copy()
    view_res3[:,j] = np.mean(view_res3[:,j])
    test = [torch.tensor(view1), torch.tensor(view2), torch.tensor(view_res3)]
    _ , loss, outputs_def = solver.test(test, apply_linear_gcca)
    C[j] = loss - loss0

df_u1_total = pd.DataFrame()
df_u2_total = pd.DataFrame()
df_u3_total = pd.DataFrame()
df_u1 = pd.DataFrame(A, columns=['u1_' + str(0)])
df_u1_total = pd.concat([df_u1_total, df_u1], axis=1)
df_u2 = pd.DataFrame(B, columns=['u2_' + str(0)])
df_u2_total = pd.concat([df_u2_total, df_u2], axis=1)
df_u3 = pd.DataFrame(C, columns=['u3_' + str(0)]) 
df_u3_total = pd.concat([df_u3_total, df_u3], axis=1)

#df_u1_total.to_csv('U1.csv')
#df_u2_total.to_csv('U2.csv')
df_u3_total.to_csv('U3.csv')