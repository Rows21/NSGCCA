## similar to github.com/Michaelvll/DeepCCA main
import torch
import torch.nn as nn
import numpy as np
from linear_gcca import linear_gcca
from synth_data import create_synthData_new
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from models import DeepGCCA
# from utils import load_data, svm_classify
import time
import logging
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import torch.nn as nn

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
learning_rate = 5*1e-2
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

# 1
N = 100
U_sum = []
u1 = np.empty((100, 200))
u2 = np.empty((100, 200))
u3 = np.empty((100, 200))
outputs_sum = []
test = []
for i in range(3):
    testm = torch.eye(200)
    test.append(testm)

import pandas as pd
for r in range(100):
    views = create_synthData_new(v=5,N=N,mode=2,F=200)
    print(f'input views shape :')
    for i, view in enumerate(views):
        print(f'view_{i} :  {view.shape}')
        view = view.to(device)


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
    _ , _, outputs_def = solver.test(test, apply_linear_gcca)

    A = outputs_def[0]
    B = outputs_def[1]
    C = outputs_def[2]
    U = [A,B,C]

    A = outputs_def[0]
    B = outputs_def[1]
    C = outputs_def[2]

    os = [A,B,C]
    u1[r] = A.numpy().flatten()
    u2[r] = B.numpy().flatten()
    u3[r] = C.numpy().flatten()
    outputs_sum.append(os)
    
np.savetxt('dgcca_u1.csv', u1, delimiter=',')
np.savetxt('dgcca_u2.csv', u2, delimiter=',')
np.savetxt('dgcca_u3.csv', u3, delimiter=',')
#variables = pd.DataFrame(U_sum)
#results = pd.DataFrame(results_sum)