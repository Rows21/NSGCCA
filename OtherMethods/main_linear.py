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
from validation_method import swiss_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
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
epoch_num = 2000
batch_size = 800

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

Exp_label = pd.read_csv('C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/Exp664_genes.txt', sep='\t',header = None)
Exp_list = Exp_label.iloc[:, 0].values.tolist()
Exp = pd.DataFrame(np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/Exp664.txt").T,columns = Exp_label)

Meth_label = pd.read_csv('C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/Meth664_probes.txt', sep='\t',header = None)
Meth_list = Meth_label.iloc[:, 0].values.tolist()
Meth = pd.DataFrame(np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/Meth664.txt").T,columns = Meth_label)

miRNA_label = pd.read_csv('C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/miRNA664_miRNA.txt', sep='\t',header = None)
miRNA_list = miRNA_label.iloc[:, 0].values.tolist()
miRNA = pd.DataFrame(np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/miRNA664.txt").T,columns = miRNA_label)

y = pd.read_csv('C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/PAM50label664.txt',header = None)

Exp_value = np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/Exp664.txt")
Meth_value = np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/Meth664.txt")
miRNA_value = np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/miRNA664.txt")

views = [torch.tensor(Exp_value).T,torch.tensor(Meth_value).T,torch.tensor(miRNA_value).T]

U_sum = []
outputs_sum = []
test = []
for i in range(3):
    testm = torch.eye(views[i].size()[1])
    test.append(testm)

print(f'input views shape :')
for i, view in enumerate(views):
    print(f'view_{i} :  {view.shape}')
    view = view.to("cpu")

for iii in range(1):

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
    
    def _get_outputs(x_list):
        with torch.no_grad():
            solver.model.eval()
            data_size = 664
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=solver.batch_size, drop_last=False))
            losses = []
            outputs_list = []
            for batch_idx in batch_idxs:
                batch_x = x_list
                outputs = solver.model(batch_x)
                outputs_list.append([o_j.clone().detach() for o_j in outputs])
        outputs_final = []
        for i in range(len(x_list)):
            view = []
            for j in range(len(outputs_list)):
                view.append(outputs_list[j][i].clone().detach())
            view = torch.cat(view, dim=0)
            outputs_final.append(view)
        return losses, outputs_final


    solver.fit(views, checkpoint=save_name)

    _,outputs_list = _get_outputs(test)
    outputs_def = solver.linear_gcca.test(outputs_list)

    Exp_df = pd.DataFrame({'Name': Exp_list, 'Score': outputs_def[0].reshape(-1).tolist()})
    Meth_df = pd.DataFrame({'Name': Meth_list, 'Score': outputs_def[1].reshape(-1).tolist()})
    miRNA_df = pd.DataFrame({'Name': miRNA_list, 'Score': outputs_def[2].reshape(-1).tolist()})

    Exp_df = pd.concat([Exp_df,pd.DataFrame(Exp_value)],axis=1)
    Meth_df = pd.concat([Meth_df,pd.DataFrame(Meth_value)],axis=1)
    miRNA_df = pd.concat([miRNA_df,pd.DataFrame(miRNA_value)],axis=1)

    swiss = swiss_score(Exp_df.iloc[:,2:].T, y[0])
    db = davies_bouldin_score(Exp_df.iloc[:,2:].T, y[0])
    ss = silhouette_score(Exp_df.iloc[:,2:].T, y[0])
    ch = calinski_harabasz_score(Exp_df.iloc[:,2:].T, y[0])
    print(swiss, db, ss, ch)

    Filter_Exp = Exp_df[Exp_df['Score'] > np.mean(Exp_df['Score'])]
    Filter_Meth = Meth_df[Meth_df['Score'] > np.mean(Meth_df['Score'])]
    Filter_miRNA = miRNA_df[miRNA_df['Score'] > np.mean(miRNA_df['Score'])]

    print(len(Filter_Exp), len(Filter_Meth), len(Filter_miRNA))

    swiss = swiss_score(Filter_Exp.iloc[:,2:].T, y[0])
    db = davies_bouldin_score(Filter_Exp.iloc[:,2:].T, y[0])
    ss = silhouette_score(Filter_Exp.iloc[:,2:].T, y[0])
    ch = calinski_harabasz_score(Filter_Exp.iloc[:,2:].T, y[0])
    print(swiss, db, ss, ch)
