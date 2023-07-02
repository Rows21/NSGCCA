# 定义一个包含元素的列表
elements = [1, 2, 3, 4]

# 定义一个变量，表示要从列表中取出的循环长度
loop_len = 3

# 使用嵌套的 for 循环和切片，从列表中取出指定长度的子序列
for i in range(len(elements) - loop_len + 1):
    subseq = elements[i:i+loop_len]
    # 处理每个子序列
    print(subseq)

from main import Solver
import pandas as pd
import torch
from synth_data import create_synthData_new
from validation_method import FS_MCC
import numpy as np
from sngcca_approx import SNGCCA_APPROX
import scipy
from sngcca import SGCCA_HSIC
import torch.optim as optim
torch.set_default_tensor_type(torch.DoubleTensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")

SNGCCA_ADAM = SNGCCA_APPROX(device)

x = pd.read_csv("x.csv").values
y = pd.read_csv("y.csv").values
z = pd.read_csv("z.csv").values

views = [torch.tensor(x),torch.tensor(y),torch.tensor(z)]

N = 400
views = create_synthData_new(N, mode=2, F=20)
SGCCA_HSIC = SGCCA_HSIC(device)
a = SGCCA_HSIC.fit(views=views, eps=1e-7, maxit=30, b=(20,20,20),early_stopping=True, patience=10, logging=1)

b = (1,1,1)
SGCCA_HSIC.set_init(views,b)

obj_old = SGCCA_HSIC.ff(SGCCA_HSIC.K_list,SGCCA_HSIC.cK_list)

cK_list_SGD = [SGCCA_HSIC.cK_list[j] for j in range(3) if j != 0]

## Calculate Delta and Gamma
grad = SGCCA_HSIC.gene_SGD(SGCCA_HSIC.K_list[0], cK_list_SGD, views[0], SGCCA_HSIC.a_list[0], SGCCA_HSIC.u_list[0])
gamma = torch.norm(grad, p=2)

