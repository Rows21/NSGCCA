import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import pandas as pd
from synth_data import create_synthData,create_synthData_new
from sgcca_hsic import projL1,rbf_kernel,centre_kernel,f,gene_SGD,gradf_gauss_SGD
from sgcca_hsic import SGCCA_HSIC
N = 400
views = create_synthData_new(N,mode=1,F=20)
#print(f'input views shape :')
for i, view in enumerate(views):
    #print(f'view_{i} :  {view.shape}')
    view = view.to("cpu")

a = SGCCA_HSIC(views)


# size of the input for view 1 and view 2
input_shape_list = [view.shape[-1] for view in views]

b = 3

K_list = []
a_list = []
cK_list = []
u_list = []

for i,view in enumerate(views):
    v = torch.rand(20)
    umr = torch.reshape(projL1(v,b),(20,1))
    Xu = view @ umr

    sigma = None
    if sigma is None:
        K,a = rbf_kernel(Xu)
    else:
        K,a = rbf_kernel(Xu, sigma)
    cK = centre_kernel(K)


    K_list.append(K)
    a_list.append(a)
    cK_list.append(cK)
    u_list.append(umr)


diff = 1.00
ite = 0
obj_old = f(K_list[0], K_list[1], K_list[2])

while (diff > 1e-7) & (ite < 20):
    print("iter=",ite)
    ite += 1

    for i,view in enumerate(views):
        print(i)
        cK_list_SGD = [cK_list[j] for j in range(3) if j != i]
        grad = gene_SGD(K_list[i], cK_list, view, a_list[i], u_list[i])
        gamma = torch.norm(grad)
        chk = 1
        while chk == 1:
            v_new = torch.reshape(u_list[i] + grad * gamma,(-1,))
            u_new = torch.reshape(projL1(v_new,b),(20,1))
            Xu = view @ u_new

            if sigma is None:
                K_new, a_new = rbf_kernel(Xu)
            else:
                K_new, a_new = rbf_kernel(Xu, sigma)
            cK_new = centre_kernel(K_new)

            K_list_SGD = [K_list[j] for j in range(3) if j != i]
            K_list_SGD.append(K_new)
            obj_new = f(K_list_SGD[0],K_list_SGD[1],K_list_SGD[2])
            if obj_new > obj_old + 1e-4*abs(obj_old):
                chk = 0
                K_list[i] = K_new
                a_list[i] = a_new
                cK_list[i] = cK_new
                u_list[i] = u_new
            else:
                gamma = gamma/2
            print(obj_old)
            obj_old = obj_new




