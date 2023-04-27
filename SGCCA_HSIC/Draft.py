import torch
import math
def projL1(v, b):
    if b < 0:
        raise ValueError("Radius of L1 ball is negative: {}".format(b))
    if torch.norm(v, 1) < b:
        return v
    u, indices = torch.sort(torch.abs(v), descending=True)
    sv = torch.cumsum(u, dim=0)
    rho = torch.sum(u > (sv - b) / torch.arange(1, len(u) + 1), dim=0)
    theta = torch.max(torch.zeros_like(sv), (sv[rho - 1] - b) / rho)
    w = torch.sign(v) * torch.max(torch.abs(v) - theta, torch.zeros_like(v))
    return w

def sqdist(X1, X2):
    n1 = X1.shape[1]
    n2 = X2.shape[1]
    D = torch.sum(X1**2, dim=0).reshape(-1, 1).repeat(1, n2) + torch.sum(X2**2, dim=0).reshape(1, -1).repeat(n1, 1) - 2 * torch.mm(X1.T, X2)
    return D

def gradf_gauss_SGD(K1, cK2, X, a, u):
    N = K1.shape[0]
    temp1 = torch.zeros((X.shape[1], X.shape[1]))
    au = a

    id1 = torch.sort(torch.rand(N))[1]
    id2 = torch.sort(torch.rand(N))[1]
    N = math.floor(N / 10)

    for i in range(N):
        for j in range(N):
            a = id1[i]
            b = id2[j]
            temp1 += K1[a, b] * cK2[a, b] * torch.ger(X[a,:] - X[b,:], X[a,:] - X[b,:])
    final = -2 * au * u.t() @ temp1
    return final.t()

def rbf_kernel(X, sigma=None):
    # 计算距离矩阵
    D = torch.sqrt(torch.abs(sqdist(X.t(),X.t())))

    if sigma is None:
        # 中位数启发式法估计 sigma
        sigma = torch.median(D)

    # 计算核矩阵
    K = torch.exp(- (D ** 2) / (2 * sigma ** 2))
    return K, sigma

def centre_kernel(K):
    return K + torch.mean(K) - (torch.mean(K, dim=0).reshape((1, -1)) + torch.mean(K, dim=1).reshape((-1, 1)))

def f(K1,K2,K3):
    N = K1.shape[0]
    cK2 = centre_kernel(K2)
    cK3 = centre_kernel(K3)

    res = torch.trace(K1 @ cK2) / ((N - 1) ** 2) + torch.trace(K1 @ cK3) / ((N - 1) ** 2) + torch.trace(K2 @ cK3) / ((N - 1) ** 2)

    return res

def gene_SGD(K1, cK_list, X, a, u):
    res = torch.empty(20, 1)
    for i in range((len(cK_list))):
        temp = gradf_gauss_SGD(K1, cK_list[i], X, a, u)
        res += temp
    return res

# size of the input for view 1 and view 2
input_shape_list = [view.shape[-1] for view in views]

## Initialization
b = 3

K_list = []
a_list = []
cK_list = []
u_list = []

for i,view in enumerate(views):
    v = torch.rand(20)
    umr = torch.reshape(projL1(v,b),(20,1))
    u_norm = umr/torch.norm(umr, p=2)

    ## Calculate Kernel
    Xu = view @ u_norm
    sigma = None
    if sigma is None:
        K,a = rbf_kernel(Xu)
    else:
        K,a = rbf_kernel(Xu, sigma)
    cK = centre_kernel(K)

    ## Save Parameters
    K_list.append(K)
    a_list.append(a)
    cK_list.append(cK)
    u_list.append(u_norm)

diff = 1.00
ite = 0

while (diff > 1e-7) & (ite < 40):
    ite += 1
    for i, view in enumerate(views):
        obj_old = f(K_list[0], K_list[1], K_list[2])
        cK_list_SGD = [cK_list[j] for j in range(3) if j != i]

        ## Calculate Delta and Gamma
        grad = gene_SGD(K_list[i], cK_list_SGD, view, a_list[i], u_list[i])
        gamma = torch.norm(grad, p=2)

        ## Start Line Search
        chk = 1
        while chk == 1:
            ## Update New latent variable
            v_new = torch.reshape(u_list[i] + grad * gamma, (-1,))
            u_new = torch.reshape(projL1(v_new, b), (20, 1))
            u_norm = u_new / torch.norm(u_new, p=2)

            Xu_new = view @ u_norm
            if sigma is None:
                K_new, a_new = rbf_kernel(Xu_new)
            else:
                K_new, a_new = rbf_kernel(Xu_new, sigma)
            cK_new = centre_kernel(K_new)

            K_list_SGD = [K_list[j] for j in range(3) if j != i]
            K_list_SGD.append(K_new)
            obj_new = f(K_list_SGD[0], K_list_SGD[1], K_list_SGD[2])
            #print(obj_new,"OBJ_NEW",i,"I")

            ## Update Params
            if obj_new > obj_old + 1e-4*abs(obj_old):
                chk = 0
                u_list[i] = u_norm
                K_list[i] = K_new
                cK_list[i] = cK_new
                a_list[i] = a_new
                obj = obj_new
            else:
                gamma = gamma/2
                if gamma < 1e-7:
                    chk = 0
        obj = obj_new

    diff = abs(obj - obj_old) / abs(obj + obj_old)
    print('iter=',ite,"diff=",diff,'obj=',obj)

