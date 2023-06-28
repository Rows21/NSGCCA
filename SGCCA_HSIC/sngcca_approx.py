import torch
import math
import numpy as np
import scipy
from scipy.linalg import sqrtm
import itertools
import torch.optim as optim


class SNGCCA_APPROX():
    def __init__(self, device):
        self.device = device
        self.batch_size = 100

    def projL1(self, v, b):
        if b < 0:
            raise ValueError("Radius of L1 ball is negative: {}".format(b))
        if torch.norm(v, 1) < b:
            return v
        u, indices = torch.sort(torch.abs(v), descending=True)
        sv = torch.cumsum(u, dim=0)
        rho = torch.sum(u > (sv - b) / torch.arange(1, len(u) + 1).to(self.device), dim=0)
        theta = torch.max(torch.zeros_like(sv), (sv[rho - 1] - b) / rho)
        w = torch.sign(v) * torch.max(torch.abs(v) - theta, torch.zeros_like(v))
        return w

    def sqdist(self, X, n = None):
        X_2d = X
        if n is not None:
            X_n = X[n]
        else:
            X_n = X_2d
        dist = torch.cdist(X_2d, X_n, p=2) ** 2
        return dist

    def gradf_gauss_SGD(self, K1, cK2, X, sigma, u):
        N = K1.shape[0]
        temp1 = torch.zeros((X.shape[1], X.shape[1])).to(self.device)
        au = sigma

        id1 = torch.sort(torch.rand(N))[1]
        id2 = torch.sort(torch.rand(N))[1]
        N = math.floor(N / 10)

        for i in range(N):
            for j in range(N):
                a = id1[i]
                b = id2[j]
                temp1 += K1[a, b] * cK2[a, b] * torch.ger(X[a, :] - X[b, :], X[a, :] - X[b, :]).to(self.device)

        final = -2 * au * u.t() @ temp1
        return final.t()

    def gene_SGD(self, K1, cK_list, X, a, u):
        res = torch.empty(u.shape[0], 1).to(self.device)
        for i in range((len(cK_list))):
            temp = self.gradf_gauss_SGD(K1, cK_list[i], X, a, u)
            res += temp
        return res

    def rbf_approx(self, X, ind, sigma=None):
        D_mn = torch.sqrt(torch.abs(self.sqdist(X, ind)))

        if sigma is None:  # median heuristic
            sigma = torch.median(D_mn)
        else:
            sigma = torch.tensor(sigma)

        K_mn = torch.exp(- (D_mn ** 2) / (2 * sigma ** 2))
        D_nn = torch.sqrt(torch.abs(self.sqdist(X[ind])))

        K_nn = torch.exp(- (D_nn ** 2) / (2 * sigma ** 2)) + torch.eye(D_nn.shape[0]) * 0.001

        #K_nn_np = np.nan_to_num(K_nn.numpy(), nan=0.001)
        K_nn_sqrt_real = np.nan_to_num(np.real(sqrtm(K_nn)), nan=0.001)

        K_nn_sqrt_inv = np.linalg.inv(K_nn_sqrt_real)

        K_nn_sqrt_inv_tensor = torch.tensor(K_nn_sqrt_inv)
        phi = torch.matmul(K_mn, K_nn_sqrt_inv_tensor)
        return phi, sigma

    def centre_nystrom_kernel(self,phi):
        N = phi.size(0)
        phic = (torch.eye(N) - torch.ones(N) / N) @ phi
        return phic

    def ff_nystrom(self, phic_list):
        N = phic_list[0].shape[0]
        res_nystrom = 0.0
        for items in itertools.combinations(range(len(phic_list)), 2):
            res_nystrom += torch.norm((1 / N) * torch.matmul(phic_list[items[0]].t(), phic_list[items[1]]), p='fro') ** 2
        return res_nystrom

    def fit(self, views, eps, maxit, b,early_stopping=True, patience=10, logging=0):

        ## initial
        batch_size = self.batch_size
        n_view = len(views)

        K_list = []
        a_list = []
        cK_list = []
        u_list = []
        phic_list = []
        ind = torch.randperm(views[0].shape[0])[:batch_size]

        for i, view in enumerate(views):
            v = torch.rand(view.shape[1]).to(self.device)
            umr = torch.reshape(self.projL1(v, b[i]), (view.shape[1], 1))
            u_norm = umr / torch.norm(umr, p=2).to(self.device)

            ## Calculate Kernel
            Xu = view.to(self.device) @ u_norm
            sigma = None
            if sigma is None:
                phi, a = self.rbf_approx(Xu, ind)
            else:
                phi, a = self.rbf_approx(Xu, ind, sigma)
            K = phi.t() @ phi
            phic = self.centre_nystrom_kernel(phi)
            cK = phic.t() @ phic

            ## Save Parameters
            K_list.append(K)
            a_list.append(a)
            cK_list.append(cK)
            u_list.append(umr)
            phic_list.append(phic)

        diff = 99999
        ite = 0
        obj_list = []
        while (diff > eps) & (ite < maxit):
            ite += 1
            for i, view in enumerate(views):
                obj_old = self.ff_nystrom(phic_list)

                ## Calculate Delta and Gamma
                cK_list_SGD = [cK_list[j] for j in range(n_view) if j != i]
                grad = self.gene_SGD(K_list[i], cK_list_SGD, view[ind,:], a_list[i], u_list[i])
                gamma = torch.norm(grad, p=2)

                ## Start Line Search
                chk = 1
                while chk == 1:
                    ## Update New latent variable
                    v_new = torch.reshape(u_list[i] + grad * gamma, (-1,))
                    u_new = torch.reshape(self.projL1(v_new, b[i]), (view.shape[1], 1))
                    u_norm = u_new / torch.norm(u_new, p=2)

                    Xu_new = view.to(self.device) @ u_norm

                    sigma = None
                    if sigma is None:
                        phi_new, a_new = self.rbf_approx(Xu_new,ind)
                    else:
                        phi_new, a_new = self.rbf_approx(Xu_new, ind,sigma)
                    K_new = phi_new.t() @ phi_new

                    phic_new = self.centre_nystrom_kernel(phi_new)
                    cK_new = phic_new.t() @ phic_new

                    ## update phic
                    phic_list_SGD = [phic_list[j] for j in range(n_view) if j != i]
                    phic_list_SGD.append(phic_new)
                    obj_new = self.ff_nystrom(phic_list_SGD)

                    ## Update Params
                    if obj_new > obj_old + 1e-5 * abs(obj_old):
                        chk = 0
                        u_list[i] = u_norm
                        K_list[i] = K_new
                        cK_list[i] = cK_new
                        a_list[i] = a_new
                        phic_list[i] = phic_new
                    else:
                        gamma = gamma / 2
                        if gamma < 1e-7:
                            chk = 0
                obj = obj_new
                ## End Line Search
            diff = abs(obj - obj_old) / abs(obj + obj_old)
            obj_list.append(round(obj.item(), 5))
            if logging == 1:
                print('iter=', ite, "diff=", diff, 'obj=', obj)

            if early_stopping is True:
                if self.EarlyStopping(obj_list, patience=patience):
                    return u_list
        if logging == 2:
            print("diff=", diff, 'obj=', obj)
        return u_list

    def test(self):
        pass

    def EarlyStopping(self, lst, patience=5):
        if len(lst) < patience:
            return False
        last_five = lst[-patience:]
        return len(set(last_five)) == 1

