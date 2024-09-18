import torch
import itertools
import numpy as np
from tqdm import tqdm
from scipy.sparse.linalg import eigs

class SNGCCA():
    def __init__(self, device):
        self.K_list = []
        self.a_list = []
        self.cK_list = []
        self.u_list = []
        self.device = device

        self.Momentum_V: list = [None] * 3
        self.Adam_V: list = [None] * 3
        self.Adam_M: list = [None] * 3

    def sqdist(self, X1, X2):
        n1 = X1.shape[1]
        n2 = X2.shape[1]
        D = torch.sum(X1 ** 2, dim=0).reshape(-1, 1).repeat(1, n2) + torch.sum(X2 ** 2, dim=0).reshape(1, -1).repeat(n1,
            1) - 2 * torch.mm(X1.T, X2)
        return D

    def rbf_kx(self, x, Pi, sigma=None):
        n = x.shape[0]
        Kx = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                Kx[i, j] = torch.trace(Pi @ torch.ger(x[i] - x[j], x[i] - x[j]))
                
        #Kx = torch.exp(- (Kx ** 2) / 2)
        return torch.exp(- (Kx ** 2) / 2)
    
    def outer(self, x, y):
        return torch.ger(x, y)
    
    def rbf_kl(self, sum_K):
        n = len(sum_K)
        I_n = torch.eye(n)
        H = I_n - torch.outer(torch.ones(n), torch.ones(n)) / n
        return H @ sum_K @ H
    
    def Z(self, x, p):
        n = x.shape[0]
        Z_F2 = torch.zeros(n, n)
        Z_FT = torch.zeros(n, n)
        Z = torch.zeros(n, n, p, p)
        for i in range(n):
            for j in range(n):
                Z[i,j,:,:] = torch.ger(x[i] - x[j], x[i] - x[j])
                Z_F2[i, j] = torch.norm(Z[i,j,:,:], p='fro') ** 2
                Z_FT[i, j] = torch.norm(Z[i,j,:,:], p='fro') * torch.trace(Z[i,j,:,:])
        
        return {"Z_ij":Z, "Z_Fnorm2":Z_F2, "Z_FT":Z_FT} # "Z_Fnorm22":Z_F22
    
    def delta_Pi(self, Coeft, Z):
        n = Coeft.shape[0]
        p = Z.shape[-1]
        temp = torch.zeros(p, p)
        for i in range(n):
            for j in range(n):
                temp += Coeft[i,j] * Z[i, j, :, :]
        
        return temp /(2 * n ** 2)
    
    def rbf_kernel(self, X, sigma=None):
        # dist
        D = torch.sqrt(torch.abs(self.sqdist(X.t(), X.t())))

        if sigma is None:
            # median sigma
            sigma = torch.median(D)

        # kernel
        K = torch.exp(- (D ** 2) / (2 * sigma ** 2))
        return K, sigma

    def centre_kernel(self, K):
        return K + torch.mean(K) - (torch.mean(K, dim=0).reshape((1, -1)) + torch.mean(K, dim=1).reshape((-1, 1)))

    def projL1(self, v, b):
        #if b < 0:
        #    raise ValueError("Radius of L1 ball is negative: {}".format(b))
        #if torch.sum(torch.min(torch.tensor(1.0), torch.max(v, torch.zeros_like(v)))) <= b:
        #    return v
        u, indices = torch.sort(v, descending=True)
        u = u.flip(dims=(0,))
        sv = torch.cumsum(u, dim=0)
        #rho = torch.sum(u > (sv - b) / torch.arange(1, len(u) + 1), dim=0)
        rho = torch.max(torch.max(u - (sv - b) / torch.arange(1, len(u) + 1).reshape(-1,1),torch.zeros_like(sv)),0)[1].item() + 1
        #theta = torch.max(torch.max(torch.zeros_like(sv), (sv[rho - 1] - b) / (rho.reshape(-1,1))))
        theta = sv[rho - 1] - b / rho#torch.max(torch.tensor(0),((sv[rho - 1] - b) / rho))
        w = torch.max(u - theta, torch.zeros_like(v))
        return w
    
    def __FantopeProjection(self, W):
        #temp = (W + W.T)/2
        
        D, V = torch.linalg.eigh(W)
        d = D.reshape(-1,1)
        d_final = self.projL1(d, 1)
        H = V @ torch.diag(d_final.squeeze()) @ V.T
        return H
    
    def _FantopeProjection(self, W):
        # This code is to solve the projection problem onto the fantope constraint
        # min_H || H - W ||
        # s.t. || H || _{*} <= K, || H || _{sp} <= 1
        temp = (W + W.T)/2
        
        D, V = torch.linalg.eigh(temp)
        #D,V = eigs(temp.numpy())
        d = D.reshape(-1,1)

        if torch.sum(torch.min(torch.tensor(1.0), torch.max(d, torch.zeros_like(d)))) <= 0:
            gamma = 0
        else:
            knots = torch.unique(torch.cat([(d - 1), d]))
            knots = torch.sort(knots, descending=True).values

            temp = torch.where(torch.sum(torch.min(torch.tensor(1.0), torch.max(D - knots.unsqueeze(1), torch.tensor(0.0))), dim=1) <= 1)
            temp = temp[0]
            lentemp = temp[-1]
            #if len(lentemp) != 0:
            a = knots[lentemp]
            b = knots[lentemp + 1]
            fa = torch.sum(torch.min(torch.tensor(1.0),torch.max(d - a, torch.tensor(0.0))))
            fb = torch.sum(torch.min(torch.tensor(1.0),torch.max(d - b, torch.tensor(0.0))))
            gamma = a + (b - a) * (1 - fa) / (fb - fa)
            #else:
            #  gamma = 0

        d_final = torch.min(torch.tensor(1.0), torch.max(d - gamma, torch.tensor(0.0)))
        H = V @ torch.diag(d_final.squeeze()) @ V.T
        print(sum(d_final))
        return H

    def fit_admm2(self, views, lamb, logging=1):
        n_views = len(views)
        self.K_list = []
        self.a_list = []
        self.cK_list = []
        self.u_list = [None] * n_views
        rho = 1
        # p = 150

        self.covx_list = []
        self.y_lab = []
        self.sqcovx_list = []
        self.tau_list = []
        self.H_list = []
        self.Gamma_list = []
        self.Pi_list = []
        self.L_list = []
        self.Z_list = []
        for i, view in enumerate(views):

            n, p = view.shape
            # set sqcovx
            covx = torch.cov(view.T)
            self.covx_list.append(covx)

            eigval, eigvec = torch.linalg.eigh((covx + covx.T) / 2)
            # covx max eigval
            eigenvalues = torch.real(eigval)
            eigenvalues = torch.where(eigenvalues < 0, torch.zeros_like(eigenvalues), eigenvalues)
            sqrt_eigenvalues = torch.sqrt(eigenvalues)
            # eigvec * sqrt(max(eigval, 0)) * eigvec'
            sqcovx = eigvec @ torch.diag(sqrt_eigenvalues) @ eigvec.t()
            self.sqcovx_list.append(sqcovx)

            # set tau
            tau = 4 * rho * torch.max(eigenvalues) ** 2
            self.tau_list.append(tau)

            # set init u
            u = torch.ones((p, 1))
            self.u_list[i] = u
            initPi = u @ u.t() / (u.t() @ covx @ u)
            Pi = initPi#/torch.norm(initPi)
            self.Pi_list.append(Pi)

            Kx = self.rbf_kx(view, Pi)
            self.K_list.append(Kx)

            Z = self.Z(view, p)
            self.Z_list.append(Z)

            H = self.sqcovx_list[i] * self.Pi_list[i] * self.sqcovx_list[i]
            self.H_list.append(H)
            Gamma = torch.zeros((p, p))
            self.Gamma_list.append(Gamma* (n_views - 1))

        outer_maxiter = 2000
        outer_tol = 1e-5
        inner_maxiter = 200
        inner_tol = 1e-3
        outer_error = 1
        diff_list = [999] * n_views
        criterion = 1e-4

        progress_bar = tqdm(total=outer_maxiter, ncols=200)
        
        for outer_iter in range(outer_maxiter):
            for i, view in enumerate(views):
                
                Kl_grad = sum([self.K_list[j] for j in range(len(views)) if j != i])
                K_tilde = self.rbf_kl(Kl_grad) # H @ Kl_grad @ H
                Coeft = self.K_list[i] * K_tilde

                dF = self.delta_Pi(Coeft, self.Z_list[i]['Z_ij'])
                L = torch.sum(torch.abs(K_tilde) * self.Z_list[i]['Z_FT'])/ (2 * n ** 2)
                Pi = self.Pi_list[i]

                a = Pi - dF / L
                Pi_pre = Pi
                inner_error = 1
                inner_iter = 0

                sqcovx = self.sqcovx_list[i]
                covx = self.covx_list[i]
                tau = self.tau_list[i]

                H = self.H_list[i]
                Gamma = self.Gamma_list[i]
                #print(torch.trace(sqcovx @ Pi @ sqcovx) )
                while (inner_iter <= inner_maxiter) & (inner_error > inner_tol):
                    #print(inner_iter)
                    #print("Pi",torch.trace(sqcovx @ Pi @ sqcovx) )
                    temp = Pi-(rho/tau) * covx @ Pi @ covx + (rho/tau) * sqcovx @ (H-Gamma) @ sqcovx
                    temp = tau/(tau+L)*temp+L/(tau+L) * a
                    
                    Pi = torch.max((temp - lamb[i]/(L + tau)), torch.zeros(temp.size())) * torch.sign(temp)                                
                    H = self.__FantopeProjection(sqcovx @ Pi @ sqcovx + Gamma)
                    #print("H",torch.trace(H) )
                    #if torch.trace(Pi) >= 1e+5 or torch.trace(Pi) <= -1e+5:
                    #    print("H",torch.trace(H) )
                    #    return self.u_list
                    Gamma = Gamma + sqcovx @ Pi @ sqcovx - H
                    
                    inner_error = torch.max(torch.max(torch.abs(sqcovx @ Pi @ sqcovx - H)), torch.max(torch.abs(Pi-Pi_pre)))
                    #inner_error = torch.norm(sqcovx @ Pi @ sqcovx - H, 'fro')
                    inner_iter = inner_iter + 1

                #print(sum(sum(Pi)))
                _, u_new_meta = torch.linalg.eigh((Pi + Pi.T) / 2)
                l1_norms = torch.sum(torch.abs(u_new_meta),dim=0)
                top_indices = np.argsort(l1_norms)[-1:]
                u_new_meta = u_new_meta[:, top_indices]
                u_new = u_new_meta

                self.u_list[i] = u_new
                #print(torch.norm(u_new))
                self.Pi_list[i] = Pi #/ torch.norm(Pi)
                self.H_list[i] = H
                self.Gamma_list[i] = Gamma
                K0 = view @ self.Pi_list[i] @ view.T
                #K0 = K0 / torch.norm(K0)
                sx = torch.diag(K0).reshape(-1,1)
                Kx_new = sx + sx.T - 2 * K0
                Kx_new = torch.exp(-Kx_new / 2)
                self.K_list[i] = Kx_new
                diff_list[i] = torch.max(torch.abs(Pi - Pi_pre))
                
            error_iter = torch.max(torch.stack(diff_list))
            F_trial = - sum([lamb[i] * torch.norm(self.Pi_list[i], p=1) for i in range(len(self.Pi_list))])
            for items in itertools.combinations(range(len(self.K_list)), 2):
                F_trial += torch.mean(torch.mean(self.K_list[items[0]] @ self.K_list[items[1]]))

            loss = '{:.4g}'.format(sum([abs(i.item()) for i in diff_list]))
            if logging == 1:
                
                print('outer_iter=', outer_iter, 'loss=', sum(diff_list), "diff_tol=",self.L_list, "diff_list=", diff_list, 'obj=', F_trial)
            elif logging == 0:
                #print(f"outer_iter=: {outer_iter}, Loss: {sum(diff_list)}, diff_tol: {L_list}, diff_list: {diff_list}, obj: {F_trial}")
                progress_bar.set_description(f"outer_iter=: {outer_iter},obj: {'{:.4g}'.format(F_trial)}, Loss: {loss}, diff_list: {diff_list}")
                                #diff_tol: {self.L_list}, 
                #progress_bar.set_postfix({'Iter': outer_iter+1})
            if error_iter < criterion:
                return self.u_list
        return self.u_list