import torch
import math
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

    def rbf_kernel(self, X, sigma=None):
        # 计算距离矩阵
        D = torch.sqrt(torch.abs(self.sqdist(X.t(), X.t())))

        if sigma is None:
            # 中位数启发式法估计 sigma
            sigma = torch.median(D)

        # 计算核矩阵
        K = torch.exp(- (D ** 2) / (2 * sigma ** 2))
        return K, sigma

    def centre_kernel(self, K):
        return K + torch.mean(K) - (torch.mean(K, dim=0).reshape((1, -1)) + torch.mean(K, dim=1).reshape((-1, 1)))

    def _FantopeProjection(self, W):
        # This code is to solve the projection problem onto the fantope constraint
        # min_H || H - W ||
        # s.t. || H || _{*} <= K, || H || _{sp} <= 1
        temp = (W + W.T)/2
        
        D, V = torch.linalg.eigh(temp)
        #D,V = eigs(temp.numpy())
        d = D.reshape(-1,1)

        if torch.sum(torch.min(torch.tensor(1.0), torch.max(d, torch.zeros_like(d)))) <= 1:
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
        self.Ly_list = []
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
            tau = 2 * rho * torch.max(eigenvalues) ** 2
            self.tau_list.append(tau)

            # set init u
            u = torch.ones((p, 1))
            u = torch.sqrt(u / torch.norm(u, p=2))
            self.u_list[i] = u
            u = u / torch.sqrt(u.t() @ sqcovx @ u)
            initPi = u @ u.t()
            initPi = torch.zeros((p,p))
            Pi = initPi#/torch.norm(initPi)
            self.Pi_list.append(Pi)

            # set Label kernel
            sx = torch.diag(view @ Pi @ view.T)
            Kx = sx + sx.T - 2 * view @ Pi @ view.T
            Kx = torch.exp(-Kx / 2)
            self.K_list.append(Kx)

            self.y_lab.append(view @ u)

            # construct Lipchitz Constant L
            # product kernel
            #y = np.genfromtxt('ydata.csv', delimiter=',')
            #y = torch.tensor(y).reshape(-1,1)
        
        for i, view in enumerate(views):
            y = sum([self.y_lab[j] for j in range(len(views))])/(len(views)) # if j != i
            y = y / torch.sqrt(torch.sum(y ** 2))
            sigmaY2 = torch.var(y)
            sy = torch.diag(y @ y.T).reshape(-1, 1)

            Ly = sy + sy.T - 2*(y @ y.T)
            Ly = torch.exp(-Ly/(2*sigmaY2))
            Ly = self.centre_kernel(Ly)
            self.Ly_list.append(Ly)

            Coef = - Ly * (Ly < 0)/4
            Coef = 2 * (torch.diag(torch.squeeze(torch.sum(Coef,dim=1,keepdim=True))) - Coef)/(n ** 2)
            Coef = view.T @ Coef @ view
            L, _ = torch.linalg.eigh((Coef + Coef.t())/2)
            self.L_list.append(L[-1])

            # Set initial H, Gamma
            H = sqcovx * Pi * sqcovx
            self.H_list.append(H)
            Gamma = torch.zeros((p, p))
            self.Gamma_list.append(Gamma* (n_views - 1))

        print(self.L_list)
        #lamb = [i/100 * 1.5 for i in L_list]
        outer_maxiter = 1000
        outer_tol = 1e-5
        inner_maxiter = 1000
        inner_tol = 1e-3
        outer_error = 1
        diff_list = [999] * n_views

        diff_log = [[999], [999], [999]]
        norms = [None] * n_views
        diff_L = self.L_list
        out_idx = []
        criterion = 1e-6

        progress_bar = tqdm(total=outer_maxiter, ncols=200)
        
        for outer_iter in range(outer_maxiter):
            #if outer_error < outer_tol:
            #    return Pi_list
            Pi_list = self.Pi_list
            H_list = self.H_list
            Gamma_list = self.Gamma_list

            L_list = self.L_list
            Ly_list = self.Ly_list

            if set(out_idx) == set(range(n_views)):
                a=1

            for i, view in enumerate(views):
                #print(sum(diff_log[i][-1:-11:-1]))
                if i in out_idx:
                    continue
                    
                #Ly_grad = [self.Ly_list[j] for j in range(len(views)) if j != i]
                #Ly_grad = sum(Ly_grad)
                Ly_grad = self.Ly_list[i]
                    
                #L_grad = [self.L_list[j] for j in range(len(views)) if j != i]
                L_grad = self.L_list[i]#sum(L_grad)/len(L_grad)

                u_pre = self.u_list[i]
                    
                Coeft = self.K_list[i] * Ly_grad /2

                n = Coeft.shape[0]
                sum_Coeft = torch.sum(Coeft, dim=1)
                diag_sum_Coeft = torch.diag(sum_Coeft)
                dF = torch.matmul(torch.matmul(view.T, (2 * (diag_sum_Coeft - Coeft)) / (n ** 2)), view)

                Pi = self.Pi_list[i]

                L = L_grad
                a = Pi - dF / L
                Pi_pre = Pi
                inner_error = 1
                inner_iter = 0

                sqcovx = self.sqcovx_list[i]
                covx = self.covx_list[i]
                tau = self.tau_list[i]

                H = self.H_list[i]
                Gamma = self.Gamma_list[i]
                
                while (inner_iter <= inner_maxiter) & (inner_error > inner_tol):
                    #print(inner_iter)
                    temp = Pi-(rho/tau) * covx @ Pi @ covx + (rho/tau) * sqcovx @ (H-Gamma) @ sqcovx
                    #if (sum(sum(temp)) == torch.tensor(0)).item():
                    #    break
                    temp = tau/(tau+L)*temp+L/(tau+L) * a
                    
                    Pi = torch.max((torch.abs(temp)- lamb[i]/(L + tau)), torch.zeros(temp.size())) * torch.sign(temp)                                
                    H = self._FantopeProjection(sqcovx @ Pi @ sqcovx + Gamma)
                    Gamma = Gamma + sqcovx @ Pi @ sqcovx - H
                        
                    inner_error = torch.norm(sqcovx @ Pi @ sqcovx - H, 'fro')
                    inner_iter = inner_iter + 1

                try:
                    _, u_new_meta = torch.linalg.eigh((Pi + Pi.T) / 2)
                    l1_norms = torch.sum(torch.abs(u_new_meta),dim=0)
                    top_indices = np.argsort(l1_norms)[-1:]
                    u_new_meta = u_new_meta[:, top_indices]
                    #u_new = u_new / torch.sqrt(u_new.T @ sqcovx_list[i] @ u_new)
                    u_new = u_new_meta

                except:
                    Pi = Pi.numpy()
                    _, u_new_meta = eigs((Pi + Pi.T) / 2)
                    l1_norms = np.sum(np.abs(u_new_meta), axis=0)
                    top_indices = np.argsort(l1_norms)[-1:]
                    u_new_meta = u_new_meta[:, top_indices]
                    u_new = torch.tensor(np.real(u_new_meta))
                    Pi = torch.tensor(np.real(Pi))

                outer_error = torch.norm(torch.abs(u_new-self.u_list[i]))
                outer_error_P = torch.norm(torch.abs(Pi-Pi_pre))
                #print(outer_error_P)
                #print(u_new-self.u_list[i])
                
                #print(torch.norm(Pi))
                #diff_list[i] = outer_error
                #diff_log[i].append(outer_error)
                u_new = u_new /  torch.sqrt(torch.sum(u_new ** 2))

                if (outer_error > outer_tol).item() or (sum(abs(self.u_list[i]) - abs(u_pre)).numpy()[0] == 0): # or torch.norm(u_new, p=1) - torch.norm(self.u_list[i], p=1) *2 < 0
                        #if (self.u_list[1][-98:].sum() == 0).item() and (self.u_list[1][:1].sum() != 1).item() and (self.u_list[1][:1].sum() != 0).item():
                        #    out_idx.append(i)
                        self.u_list[i] = u_new
                        #print(torch.norm(u_new))
                        Pi_list[i] = Pi / torch.norm(Pi)
                        H_list[i] = H
                        Gamma_list[i] = Gamma
                            
                        # normalize
                        # new L
                        self.y_lab[i] = view @ u_new
                        y = sum([self.y_lab[j] for j in range(len(views)) if j != i])/(len(views)-1)
                        y = y / torch.sqrt(torch.sum(y ** 2))
                        #y = view @ u_new
                        sigmaY2 = torch.var(y)

                        #print(sigmaY2)
                        sy = torch.diag(y @ y.T).reshape(-1, 1)

                        # new Ly
                        Ly = sy + sy.T - 2 * (y @ y.T)
                        Ly = torch.exp(-Ly / (2 * sigmaY2))
                        Ly = self.centre_kernel(Ly)
                        if torch.isnan(torch.sum(Ly)).item() == None:
                            a = 1
                        Ly_list[i] = Ly 

                        # new Lipchitz Constant L
                        Coef = - Ly * (Ly < 0) / 4
                        Coef = 2 * (torch.diag(torch.squeeze(torch.sum(Coef, dim=1, keepdim=True))) - Coef) / (n ** 2)
                        Coef = view.T @ Coef @ view
                        L, _ = torch.linalg.eigh((Coef + Coef.t()) / 2)
                        L_list[i] = L[-1]
                        
                        sx = torch.diag(view @ Pi @ view.T).reshape(-1,1)
                        Kx_new = sx + sx.T - 2 * view @ Pi @ view.T
                        Kx_new = torch.exp(-Kx_new / 2)
                        self.K_list[i] = Kx_new

                diff_list[i] = sum(torch.abs(self.u_list[i]) - torch.abs(u_pre))
                if abs(diff_list[i].numpy()[0]) <= 1e-10:
                    #print(self.u_list[i])
                    self.u_list[i] = u_pre
            
            self.Pi_list = Pi_list
            self.H_list = H_list
            self.Gamma_list = Gamma_list
            self.L_list = L_list
            self.Ly_list = Ly_list

            F_trial = - sum([lamb[i] * torch.norm(self.Pi_list[i], p=1) for i in range(len(self.Pi_list))])
            for items in itertools.combinations(range(len(self.K_list)), 2):
                F_trial += torch.mean(torch.mean(self.K_list[items[0]] @ self.K_list[items[1]]))

            loss = '{:.4g}'.format(sum([abs(i.item()) for i in diff_list]))
            if logging == 1:
                
                print('outer_iter=', outer_iter, 'loss=', sum(diff_list), "diff_tol=",L_list, "diff_list=", diff_list, 'obj=', F_trial)
            elif logging == 0:
                #print(f"outer_iter=: {outer_iter}, Loss: {sum(diff_list)}, diff_tol: {L_list}, diff_list: {diff_list}, obj: {F_trial}")
                progress_bar.set_description(f"outer_iter=: {outer_iter},obj: {F_trial}, Loss: {loss}, diff_tol: {L_list}, diff_list: {diff_list}")
                #progress_bar.set_postfix({'Iter': outer_iter+1})
            if sum([abs(i.item()) for i in diff_list])/len(diff_list) < 1e-4/3:
                return self.u_list
        return self.u_list


