#import torch
import itertools
import numpy as np
device = 'cpu'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print("Using", torch.cuda.device_count(), "GPUs")
if device == 'cuda':
    import cupy as cp
#import cupy as cp
from tqdm import tqdm

from networks.utils import *

class SGCCA_HSIC():
    def __init__(self, device='cpu'):
        self.device = device
    
    def _initilize(self, view, p, initPi=None, rho=1):
        n, p = view.shape
        covx = np.cov(view.T)
        if not is_invertible(covx):
            epsilon = 1e-4 * np.linalg.norm(covx, 'fro') / np.linalg.norm(np.eye(covx.shape[0]) - covx, 'fro')
            covx = (1 - epsilon) * covx + epsilon * np.eye(covx.shape[0])

        eigval, eigvec = np.linalg.eigh((covx + covx.T) / 2)
        eigenvalues = np.real(eigval)
        eigenvalues = np.where(eigenvalues < 0, np.zeros_like(eigenvalues), eigenvalues)
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        sqcovx = eigvec @ np.diag(sqrt_eigenvalues) @ eigvec.T

        tau = 4 * rho * np.linalg.norm(covx, 2) ** 2
        if initPi is None:
            #u = np.random.uniform(0, 1, p)
            #initPi = np.diag(u)/ np.trace(np.diag(u) @ covx)
            initPi = np.eye(p) / np.trace(covx)
            #initPi = np.outer(u, u)
        
        Pi = initPi

        Kx = rbf_kx(view, Pi)
        Z = z(view, p)

        H = sqcovx * Pi * sqcovx
        #H /= np.trace(H)
        Gamma = np.zeros((p, p))
        return covx, sqcovx, tau, Pi, Kx, Z, H, Gamma
    
    def _initilize_cp(self, view, p, rho=1):
        n, p = view.shape
        covx = cp.cov(view.T)
        epsilon = 1e-4 * cp.linalg.norm(covx, 'fro') / cp.linalg.norm(cp.eye(covx.shape[0]) - covx, 'fro')
        covx = (1 - epsilon) * covx + epsilon * cp.eye(covx.shape[0])

        eigval, eigvec = cp.linalg.eigh((covx + covx.T) / 2)
        eigenvalues = cp.real(eigval)
        eigenvalues = cp.where(eigenvalues < 0, cp.zeros_like(eigenvalues), eigenvalues)
        sqrt_eigenvalues = cp.sqrt(eigenvalues)
        sqcovx = eigvec @ cp.diag(sqrt_eigenvalues) @ eigvec.T

        tau = 4 * rho * cp.linalg.norm(covx, 2) ** 2
        initPi = cp.eye(p) / cp.trace(covx)
        Pi = initPi

        Kx = rbf_kx_cp(view, Pi)
        Z = z_cp(view, p)

        H = sqcovx * Pi * sqcovx
        Gamma = cp.zeros((p, p))
        return covx, sqcovx, tau, Pi, Kx, Z, H, Gamma

    def _u_svd(self, i, Pi):
        _, u_new_meta = np.linalg.eigh((Pi + Pi.T) / 2)
        l1_norms = np.sum(np.abs(u_new_meta), axis=0)
        top_indices = np.argsort(l1_norms)[-1:]
        u_new_meta = u_new_meta[:, top_indices]
        u_new = u_new_meta
        self.u_list[i] = u_new
        
    def _u_svd_cp(self, i, Pi):
        _, u_new_meta = cp.linalg.eigh((Pi + Pi.T) / 2)
        l1_norms = cp.sum(cp.abs(u_new_meta), axis=0)
        top_indices = cp.argsort(l1_norms)[-1:]
        u_new_meta = u_new_meta[:, top_indices]
        u_new = u_new_meta
        self.u_list[i] = u_new

    def fit_admm(self, views, constraint, criterion=5e-7,logging=1, mode = 'compute', Pi0_list = None):
        
        self.K_list = []
        self.a_list = []
        self.cK_list = []
        
        self.covx_list = []
        self.y_lab = []
        self.sqcovx_list = []
        self.tau_list = []
        self.H_list = []
        self.Gamma_list = []
        self.Pi_list = []
        self.L_list = []
        self.Z_list = []
        self.Pi0_list = []
        
        self.views = views
        self.n_views = len(views)
        self.u_list = [None] * self.n_views
        rho = 1
        for i, view in enumerate(self.views):
            if Pi0_list is not None:
                covx, sqcovx, tau, Pi, Kx, Z, H, Gamma = self._initilize(view, rho, Pi0_list[i])
            else:
                covx, sqcovx, tau, Pi, Kx, Z, H, Gamma = self._initilize(view, rho)
            
            self.covx_list.append(covx)
            self.sqcovx_list.append(sqcovx)
            self.tau_list.append(tau)
            self.Z_list.append(Z)
            self.K_list.append(Kx)
            
            self.Pi_list.append(Pi)
            if mode == 'multi_start':
                self.Pi0_list.append(Pi)
            self.H_list.append(H)
            self.Gamma_list.append(Gamma* (self.n_views - 1))
            

        outer_maxiter = 10
        outer_tol = 1e-5
        inner_maxiter = 50
        
        inner_tol = 2e-9
        outer_error = 1
        diff_list = [999] * self.n_views
        criterion = criterion
        if logging == 0:
            progress_bar = tqdm(total=outer_maxiter, ncols=200)
        
        for outer_iter in range(outer_maxiter):            
            for i, view in enumerate(self.views):
                
                n, p = view.shape
                Kl_grad = sum([self.K_list[j] for j in range(self.n_views) if j != i])
                K_tilde = rbf_kl(Kl_grad) # H @ Kl_grad @ H
                Coeft = self.K_list[i] * K_tilde
                if p < n:
                    dF = delta_Pi(view, Coeft)
                else:
                    dF = delta_Pi(view, Coeft)
                    
                L = 2 * np.sum(abs(K_tilde) * self.Z_list[i])/ (4 * n ** 2)
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

                while (inner_iter <= inner_maxiter) & (inner_error > inner_tol):
                    #print(inner_iter)
                    #print("Pi",torch.trace(sqcovx @ Pi @ sqcovx) )
                    temp = Pi-(rho/tau) * covx @ Pi @ covx + (rho/tau) * sqcovx @ (H-Gamma) @ sqcovx
                    temp = tau/(tau+L)*temp+L/(tau+L) * a
                    
                    Pi = np.maximum(temp - constraint[i] / (L + tau), np.zeros(temp.shape)) * np.sign(temp)                        
                    H = FantopeProjection(sqcovx @ Pi @ sqcovx + Gamma)
                    #print("H",torch.trace(H))
                    Gamma = Gamma + sqcovx @ Pi @ sqcovx - H
                    
                    inner_error = np.max([np.max(np.max(np.abs(sqcovx @ Pi @ sqcovx - H))), np.max(np.max(np.abs(Pi - Pi_pre)))])
                    inner_iter = inner_iter + 1

                self.Pi_list[i] = Pi
                self.H_list[i] = H
                self.Gamma_list[i] = Gamma
                
                self.K_list[i] = rbf_kx(view, Pi)
                
                diff_list[i] = np.max(abs(Pi - Pi_pre))
                #print(inner_iter)
                #print(inner_error)
            error_iter = np.max(np.stack(diff_list))
            F_trial = -np.sum([constraint[i] * np.linalg.norm(self.Pi_list[i], ord=1) for i in range(len(self.Pi_list))])
            for items in itertools.combinations(range(len(self.K_list)), 2):
                F_trial += np.trace(self.K_list[items[0]] @ rbf_kl(self.K_list[items[1]]))

            loss = '{:.4g}'.format(sum([abs(i) for i in diff_list]))
            if logging == 1:
                print('outer_iter=', outer_iter, 'loss=', loss, "diff_list=", diff_list, 'obj=', F_trial)
            elif logging == 0:
                progress_bar.set_description(f"outer_iter=: {outer_iter},obj: {'{:.4g}'.format(F_trial)}, Loss: {loss}, diff_list: {np.around(diff_list, decimals=8)}")

            if error_iter < criterion:
                for i in range(self.n_views):
                    self._u_svd(i, self.Pi_list[i])
                if mode == 'cv':
                    return self.Pi_list, self.u_list
                elif mode == 'multi_start':
                    return self.Pi_list, self.u_list, F_trial
                else:
                    return self.u_list
            else:
                continue
        for i in range(self.n_views):
            self._u_svd(i, self.Pi_list[i])
        if mode == 'multi_start':
            return self.Pi_list, self.u_list, F_trial
        else: 
            return self.u_list
    
    def fit_admm_cp(self, views, constraint, logging=1, mode = 'compute'):
        
        self.K_list = []
        self.a_list = []
        self.cK_list = []
        
        self.covx_list = []
        self.y_lab = []
        self.sqcovx_list = []
        self.tau_list = []
        self.H_list = []
        self.Gamma_list = []
        self.Pi_list = []
        self.L_list = []
        self.Z_list = []
        
        self.views = [cp.asarray(view) for view in views]
        self.n_views = len(views)
        self.u_list = [None] * self.n_views
        rho = 1

        for i, view in enumerate(self.views):
            covx, sqcovx, tau, Pi, Kx, Z, H, Gamma = self._initilize_cp(view, rho)
            
            self.covx_list.append(covx)
            self.sqcovx_list.append(sqcovx)
            self.tau_list.append(tau)
            self.Z_list.append(Z)
            self.K_list.append(Kx)
            
            self.Pi_list.append(Pi)
            self.H_list.append(H)
            self.Gamma_list.append(Gamma* (self.n_views - 1))
            

        outer_maxiter = 2000
        outer_tol = 1e-5
        inner_maxiter = 50
        inner_tol = 1e-4
        outer_error = 1
        diff_list = [999] * self.n_views
        criterion = 5e-7
        if logging == 0:
            progress_bar = tqdm(total=outer_maxiter, ncols=200)
        
        for outer_iter in range(outer_maxiter):            
            for i, view in enumerate(self.views):
                n, p = view.shape
                Kl_grad = sum([self.K_list[j] for j in range(self.n_views) if j != i])
                K_tilde = rbf_kl_cp(Kl_grad) # H @ Kl_grad @ H
                Coeft = self.K_list[i] * K_tilde

                dF = delta_Pi_cp(view, Coeft)
                L = 4 * sum(abs(K_tilde) * self.Z_list[i])/ (4 * n ** 2)
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

                while (inner_iter <= inner_maxiter) & (inner_error > inner_tol):
                    #print(inner_iter)
                    #print("Pi",torch.trace(sqcovx @ Pi @ sqcovx) )
                    temp = Pi-(rho/tau) * covx @ Pi @ covx + (rho/tau) * sqcovx @ (H-Gamma) @ sqcovx
                    temp = tau/(tau+L)*temp+L/(tau+L) * a
                    
                    Pi = cp.maximum(temp - constraint[i] / (L + tau), cp.zeros(temp.shape)) * cp.sign(temp)                        
                    H = FP_cp(sqcovx @ Pi @ sqcovx + Gamma)
                    #print("H",torch.trace(H))
                    Gamma = Gamma + sqcovx @ Pi @ sqcovx - H
                    
                    inner_error = max([cp.max(cp.max(cp.abs(sqcovx @ Pi @ sqcovx - H))), cp.max(cp.max(cp.abs(Pi - Pi_pre)))])
                    inner_iter = inner_iter + 1

                self.Pi_list[i] = Pi
                self.H_list[i] = H
                self.Gamma_list[i] = Gamma
                
                self.K_list[i] = rbf_kx_cp(view, Pi)
                
                diff_list[i] = cp.max(abs(Pi - Pi_pre))
                
            error_iter = cp.max(cp.stack(diff_list))
            F_trial = -sum([constraint[i] * cp.linalg.norm(self.Pi_list[i], ord=1) for i in range(len(self.Pi_list))])
            for items in itertools.combinations(range(len(self.K_list)), 2):
                F_trial += cp.trace(self.K_list[items[0]] @ rbf_kl_cp(self.K_list[items[1]]))

            loss = '{:.4g}'.format(sum([abs(i) for i in diff_list]))
            if logging == 1:
                print('outer_iter=', outer_iter, 'loss=', sum(diff_list), "diff_tol=",self.L_list, "diff_list=", diff_list, 'obj=', F_trial)
            elif logging == 0:
                progress_bar.set_description(f"outer_iter=: {outer_iter},obj: {'{:.4g}'.format(F_trial)}, Loss: {loss}, diff_list: {cp.around(diff_list, decimals=8)}")

            if error_iter < criterion:
                for i in range(self.n_views):
                    self._u_svd_cp(i, self.Pi_list[i])
                if mode == 'cv':
                    return self.Pi_list, self.u_list
                elif mode == 'multi_start':
                    return self.Pi_list, self.u_list, F_trial
                else:
                    return self.u_list
        return self.u_list