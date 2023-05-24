import itertools
import torch
import torch.optim as optim

class SGCCA_HSIC():
    def __init__(self):
        self.K_list = []
        self.a_list = []
        self.cK_list = []
        self.u_list = []

    def projL1(self, v, b):
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

    def sqdist(self, X1, X2):
        n1 = X1.shape[1]
        n2 = X2.shape[1]
        D = torch.sum(X1 ** 2, dim=0).reshape(-1, 1).repeat(1, n2) + torch.sum(X2 ** 2, dim=0).reshape(1, -1).repeat(n1,1) - 2 * torch.mm(X1.T, X2)
        return D

    def gradf_gauss(self, K1, cK2, X, a, u):
        N = K1.shape[0]
        temp1 = torch.zeros((X.shape[1], X.shape[1]))
        au = a

        indices1 = torch.randperm(N)[:N//10]
        indices2 = torch.randperm(N)[:N//10]

        for i in range(N//10):
            for j in range(N//10):
                a = indices1[i]
                b = indices2[j]
                temp1 += K1[a, b] * cK2[a, b] * torch.ger(X[a, :] - X[b, :], X[a, :] - X[b, :])
        final = -2 * au * u.T @ temp1
        return final.T

    def gene(self, K1, cK_list, X, a, u):
        res = torch.empty(u.shape[0], 1)
        for i in range((len(cK_list))):
            temp = self.gradf_gauss(K1, cK_list[i], X, a, u)
            res += temp
        return res

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

    def f(self, K1, K2, K3):
        N = K1.shape[0]
        cK2 = self.centre_kernel(K2)
        cK3 = self.centre_kernel(K3)

        res = torch.trace(K1 @ cK2) / ((N - 1) ** 2) + torch.trace(K1 @ cK3) / ((N - 1) ** 2) + torch.trace(
            K2 @ cK3) / ((N - 1) ** 2)
        return res

    def ff(self,K_list,cK_list):
        N = K_list[0].shape[0]
        res = 0
        for items in itertools.combinations(range(len(K_list)), 2):
            res += torch.trace(K_list[items[0]] @ cK_list[items[1]]) / ((N - 1) ** 2)
        return res

    def set_init(self,views,b):
        for i, view in enumerate(views):
            v = torch.rand(view.shape[1])
            umr = torch.reshape(self.projL1(v, b[i]), (view.shape[1], 1))
            u_norm = umr / torch.norm(umr, p=2)

            ## Calculate Kernel
            Xu = view @ u_norm
            sigma = None
            if sigma is None:
                K, a = self.rbf_kernel(Xu)
            else:
                K, a = self.rbf_kernel(Xu, sigma)
            cK = self.centre_kernel(K)

            ## Save Parameters
            self.K_list.append(K)
            self.a_list.append(a)
            self.cK_list.append(cK)
            self.u_list.append(u_norm)

    def fit(self,views, eps, maxit,b,early_stopping = True,patience = 10,logging = 0):
        n_view = len(views)
        self.K_list = []
        self.a_list = []
        self.cK_list = []
        self.u_list = []

        self.set_init(views,b)

        diff = 99999
        ite = 0
        obj_list = []

        optimizers = [optim.SGD([u], lr=0.1) for u in self.u_list]

        while (diff > eps) & (ite < maxit):
            ite += 1
            u_list_new = []
            obj_new = []
            for j in range(n_view):
                u = self.u_list[j]
                a = self.a_list[j]
                cK = self.cK_list[j]
                K = self.K_list[j]

                optimizers[j].zero_grad()

                gene_u = self.gene(K, self.cK_list[:j] + self.cK_list[j + 1:], views[j], a, u)
                u_new = self.projL1(u - gene_u, b[j])
                u_new = u_new / torch.norm(u_new, p=2)

                u_new.requires_grad_(True)

                Xu = views[j] @ u_new
                if self.a_list[j] is None:
                    K, a = self.rbf_kernel(Xu)
                else:
                    K, a = self.rbf_kernel(Xu, self.a_list[j])

                cK = self.centre_kernel(K)

                self.K_list[j] = K
                self.a_list[j] = a
                self.cK_list[j] = cK
                self.u_list[j] = u_new

                obj_new.append(torch.trace(K @ cK) / ((K.shape[0] - 1) ** 2))
                u_list_new.append(u_new)

            obj = self.ff(self.K_list, self.cK_list)
            diff = abs(torch.sum(torch.tensor(obj_new)) - obj)
            self.u_list = u_list_new

            obj_list.append(obj)

            if early_stopping:
                if ite > patience:
                    if obj_list[-1] - obj_list[-patience] < 1e-7:
                        if logging > 0:
                            print(f"Early stopping after {ite} iterations.")
                        break

            if logging > 0:
                print(f"Iteration {ite} : obj = {obj:.4f}, diff = {diff:.4f}")

        self.obj_list = obj_list
        return self.u_list