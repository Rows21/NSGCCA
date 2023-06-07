import torch
import math
import itertools
import torch.optim as optim


class SNGCCA_ADAM():
    def __init__(self, device):
        self.K_list = []
        self.a_list = []
        self.cK_list = []
        self.u_list = []
        self.device = device

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

    def gradf_gauss_SGD(self, K1, cK2, X, a, u):
        N = K1.shape[0]
        temp1 = torch.zeros((X.shape[1], X.shape[1])).to(self.device)
        au = a

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
        # Kernel Matrix
        D_mn = torch.sqrt(torch.abs(self.sqdist(X, ind)))

        if sigma is None:  # median heuristic
            sigma = torch.median(D_mn)
        else:
            sigma = torch.tensor(sigma)

        K_mn = torch.exp(- (D_mn ** 2) / (2 * sigma ** 2))
        D_nn = torch.sqrt(torch.abs(self.sqdist(X[ind])))

        K_nn = torch.exp(- (D_nn ** 2) / (2 * sigma ** 2))
        K_nn = K_nn + torch.eye(K_nn.shape[0]) * 0.001

        eigenvalues, eigenvectors = torch.linalg.eig(K_nn)
        D_sqrt = torch.diag(torch.sqrt(eigenvalues))
        K_nn_sqrt = (eigenvectors @ D_sqrt @ eigenvectors.inverse()).real

        phi = K_mn @ torch.linalg.inv(K_nn_sqrt)

        return phi, sigma

    def centre_nystrom_kernel(self,phi):
        N = phi.size(0)
        phic = (torch.eye(N) - torch.ones(N) / N) @ phi
        return phic

    def ff(self, K_list, cK_list):
        N = K_list[0].shape[0]
        res = 0
        for items in itertools.combinations(range(len(K_list)), 2):
            res += torch.trace(K_list[items[0]] @ cK_list[items[1]]) / ((N - 1) ** 2)
        return res

    def set_init(self, views, ind, b):
        ## initial
        for i, view in enumerate(views):
            v = torch.rand(view.shape[1]).to(self.device)
            umr = torch.reshape(self.projL1(v, b[i]), (view.shape[1], 1))
            u_norm = umr / torch.norm(umr, p=2).to(self.device)

            ## Calculate Kernel
            Xu = view.to(self.device) @ u_norm
            sigma = None
            if sigma is None:
                phi, a = self.rbf_approx(Xu,ind)
            else:
                phi, a = self.rbf_approx(Xu, ind,sigma)
            K = phi.t() @ phi
            phic = self.centre_nystrom_kernel(phi)
            cK = phic.t() @ phic

            ## Save Parameters
            self.K_list.append(K)
            self.a_list.append(a)
            self.cK_list.append(cK)
            self.u_list.append(u_norm)
        return self.u_list

    def fit(self, views, eps, maxit, b, early_stopping=True, patience=10, logging=0):
        n_view = len(views)
        self.K_list = []
        self.a_list = []
        self.cK_list = []
        self.u_list = []

        # initialization
        self.set_init(views, b)

        diff = 99999
        ite = 0
        obj_list = []
        while (diff > eps) & (ite < maxit):
            ite += 1
            for i, view in enumerate(views):
                obj_old = self.ff(self.K_list, self.cK_list)
                cK_list_SGD = [self.cK_list[j] for j in range(n_view) if j != i]

                ## Calculate Delta and Gamma
                grad = self.gene_SGD(self.K_list[i], cK_list_SGD, view, self.a_list[i], self.u_list[i])

                gamma = torch.norm(grad, p=2)

                ## Start Line Search
                chk = 1
                while chk == 1:
                    ## Update New latent variable
                    v_new = torch.reshape(self.u_list[i] + grad * gamma, (-1,))
                    u_new = torch.reshape(self.projL1(v_new, b[i]), (view.shape[1], 1))
                    u_norm = u_new / torch.norm(u_new, p=2)

                    Xu_new = view.to(self.device) @ u_norm

                    sigma = None
                    if sigma is None:
                        K_new, a_new = self.rbf_kernel(Xu_new)
                    else:
                        K_new, a_new = self.rbf_kernel(Xu_new, sigma)
                    cK_new = self.centre_kernel(K_new)

                    ## update K
                    K_list_SGD = [self.K_list[j] for j in range(n_view) if j != i]
                    K_list_SGD.append(K_new)

                    ## update cK
                    cK_list_SGD = [self.cK_list[j] for j in range(n_view) if j != i]
                    cK_list_SGD.append(cK_new)
                    obj_new = self.ff(K_list_SGD, cK_list_SGD)

                    ## Update Params
                    if obj_new > obj_old + 1e-5 * abs(obj_old):
                        chk = 0
                        self.u_list[i] = u_norm
                        self.K_list[i] = K_new
                        self.cK_list[i] = cK_new
                        self.a_list[i] = a_new
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
                    return self.u_list
        if logging == 2:
            print("diff=", diff, 'obj=', obj)
        return self.u_list

    def test(self):
        pass

    def EarlyStopping(self, lst, patience=5):
        if len(lst) < patience:
            return False
        last_five = lst[-patience:]
        return len(set(last_five)) == 1

