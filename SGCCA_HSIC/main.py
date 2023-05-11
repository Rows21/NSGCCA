import torch
import numpy as np
import math
from itertools import combinations_with_replacement
torch.set_default_tensor_type(torch.DoubleTensor)

from synth_data import create_synthData_new
from sgcca_hsic import SGCCA_HSIC

from validation_method import FS_MCC

class Solver():
    def __init__(self):
        self.SGCCA_HSIC = SGCCA_HSIC()

    def fit(self, x_list, vx_list=None, tx_list=None, checkpoint='checkpoint.model'):
        x_list = [x.to(device) for x in x_list]
        data_size = x_list[0].size(0)

        ##### ADD test and validation sets

        train_losses = []

        # train_linear_gcca
        if self.SGCCA_HSIC is not None:
            _, outputs_list = self._get_outputs(x_list)
            self.train_linear_gcca(outputs_list)

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)
        if vx_list is not None:
            loss = self.test(vx_list)
            self.logger.info("loss on validation data: {:.4f}".format(loss))

        if tx_list is not None:
            loss = self.test(tx_list)
            self.logger.info('loss on test data: {:.4f}'.format(loss))

    def tune_hyper(self,x_list,set_params,eps = 1e-5,iter = 20):
        ## fixed folds number
        folds = 3
        ## split
        shuffled_index = np.random.permutation(len(x_list[0]))
        split_index = int(len(x_list[0]) * 1/folds)
        train_index = shuffled_index[:split_index]
        test_index = shuffled_index[split_index:]

        train_data = []
        test_data = []
        for i, view in enumerate(views):
            train_data.append(view[train_index, :])
            test_data.append(view[test_index, :])

        ## calculate K,cK for validation set


        ## set hyperparams set
        a = np.exp(np.linspace(0, math.log(5), num=set_params))

        ## start validation
        b0 = a[0]
        obj_validate = 0
        count = 0
        for aa in combinations_with_replacement(a, 3):
            count +=1

            u = self.SGCCA_HSIC.fit(train_data,eps,iter,aa)

            # Save iterations
            K_test = []
            cK_test = []
            for i,view in enumerate(test_data):
                Xu = view @ u[i]
                sigma = None
                if sigma is None:
                    K, a = self.SGCCA_HSIC.rbf_kernel(Xu)
                else:
                    K, a = self.SGCCA_HSIC.rbf_kernel(Xu, sigma)
                cK = self.SGCCA_HSIC.centre_kernel(K)
                K_test.append(K)
                cK_test.append(cK)
            obj_temp = self.SGCCA_HSIC.ff(K_test,cK_test)

            print("Sparsity selection number=", count, "hyperparams=", aa,"obj=",obj_temp)
            if obj_temp > obj_validate:
                b0 = aa
                obj_validate = obj_temp
            else:
                continue
        return b0,obj_validate

    def _get_outputs(self,views,eps,maxit,b):
        u = self.SGCCA_HSIC.fit(views, eps, maxit,b)
        return u

if __name__ == '__main__':
    ############
    # Hyper Params Section
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using", torch.cuda.device_count(), "GPUs")

    N = 400
    views = create_synthData_new(N, mode=3, F=10)

    print(f'input views shape :')
    for i, view in enumerate(views):
        print(f'view_{i} :  {view.shape}')
        view = view.to("cpu")

    a = Solver()
    u = []
    ## train hyper
    b0,obj = a.tune_hyper(views,5)

    ## fit results
    u = a._get_outputs(views,1e-5,20,b0)








