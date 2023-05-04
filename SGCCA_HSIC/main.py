import torch
import numpy as np
import math
from itertools import combinations_with_replacement
torch.set_default_tensor_type(torch.DoubleTensor)

from synth_data import create_synthData_new
from sgcca_hsic import SGCCA_HSIC

class Solver():
    def __init__(self):
        self.SGCCA_HSIC = SGCCA_HSIC()

    def fit(self, x_list, vx_list=None, tx_list=None, checkpoint='checkpoint.model'):
        x_list = [x.to(device) for x in x_list]
        data_size = x_list[0].size(0)

        if vx_list is not None :
            best_val_loss = 0
            vx_list = [vx.to(self.device) for vx in vx_list]

        if tx_list is not None :
            tx_list = [tx.t0(self.device) for tx in tx_list]

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

    def tune_hyper(self,x_list,set):
        folds = 3
        a = np.exp(np.linspace(0, math.log(5), num=set))
        for aa in combinations_with_replacement(a, 3):
            u = self.SGCCA_HSIC.fit(x_list,1e-5,20,aa)



if __name__ == '__main__':
    ############
    # Parameters Section
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using", torch.cuda.device_count(), "GPUs")

    N = 400
    views = create_synthData_new(N, mode=3, F=20)

    print(f'input views shape :')
    for i, view in enumerate(views):
        print(f'view_{i} :  {view.shape}')
        view = view.to("cpu")

    a = Solver()
    u = []

    u = a.tune_hyper(views,10)
    print(u)

