import torch
torch.set_default_tensor_type(torch.DoubleTensor)

from synth_data import create_synthData_new
from sgcca_hsic import SGCCA_HSIC


N = 400
views = create_synthData_new(N,mode=3,F=20)
print(f'input views shape :')
for i, view in enumerate(views):
    print(f'view_{i} :  {view.shape}')
    view = view.to("cpu")

a = SGCCA_HSIC(views)
u = []

u = a.fit(1e-7,40)
print(u)

class Solver():
    def __init__(self,model):
        self.model = model

    def fit(self):
        pass
