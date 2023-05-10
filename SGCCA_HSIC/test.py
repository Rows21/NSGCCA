import torch
import numpy as np
import math
from itertools import combinations_with_replacement
torch.set_default_tensor_type(torch.DoubleTensor)

from synth_data import create_synthData_new
from sgcca_hsic import SGCCA_HSIC

N = 400
rate = 0.7
views = create_synthData_new(N, mode=3, F=20)
for i, view in enumerate(views):
    print(view.shape[1])

