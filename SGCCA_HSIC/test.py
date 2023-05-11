import torch
import itertools
import numpy as np
import math
from itertools import combinations_with_replacement
torch.set_default_tensor_type(torch.DoubleTensor)

from synth_data import create_synthData_new
from sgcca_hsic import SGCCA_HSIC

N = 400
rate = 0.7
views = create_synthData_new(N, mode=3, F=20)

a = SGCCA_HSIC.fit(views=views, eps=1e-5, maxit=20,b=[1.195813174500402, 1.195813174500402, 1.709975946676697])
