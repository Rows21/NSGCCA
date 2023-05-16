import numpy as np
import scipy
import scipy.io
import scipy.linalg

import seaborn as sns

import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd

import torch

import os

def create_synthData_new(N=400, outDir='./', device='cpu',mode=1,F=20):
    '''
    creating Main paper Synth data,
    N : number of data
    F$ : number of features in view $ 
    '''
    views  = []
    F1 = F
    F2 = F  
    F3 = F

    V1 = np.random.randn(N, F1)
    V2 = np.random.randn(N, F2)
    V3 = np.random.randn(N, F3)
    views.append(V1)
    if mode==1:
        V2[:,0]=V1[:,0]+V1[:,1]-V2[:,1]
        V3[:,0]=V1[:,0]+2*V1[:,1]-V3[:,1]
        
    if mode==2:
        V2[:,0]=np.sin(V1[:,0]+V1[:,1])-V2[:,1]
        V3[:,0]=np.sin(V2[:,0]+V2[:,1])-V3[:,1]
        
    if mode==3:
        V2[:,0]=1/(V1[:,0]+V1[:,1])-V2[:,1]
        V3[:,0]=1/(V1[:,0]+V1[:,1])-V3[:,1]

    views.append(V2) 
    views.append(V3)

    views = [torch.tensor(view).to(device) for view in views]
    return views
