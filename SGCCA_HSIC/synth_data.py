import numpy as np
import torch


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
        V3[:,0]=np.sin(V2[:,0]+V2[:,1]) * (V2[:,0]+V2[:,1]) -V3[:,1]
        
    if mode==3:
        V2[:,0]=(V1[:,0]+V1[:,1]) ** 3-V2[:,1]
        V3[:,0]=(V1[:,0]+V1[:,1]) ** 2-V3[:,1]

    views.append(V2) 
    views.append(V3)

    views = [torch.tensor(view).to(device) for view in views]
    return views

def create_synthData_multi(i, data_type,N=400, p=20, q=20, r=20,device='cpu'):
    X = torch.randn(N, p)
    Y = torch.randn(N, q)
    Z = torch.randn(N, r)

    if i == 2:
        if data_type == 1:
            Y[:, 0] = X[:, 0] + X[:, 1] - Y[:, 1] + torch.randn(N) * 0.05
            Z[:, 0] = 2 * (X[:, 0] + X[:, 1]) - Z[:, 1] + torch.randn(N) * 0.05
        elif data_type == 2:
            Y[:, 0] = np.sin(X[:, 0] + X[:, 1]) - Y[:, 1] + torch.randn(N) * 0.05
            Z[:, 0] = np.cos(X[:, 0] + X[:, 1]) - Z[:, 1] + torch.randn(N) * 0.05
        elif data_type == 3:
            Y[:, 0] = (X[:, 0] + X[:, 1]) ** 3 - Y[:, 1] + torch.randn(N) * 0.05
            Z[:, 0] = (X[:, 0] + X[:, 1]) ** 2 - Z[:, 1] + torch.randn(N) * 0.05
        elif data_type == 4:
            Y[:, 0] = torch.cos(X[:, 0] + X[:, 1]) - Y[:, 1]
            Z[:, 0] = torch.sin(X[:, 0] + X[:, 1]) - Z[:, 1]
        elif data_type == 5:
            Z[:, 0] = torch.sqrt(4 - (X[:, 0] + X[:, 1])**2 - (Y[:, 0] + Y[:, 1])**2) - Z[:, 1]
    else:
        for j in range(1, i):
            if data_type == 1:
                Y[:, 0] = torch.sum(X[:, :j], dim=1) - torch.sum(Y[:, 1:j], dim=1) + torch.randn(N) * 0.05
                Z[:, 0] = 2 * torch.sum(X[:, :j], dim=1) - torch.sum(Z[:, 1:j], dim=1) + torch.randn(N) * 0.05
            elif data_type == 2:
                Y[:, 0] = 1 * np.sin(1 * torch.sum(X[:, :j], dim=1)) - torch.sum(Y[:, 1:j], dim=1) + torch.randn(N) * 0.05
                Z[:, 0] = 1 * np.cos(1 * torch.sum(X[:, :j], dim=1)) - torch.sum(Z[:, 1:j], dim=1) + torch.randn(N) * 0.05
            elif data_type == 3:
                Y[:, 0] = torch.sum(X[:, :j], dim=1) **3 - torch.sum(Y[:, 1:j], dim=1) + torch.randn(N) * 0.05
                Z[:, 0] = torch.sum(X[:, :j], dim=1) **2 - torch.sum(Z[:, 1:j], dim=1) + torch.randn(N) * 0.05
            elif data_type == 4:
                Y[:, 0] = torch.cos(torch.sum(X[:, :j], dim=1)) - torch.sum(Y[:, 1:j], dim=1)
                Z[:, 0] = torch.sin(torch.sum(X[:, :j], dim=1)) - torch.sum(Y[:, 1:j], dim=1)
            elif data_type == 5:
                Z[:, 0] = torch.sqrt(30 - (X[:, 0] + X[:, 1])**2 - (Y[:, 0] + Y[:, 1])**2) - Z[:, 1]

    views = [X,Y,Z]
    views = [torch.tensor(view).to(device) for view in views]
    return views
