import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
from scipy.stats import spearmanr

#pearson cor
Labelpath1 = './SNGCCA/RealData/Exp664_genes.txt'
Datapath1 = "./SNGCCA/RealData/Exp664.txt"
Respath1 = "./SNGCCA//Results/Exp_score.csv"

Labelpath2 = './SNGCCA/RealData/Meth664_probes.txt'
Datapath2 = "./SNGCCA/RealData/Meth664.txt"
Respath2 = "./SNGCCA//Results/Meth_score.csv"

Labelpath3 = './SNGCCA/RealData/miRNA664_miRNA.txt'
Datapath3 = "./SNGCCA/RealData/miRNA664.txt"
Respath3 = "./SNGCCA//Results/miRNA_score.csv"

def _getgradient(Labelpath, Datapath, Respath):

    Exp_label = pd.read_csv(Labelpath, sep='\t', header=None)
    Exp_list = Exp_label.iloc[:, 0].values.tolist()
    Exp = pd.DataFrame(np.loadtxt(Datapath).T, columns=Exp_label)

    # X = Exp
    ExpRes = pd.read_csv(Respath).values
    listname = [ExpRes[0, 0]]
    FilterRes = []
    for i in range(len(listname)):
        list_index: int = Exp_list.index(listname[i])
        FilterRes.append(list_index)

    X = Exp.iloc[:, FilterRes].values
    return X

def spearman_cor(Labelpath1, Datapath1, Respath1,Labelpath2, Datapath2, Respath2):
    X1 = _getgradient(Labelpath1, Datapath1, Respath1)
    X2 = _getgradient(Labelpath2, Datapath2, Respath2)
    corr,p = spearmanr(X1.flatten(), X2.flatten())

    return corr,p

corr1,p1 = spearman_cor(Labelpath1, Datapath1, Respath1,Labelpath2, Datapath2, Respath2)
corr2,p2 = spearman_cor(Labelpath1, Datapath1, Respath1,Labelpath3, Datapath3, Respath3)
corr3,p3 = spearman_cor(Labelpath2, Datapath2, Respath2,Labelpath3, Datapath3, Respath3)
print(corr1,corr2,corr3)

def cor_map(method,Labelpath,Datapath,ypath,Respath,savename):
    Exp_label = pd.read_csv(Labelpath, sep='\t',header = None)
    Exp_list = Exp_label.iloc[:, 0].values.tolist()
    Exp = pd.DataFrame(np.loadtxt(Datapath).T,columns = Exp_label)

    # labels
    y = pd.read_csv(ypath,header = None)
    #X = Exp
    ExpRes = pd.read_csv(Respath).values
    listname = ExpRes[:,0]
    FilterRes = []
    for i in range(len(listname)):
        list_index: int = Exp_list.index(listname[i])
        FilterRes.append(list_index)

    ExpFilter = Exp.iloc[:,FilterRes]
    ExpConcat = pd.concat([ExpFilter,y],axis=1,ignore_index=True)
    listname_new = np.append(listname,'Type')

    ExpConcat.columns = listname_new
    ExpConcat = ExpConcat.sort_values('Type')

    row_c = dict(zip(ExpConcat['Type'].unique(), ['#F2994A', '#FBD786', '#6dd5ed', '#2193b0']))
    row_list = ExpConcat['Type']
    ExpConcat = ExpConcat.drop('Type', axis=1)
    ExpConcat = ExpConcat.T

    linkage = None
    DF_corr = ExpConcat.T.corr()
    DF_dism = 1 - DF_corr
    linkage = hc.linkage(sp.distance.squareform(DF_dism), method='ward')

    sns.clustermap(ExpConcat, pivot_kws=None,
                   method=method,
                   metric='euclidean',
                   z_score=None,
                   standard_scale=None,
                   figsize=(10, 10),
                   cbar_kws=None,
                   row_cluster=True,
                   col_cluster=None,
                   row_linkage=linkage,
                   col_linkage=None,
                   xticklabels=False,
                   col_colors=row_list.map(row_c),
                   row_colors=None,
                   #col_colors=None,
                   mask=None,
                   dendrogram_ratio=0.1,
                   colors_ratio=0.07,
                   cbar_pos=None,
                   tree_kws=None,
                   cmap='RdBu')

    plt.savefig(savename)
    plt.show()

method = 'ward'
num = 'res10/'
datapath = 'C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/'
respath = datapath + num
ypath = datapath + 'PAM50label664.txt'

Labelpath = datapath + 'Exp664_genes.txt'
Datapath = datapath + "Exp664.txt"
Respath = respath + "Exp_score.csv"
savename = respath + "Exp_plot.png"

cor_map(method,Labelpath,Datapath,ypath,Respath,savename)

Labelpath = datapath + 'miRNA664_miRNA.txt'
Datapath = datapath + "miRNA664.txt"
Respath = respath + "miRNA_score.csv"
savename = respath + "miRNA_plot.png"

cor_map(method,Labelpath,Datapath,ypath,Respath,savename)

Labelpath = datapath + 'Meth664_probes.txt'
Datapath = datapath + "Meth664.txt"
Respath = respath + "Meth_score.csv"
savename = respath + "Meth_plot.png"

cor_map(method,Labelpath,Datapath,ypath,Respath,savename)
