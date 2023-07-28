import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score,classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Result Data
ExpRes = pd.read_csv("./Results/Exp_score.csv").values
MethRes = pd.read_csv("./Results/Meth_score.csv").values
miRNARes = pd.read_csv("./Results/miRNA_score.csv").values

# Original Data
Exp_label = pd.read_csv('RealData/Exp664_genes.txt', sep='\t', header=None)
Exp_list = Exp_label.iloc[:, 0].values.tolist()
Exp = pd.DataFrame(np.loadtxt("RealData/Exp664.txt").T, columns=Exp_label)

Meth_label = pd.read_csv('RealData/Meth664_probes.txt', sep='\t', header=None)
Meth_list = Meth_label.iloc[:, 0].values.tolist()
Meth = pd.DataFrame(np.loadtxt("RealData/Meth664.txt").T, columns=Meth_label)

miRNA_label = pd.read_csv('RealData/miRNA664_miRNA.txt', sep='\t', header=None)
miRNA_list = miRNA_label.iloc[:, 0].values.tolist()
miRNA = pd.DataFrame(np.loadtxt("RealData/miRNA664.txt").T, columns=miRNA_label)

# labels
y = pd.read_csv('RealData/PAM50label664.txt', header=None).values

def SWISS(Labelpath,Datapath,ypath,Respath):
    Exp_label = pd.read_csv(Labelpath, sep='\t',header = None)
    Exp_list = Exp_label.iloc[:, 0].values.tolist()
    Exp = pd.DataFrame(np.loadtxt(Datapath).T,columns = Exp_label)
    # labels
    y = pd.read_csv(ypath,header = None)

    Exp_0 = pd.concat([Exp,y], axis=1, ignore_index=True)

    Exp_0.columns = Exp.columns.append(pd.Index(['Type']))

    # 计算 SWCSS
    twiss = 0
    tss = np.sum((Exp_0.drop('Type', axis=1).values - np.mean(Exp_0.drop('Type', axis=1).values)) ** 2)
    for i in set(Exp_0['Type']):
        X_i = Exp_0.loc[Exp_0['Type'] == i]
        wiss = np.sum((X_i.drop('Type', axis=1).values - np.mean(X_i.drop('Type', axis=1).values)) ** 2)
        twiss += wiss

    swiss0 = twiss/tss
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

    # 计算 SWCSS
    twiss = 0
    tss = np.sum((ExpConcat.drop('Type', axis=1).values - np.mean(ExpConcat.drop('Type', axis=1).values)) ** 2)
    for i in set(ExpConcat['Type']):
        X_i = ExpConcat.loc[ExpConcat['Type'] == i]
        wiss = np.sum((X_i.drop('Type', axis=1).values - np.mean(X_i.drop('Type', axis=1).values)) ** 2)
        twiss += wiss

    swiss1 = twiss/tss
    return swiss0,swiss1

ypath = 'RealData/PAM50label664.txt'

Labelpath = 'RealData/Exp664_genes.txt'
Datapath = "RealData/Exp664.txt"
Respath = "./Results/Exp_score.csv"

s10,s11 = SWISS(Labelpath,Datapath,ypath,Respath)

Labelpath = 'RealData/miRNA664_miRNA.txt'
Datapath = "RealData/miRNA664.txt"
Respath = "./Results/miRNA_score.csv"

s20,s21 = SWISS(Labelpath,Datapath,ypath,Respath)

Labelpath = 'RealData/Meth664_probes.txt'
Datapath = "RealData/Meth664.txt"
Respath = "./Results/Meth_score.csv"

s30, s31 = SWISS(Labelpath,Datapath,ypath,Respath)

print(s10,s11,s20,s21,s30,s31)

# K-Mean
km = KMeans(n_clusters=4).fit(Exp)
#SScore0 = silhouette_score(Exp, km.labels_, metric='euclidean')
SScore1 = silhouette_score(Exp, y, metric='euclidean')
DBS1 = davies_bouldin_score(Exp, y)
CHS1 = calinski_harabasz_score(Exp, y)


listname = ExpRes[:,0]
FilterRes = []
for i in range(len(listname)):
    list_index: int = Exp_list.index(listname[i])
    FilterRes.append(list_index)

ExpFilter = Exp.iloc[:,FilterRes]

# Random forest
scaler = StandardScaler()
ExpFilter = scaler.fit_transform(ExpFilter)

## K-Mean
km = KMeans(n_clusters=4).fit(ExpFilter)
SScore2 = silhouette_score(ExpFilter, y, metric='euclidean')
print(SScore1,SScore2)
DBS2 = davies_bouldin_score(ExpFilter, y)
CHS2 = calinski_harabasz_score(ExpFilter, y)


'''
for items in itertools.combinations(range(len(ExpFilter.columns)-1), 2):
    colors = np.array(['red','green','blue','yellow','orange'])
    aaa = ExpFilter.columns[items[0]]
    sns.set(font='SimHei',style='ticks')
    plt.figure(figsize=(8,5))
    plt.scatter(ExpFilter[ExpFilter.columns[items[0]]],ExpFilter[ExpFilter.columns[items[1]]],c=colors[ExpFilter['cluster']],edgecolor='w',s=100)

    plt.tick_params(labelsize = 13)
    plt.title('K-Means')
    plt.xlabel(ExpFilter.columns[items[0]])
    plt.ylabel(ExpFilter.columns[items[1]])

    sns.despine()
'''
#plt.show()