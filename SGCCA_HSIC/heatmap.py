import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import palettable

Labelpath = 'RealData/Meth664_probes.txt'
Datapath = "RealData/Meth664.txt"
ypath = 'RealData/PAM50label664.txt'
Respath = "./Results/Meth_score.csv"
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

row_c = dict(zip(ExpConcat['Type'].unique(), ['green','yellow','pink','blue']))
plt.figure(dpi=120)
sns.clustermap(ExpConcat, pivot_kws=None,
               method='average',
               metric='correlation',
               z_score=None,
               standard_scale=None,
               figsize=(10, 10),
               cbar_kws=None,
               row_cluster=None,
               col_cluster=True,
               row_linkage=None,
               col_linkage=None,
               row_colors=ExpConcat['Type'].map(row_c),
               col_colors=None,
               mask=None,
               dendrogram_ratio=0.1,
               colors_ratio=0.05,
               cbar_pos=(0.02, 0.8, 0.05, 0.18),
               tree_kws=None,
               cmap='mako')







plt.show()
