import torch
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score

Exp = None
Meth = None
miRNA = None
for i in [1,3,7,9,10]:
    num = 'res' + str(i) + '/'
    datapath = 'C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/'
    scorepath = datapath + num

    Expi = pd.read_csv(scorepath + 'Exp_score.csv')
    Expi.columns = ['Name',i]
    Methi = pd.read_csv(scorepath + 'Meth_score.csv')
    Methi.columns = ['Name',i]
    miRNAi = pd.read_csv(scorepath + 'miRNA_score.csv')
    miRNAi.columns = ['Name',i]
    if Exp is None:
        Exp = Expi
        Meth = Methi
        miRNA = miRNAi
    else:
        Exp = pd.merge(Exp,Expi,on='Name',how='outer')
        Meth = pd.merge(Meth,Methi,on='Name',how='outer')
        miRNA = pd.merge(miRNA,miRNAi,on='Name',how='outer')

fExp = Exp.dropna(thresh=5)
fExp['mean'] = fExp.iloc[:, -5:].mean(axis=1)
fExp = fExp[(abs(fExp['mean']) >= 0.01)]
fExp_sorted = fExp.reindex(fExp['mean'].abs().sort_values(ascending=False).index)

fMeth = Meth.dropna(thresh=5)
fMeth['mean'] = fMeth.iloc[:, -5:].mean(axis=1)
fMeth = fMeth[(abs(fMeth['mean']) >= 0.01)]
fMeth_sorted = fMeth.reindex(fMeth['mean'].abs().sort_values(ascending=False).index)

fmiRNA = miRNA.dropna(thresh=5)
fmiRNA['mean'] = fmiRNA.iloc[:, -5:].mean(axis=1)
fmiRNA = fmiRNA[(abs(fmiRNA['mean']) >= 0.01)]
fmiRNA_sorted = fmiRNA.reindex(fmiRNA['mean'].abs().sort_values(ascending=False).index)

Exp_label = pd.read_csv(datapath + 'Exp664_genes.txt', sep='\t',header = None)
Exp_list = Exp_label.iloc[:, 0].values.tolist()

Meth_label = pd.read_csv(datapath + 'Meth664_probes.txt', sep='\t',header = None)
Meth_list = Meth_label.iloc[:, 0].values.tolist()

miRNA_label = pd.read_csv(datapath + 'miRNA664_miRNA.txt', sep='\t',header = None)
miRNA_list = miRNA_label.iloc[:, 0].values.tolist()

y = pd.read_csv(datapath + 'PAM50label664.txt',header = None)

Exp_value = np.loadtxt(datapath + 'Exp664.txt')
Meth_value = np.loadtxt(datapath + 'Meth664.txt')
miRNA_value = np.loadtxt(datapath + 'miRNA664.txt')

Exp_df_S = pd.concat([pd.DataFrame({'Name': Exp_list}),pd.DataFrame(Exp_value)],axis=1)
Meth_df_S = pd.concat([pd.DataFrame({'Name': Meth_list}),pd.DataFrame(Meth_value)],axis=1)
miRNA_df_S = pd.concat([pd.DataFrame({'Name': miRNA_list}),pd.DataFrame(miRNA_value)],axis=1)

ResExp = pd.merge(fExp_sorted[['Name','mean']],Exp_df_S,on='Name')
ResMeth = pd.merge(fMeth_sorted[['Name','mean']],Meth_df_S,on='Name')
ResmiRNA = pd.merge(fmiRNA_sorted[['Name','mean']],miRNA_df_S,on='Name')
print(davies_bouldin_score(ResExp.iloc[:,2:].T, y[0]),davies_bouldin_score(ResMeth.iloc[:,2:].T, y[0]), davies_bouldin_score(ResmiRNA.iloc[:,2:].T, y[0]))
savepath = datapath + 'rescv/'
ResExp.iloc[:,:2].to_csv(savepath + 'Exp_score.csv', index=False)
ResMeth.iloc[:,:2].to_csv(savepath + 'Meth_score.csv', index=False)
ResmiRNA.iloc[:,:2].to_csv(savepath + 'miRNA_score.csv', index=False)

ResExp.iloc[:,2:].to_csv(savepath + 'Exp_SNGCCA.txt', index=False)
ResMeth.iloc[:,2:].to_csv(savepath + 'Meth_SNGCCA.txt', index=False)
ResmiRNA.iloc[:,2:].to_csv(savepath + 'miRNA_SNGCCA.txt', index=False)