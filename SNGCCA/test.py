import torch
import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score

Exp_label = pd.read_csv('C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/Exp664_genes.txt', sep='\t',header = None)
Exp_list = Exp_label.iloc[:, 0].values.tolist()
Exp = pd.DataFrame(np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/Exp664.txt").T,columns = Exp_label)

Meth_label = pd.read_csv('C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/Meth664_probes.txt', sep='\t',header = None)
Meth_list = Meth_label.iloc[:, 0].values.tolist()
Meth = pd.DataFrame(np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/Meth664.txt").T,columns = Meth_label)

miRNA_label = pd.read_csv('C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/miRNA664_miRNA.txt', sep='\t',header = None)
miRNA_list = miRNA_label.iloc[:, 0].values.tolist()
miRNA = pd.DataFrame(np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/miRNA664.txt").T,columns = miRNA_label)

y = pd.read_csv('C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/PAM50label664.txt',header = None)

Exp_value = np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/Exp664.txt")
Meth_value = np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/Meth664.txt")
miRNA_value = np.loadtxt("C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/miRNA664.txt")

Score1 = pd.read_csv('C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/ressg/SGCCA_u.csv')['V1']
Score2 = pd.read_csv('C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/ressg/SGCCA_v.csv')['V1']
Score3 = pd.read_csv('C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/ressg/SGCCA_w.csv')['V1']

Exp_df_S = pd.DataFrame({'Name': Exp_list, 'Score': Score1})
Meth_df_S = pd.DataFrame({'Name': Meth_list, 'Score': Score2})
miRNA_df_S = pd.DataFrame({'Name': miRNA_list, 'Score': Score3})

Exp_df_S = pd.concat([Exp_df_S,pd.DataFrame(Exp_value)],axis=1)
Meth_df_S = pd.concat([Meth_df_S,pd.DataFrame(Meth_value)],axis=1)
miRNA_df_S = pd.concat([miRNA_df_S,pd.DataFrame(miRNA_value)],axis=1)

Filter_Exp = Exp_df_S[abs(Exp_df_S['Score'])> 0.03]
Filter_Meth = Meth_df_S[abs(Meth_df_S['Score']) > 0.03]
Filter_miRNA = miRNA_df_S[abs(miRNA_df_S['Score']) > 0.03]

print(len(Filter_Exp), len(Filter_Meth), len(Filter_miRNA))
print(davies_bouldin_score(Filter_Exp.iloc[:,2:].T, y[0]), davies_bouldin_score(Filter_Meth.iloc[:,2:].T, y[0]), davies_bouldin_score(Filter_miRNA.iloc[:,2:].T, y[0]))
scorepath = 'C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/ressg/'
Filter_Exp.iloc[:,:2].to_csv(scorepath + 'Exp_score.csv', index=False)
Filter_Meth.iloc[:,:2].to_csv(scorepath + 'Meth_score.csv', index=False)
Filter_miRNA.iloc[:,:2].to_csv(scorepath + 'miRNA_score.csv', index=False)

Filter_Exp.iloc[:,2:].to_csv(scorepath + 'Exp_sgcca.txt', index=False)
Filter_Meth.iloc[:,2:].to_csv(scorepath + 'Meth_sgcca.txt', index=False)
Filter_miRNA.iloc[:,2:].to_csv(scorepath + 'miRNA_sgcca.txt', index=False)

Score = pd.read_csv('C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/resk/KSSHIBA.csv').iloc[:,1].to_list()

Exp_df_S = pd.DataFrame({'Name': Exp_list, 'Score': Score[:2642]})
Meth_df_S = pd.DataFrame({'Name': Meth_list, 'Score': Score[2642:(2642+3298)]})
miRNA_df_S = pd.DataFrame({'Name': miRNA_list, 'Score': Score[(2642+3298):(2642+3298+437)]})

Exp_df_S = pd.concat([Exp_df_S,pd.DataFrame(Exp_value)],axis=1)
Meth_df_S = pd.concat([Meth_df_S,pd.DataFrame(Meth_value)],axis=1)
miRNA_df_S = pd.concat([miRNA_df_S,pd.DataFrame(miRNA_value)],axis=1)

Filter_Exp = Exp_df_S[Exp_df_S['Score'] > np.mean(Exp_df_S['Score'])]
Filter_Meth = Meth_df_S[Meth_df_S['Score'] > np.mean(Meth_df_S['Score'])]
Filter_miRNA = miRNA_df_S[miRNA_df_S['Score'] > 0.0003]

print(len(Filter_Exp), len(Filter_Meth), len(Filter_miRNA))
print(davies_bouldin_score(Filter_Exp.iloc[:,2:].T, y[0]), davies_bouldin_score(Filter_Meth.iloc[:,2:].T, y[0]), davies_bouldin_score(Filter_miRNA.iloc[:,2:].T, y[0]))
scorepath = 'C:/Users/Programer/Documents/GitHub/SGCCA_HSIC/SNGCCA/RealData/resk/'
Filter_Exp.iloc[:,:2].to_csv(scorepath + 'Exp_score.csv', index=False)
Filter_Meth.iloc[:,:2].to_csv(scorepath + 'Meth_score.csv', index=False)
Filter_miRNA.iloc[:,:2].to_csv(scorepath + 'miRNA_score.csv', index=False)

Filter_Exp.iloc[:,2:].to_csv(scorepath + 'Exp_sgcca.txt', index=False)
Filter_Meth.iloc[:,2:].to_csv(scorepath + 'Meth_sgcca.txt', index=False)
Filter_miRNA.iloc[:,2:].to_csv(scorepath + 'miRNA_sgcca.txt', index=False)
