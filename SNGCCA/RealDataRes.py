import pandas as pd
import numpy as np
from validation_method import swiss_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

Exp_label = pd.read_csv('./SNGCCA/RealData/Exp664_genes.txt', sep='\t',header = None)
Exp_list = Exp_label.iloc[:, 0].values.tolist()
Exp = pd.DataFrame(np.loadtxt("./SNGCCA/RealData/Exp664.txt").T,columns = Exp_label)

Meth_label = pd.read_csv('./SNGCCA/RealData/Meth664_probes.txt', sep='\t',header = None)
Meth_list = Meth_label.iloc[:, 0].values.tolist()
Meth = pd.DataFrame(np.loadtxt("./SNGCCA/RealData/Meth664.txt").T,columns = Meth_label)

miRNA_label = pd.read_csv('./SNGCCA/RealData/miRNA664_miRNA.txt', sep='\t',header = None)
miRNA_list = miRNA_label.iloc[:, 0].values.tolist()
miRNA = pd.DataFrame(np.loadtxt("./SNGCCA/RealData/miRNA664.txt").T,columns = miRNA_label)

y = pd.read_csv('./SNGCCA/RealData/PAM50label664.txt',header = None)

Exp_value = np.loadtxt("./SNGCCA/RealData/Exp664.txt")
Meth_value = np.loadtxt("./SNGCCA/RealData/Meth664.txt")
miRNA_value = np.loadtxt("./SNGCCA/RealData/miRNA664.txt")

Score1 = pd.read_csv('./SNGCCA/RealData/u1.csv', header=None).iloc[1:,1].astype(float).tolist()
Score2 = pd.read_csv('./SNGCCA/RealData/u2.csv', header=None).iloc[1:,1].astype(float).tolist()
Score3 = pd.read_csv('./SNGCCA/RealData/u3.csv', header=None).iloc[1:,1].astype(float).tolist()

Exp_df_S = pd.DataFrame({'Name': Exp_list, 'Score': Score1})
Meth_df_S = pd.DataFrame({'Name': Meth_list, 'Score': Score2})
miRNA_df_S = pd.DataFrame({'Name': miRNA_list, 'Score': Score3})

Exp_df_S = pd.concat([Exp_df_S,pd.DataFrame(Exp_value)],axis=1).iloc[abs(Exp_df_S['Score']).argsort()[::-1]]
Meth_df_S = pd.concat([Meth_df_S,pd.DataFrame(Meth_value)],axis=1).iloc[abs(Meth_df_S['Score']).argsort()[::-1]]
miRNA_df_S = pd.concat([miRNA_df_S,pd.DataFrame(miRNA_value)],axis=1).iloc[abs(miRNA_df_S['Score']).argsort()[::-1]]

Filter_Exp = Exp_df_S[abs(Exp_df_S['Score']) > 0.05]
Filter_Meth = Meth_df_S[abs(Meth_df_S['Score']) > 0.05]
Filter_miRNA = miRNA_df_S[abs(miRNA_df_S['Score']) > 0.05]

print(len(Filter_Exp), len(Filter_Meth), len(Filter_miRNA))

swiss = swiss_score(Filter_Exp.iloc[:,2:].T, y[0])
db = davies_bouldin_score(Filter_Exp.iloc[:,2:].T, y[0])
ss = silhouette_score(Filter_Exp.iloc[:,2:].T, y[0])
ch = calinski_harabasz_score(Filter_Exp.iloc[:,2:].T, y[0])
print(swiss, db, ss, ch)

Filter_Exp.to_csv('Exp_score.csv', index=False)
Filter_Meth.to_csv('Meth_score.csv', index=False)
Filter_miRNA.to_csv('miRNA_score.csv', index=False)