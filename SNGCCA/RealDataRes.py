import pandas as pd
import numpy as np
from validation_method import swiss_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


num = 'ressng/'
datapath = 'E:/GitHub/SNGCCA/SNGCCA/RealData/'
scorepath = datapath + num

Exp_label = pd.read_csv(datapath + 'Exp664_genes.txt', sep='\t',header = None)
Exp_list = Exp_label.iloc[:, 0].values.tolist()
Exp = pd.DataFrame(np.loadtxt(datapath + 'Exp664.txt').T,columns = Exp_label)

Meth_label = pd.read_csv(datapath + 'Meth664_probes.txt', sep='\t',header = None)
Meth_list = Meth_label.iloc[:, 0].values.tolist()
Meth = pd.DataFrame(np.loadtxt(datapath + '/Meth664.txt').T,columns = Meth_label)

miRNA_label = pd.read_csv(datapath + 'miRNA664_miRNA.txt', sep='\t',header = None)
miRNA_list = miRNA_label.iloc[:, 0].values.tolist()
miRNA = pd.DataFrame(np.loadtxt(datapath + 'miRNA664.txt').T,columns = miRNA_label)

y = pd.read_csv(datapath + 'PAM50label664.txt',header = None)

Exp_value = np.loadtxt(datapath + 'Exp664.txt')
Meth_value = np.loadtxt(datapath + 'Meth664.txt')
miRNA_value = np.loadtxt(datapath + 'miRNA664.txt')

    
if num == 'resk/':
    Score = pd.read_csv(scorepath + 'KSSHIBA.csv', header=None).iloc[1:,1].astype(float).tolist()
    Score1 = Score[:len(Exp_list)]
    Score2 = Score[len(Exp_list):len(Exp_list)+len(Meth_list)]
    Score3 = Score[len(Exp_list)+len(Meth_list):]
else:
    Score1 = pd.read_csv(scorepath + 'U1.csv', header=None).iloc[1:,1].astype(float).tolist()
    Score2 = pd.read_csv(scorepath + 'U2.csv', header=None).iloc[1:,1].astype(float).tolist()
    Score3 = pd.read_csv(scorepath + 'U3.csv', header=None).iloc[1:,1].astype(float).tolist()


Score1 = Score1#/np.linalg.norm(Score1)
Score2 = Score2#/np.linalg.norm(Score2)
Score3 = Score3#/np.linalg.norm(Score3)

Exp_df_S = pd.DataFrame({'Name': Exp_list, 'Score': Score1})
Meth_df_S = pd.DataFrame({'Name': Meth_list, 'Score': Score2})
miRNA_df_S = pd.DataFrame({'Name': miRNA_list, 'Score': Score3})

Exp_df_S = pd.concat([Exp_df_S,pd.DataFrame(Exp_value)],axis=1).iloc[abs(Exp_df_S['Score']).argsort()[::-1]]
Meth_df_S = pd.concat([Meth_df_S,pd.DataFrame(Meth_value)],axis=1).iloc[abs(Meth_df_S['Score']).argsort()[::-1]]
miRNA_df_S = pd.concat([miRNA_df_S,pd.DataFrame(miRNA_value)],axis=1).iloc[abs(miRNA_df_S['Score']).argsort()[::-1]]

print(davies_bouldin_score(Exp_df_S.iloc[:,2:].T, y[0]), davies_bouldin_score(Meth_df_S.iloc[:,2:].T, y[0]), davies_bouldin_score(miRNA_df_S.iloc[:,2:].T, y[0]))
print(silhouette_score(Exp_df_S.iloc[:,2:].T, y[0]), silhouette_score(Meth_df_S.iloc[:,2:].T, y[0]), silhouette_score(miRNA_df_S.iloc[:,2:].T, y[0]))
print(calinski_harabasz_score(Exp_df_S.iloc[:,2:].T, y[0]), calinski_harabasz_score(Meth_df_S.iloc[:,2:].T, y[0]), calinski_harabasz_score(miRNA_df_S.iloc[:,2:].T, y[0]))
#print(swiss, db/3, ss/3, ch/3)

Filter_Exp = Exp_df_S[abs(Exp_df_S['Score']) > 0.01]
Filter_Meth = Meth_df_S[abs(Meth_df_S['Score']) > 0.01]
Filter_miRNA = miRNA_df_S[abs(miRNA_df_S['Score']) > 0.01]

print(len(Filter_Exp), len(Filter_Meth), len(Filter_miRNA))
    
print(davies_bouldin_score(Filter_Exp.iloc[:,2:].T, y[0]), davies_bouldin_score(Filter_Meth.iloc[:,2:].T, y[0]), davies_bouldin_score(Filter_miRNA.iloc[:,2:].T, y[0]))
print(silhouette_score(Filter_Exp.iloc[:,2:].T, y[0]), silhouette_score(Filter_Meth.iloc[:,2:].T, y[0]), silhouette_score(Filter_miRNA.iloc[:,2:].T, y[0]))
print(calinski_harabasz_score(Filter_Exp.iloc[:,2:].T, y[0]), calinski_harabasz_score(Filter_Meth.iloc[:,2:].T, y[0]), calinski_harabasz_score(Filter_miRNA.iloc[:,2:].T, y[0]))
#print(swiss, db/3, ss/3, ch/3)

Filter_Exp.iloc[:,:2].to_csv(scorepath + 'Exp_score.csv', index=False)
Filter_Meth.iloc[:,:2].to_csv(scorepath + 'Meth_score.csv', index=False)
Filter_miRNA.iloc[:,:2].to_csv(scorepath + 'miRNA_score.csv', index=False)

Filter_Exp.iloc[:,2:].to_csv(scorepath + 'Exp_SNGCCA.txt', index=False)
Filter_Meth.iloc[:,2:].to_csv(scorepath + 'Meth_SNGCCA.txt', index=False)
Filter_miRNA.iloc[:,2:].to_csv(scorepath + 'miRNA_SNGCCA.txt', index=False)

d1 = Filter_Exp.iloc[:,2:].T
d1.columns = Filter_Exp['Name']
d1 = pd.concat([y, d1], axis=1)
d1.to_csv('Exp_SNGCCA.csv', index=False)

d2 = Filter_Meth.iloc[:,2:].T
d2.columns = Filter_Meth['Name']
d2 = pd.concat([y, d2], axis=1)
d2.to_csv('Meth_SNGCCA.csv', index=False)

d3 = Filter_miRNA.iloc[:,2:].T
d3.columns = Filter_miRNA['Name']
d3 = pd.concat([y, d3], axis=1)
d3.to_csv('miRNA_SNGCCA.csv', index=False)
