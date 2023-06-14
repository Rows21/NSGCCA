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

# Hyper Params Section
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using", torch.cuda.device_count(), "GPUs")

# Result Data
ExpRes = pd.read_csv("./Results/Exp_score.csv").values
MethRes = pd.read_csv("./Results/Meth_score.csv").values
miRNARes = pd.read_csv("./Results/miRNA_score.csv").values

# Original Data
Exp_label = pd.read_csv('RealData/Exp664_genes.txt', sep='\t',header = None)
Exp_list = Exp_label.iloc[:, 0].values.tolist()
Exp = pd.DataFrame(np.loadtxt("RealData/Exp664.txt").T,columns = Exp_label)

Meth_label = pd.read_csv('RealData/Meth664_probes.txt', sep='\t',header = None)
Meth_list = Meth_label.iloc[:, 0].values.tolist()
Meth = pd.DataFrame(np.loadtxt("RealData/Meth664.txt").T,columns = Meth_label)

miRNA_label = pd.read_csv('RealData/miRNA664_miRNA.txt', sep='\t',header = None)
miRNA_list = miRNA_label.iloc[:, 0].values.tolist()
miRNA = pd.DataFrame(np.loadtxt("RealData/miRNA664.txt").T,columns = miRNA_label)

# labels
y = pd.read_csv('RealData/PAM50label664.txt',header = None).values

## Random forest
scaler = StandardScaler()
Exp = scaler.fit_transform(Exp)

X_train, X_test,y_train,y_test = train_test_split(Exp,y,test_size=0.3,random_state=0)
clf = RandomForestClassifier(n_estimators=50,criterion='entropy',random_state=0)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)

cr = classification_report(y_test,y_pred)

acc = accuracy_score(y_test,y_pred)

## K-Mean
km = KMeans(n_clusters=4).fit(Exp)
SScore1 = silhouette_score(Exp, km.labels_, metric='euclidean')
DBS1 = davies_bouldin_score(Exp, km.labels_)
CHS1 = calinski_harabasz_score(Exp, km.labels_)
Exp['cluster'] = km.labels_

'''
colors = np.array(['red','green','blue','yellow'])

sns.set(font='SimHei',style='ticks')
plt.figure(figsize=(8,5))
plt.scatter(Exp[Exp.columns[1]],Exp[Exp.columns[2]],c=colors[Exp['cluster']],edgecolor='w',s=100)

plt.tick_params(labelsize = 13)
plt.title('K-Means')
plt.xlabel('calories')
plt.ylabel('alcohol')
#plt.show()
sns.despine()
'''
listname = ExpRes[:,0]
FilterRes = []
for i in range(len(listname)):
    list_index: int = Exp_list.index(listname[i])
    FilterRes.append(list_index)

ExpFilter = Exp.iloc[:,FilterRes]
print(ExpFilter.shape)

## Random forest
scaler = StandardScaler()
ExpFilter = scaler.fit_transform(ExpFilter)

X_train, X_test,y_train,y_test = train_test_split(ExpFilter,y,test_size=0.3,random_state=0)
clf = RandomForestClassifier(n_estimators=50,criterion='entropy',random_state=0)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)

cr = classification_report(y_test,y_pred)

acc = accuracy_score(y_test,y_pred)

## K-Mean
km = KMeans(n_clusters=4).fit(ExpFilter)
SScore2 = silhouette_score(ExpFilter, km.labels_, metric='euclidean')
DBS2 = davies_bouldin_score(ExpFilter, km.labels_)
CHS2 = calinski_harabasz_score(ExpFilter, km.labels_)
ExpFilter['cluster'] = km.labels_

print(ExpFilter['cluster'])
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
