import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as pio


def plot_tsne(Exp,y,perplexity=30,n_iter=1000,learning_rate=200):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate)
    X_tsne = tsne.fit_transform(Exp)

    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y.astype(int), color_continuous_scale='Jet')
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
    )
    fig.update_layout({
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white'
    })
    fig.update_layout({
        'xaxis': {'linecolor': 'black', 'tickfont': {'color': 'black'}},
        'yaxis': {'linecolor': 'black', 'tickfont': {'color': 'black'}}
    })
    fig.update_traces(marker={'size': 12, 'showscale': False})
    fig.show()

def method_tsne(method,num='res1/'):
    root = 'E:/GitHub/SNGCCA/SNGCCA/RealData/' + num
    if method == 'SNGCCA':
        Exp_filter = pd.read_csv(root + 'Exp_SNGCCA.txt').values
        Meth_filter = pd.read_csv(root + 'Meth_SNGCCA.txt').values
        miRNA_filter = pd.read_csv(root + 'miRNA_SNGCCA.txt').values
    elif method == 'DGCCA':
        Exp_filter = pd.read_csv(root + 'Exp_dgcca.txt').values
        Meth_filter = pd.read_csv(root + 'Meth_dgcca.txt').values
        miRNA_filter = pd.read_csv(root + 'miRNA_dgcca.txt').values
    elif method == 'SGCCA':
        Exp_filter = pd.read_csv(root + 'Exp_sgcca.txt').values
        Meth_filter = pd.read_csv(root + 'Meth_sgcca.txt').values
        miRNA_filter = pd.read_csv(root + 'miRNA_sgcca.txt').values
    elif method == 'KSSHIBA':
        Exp_filter = pd.read_csv(root + 'Exp_k.txt').values
        Meth_filter = pd.read_csv(root + 'Meth_k.txt').values
        miRNA_filter = pd.read_csv(root + 'miRNA_k.txt').values
    merged_data = np.vstack((Exp_filter, Meth_filter, miRNA_filter))
    
    #plot_tsne(Exp_filter.T,y,perplexity=50,n_iter=4000,learning_rate=400)
    #plot_tsne(Meth_filter.T,y,perplexity=10,n_iter=5000,learning_rate=200)
    #plot_tsne(miRNA_filter.T,y)
    plot_tsne(merged_data.T,y)

num = 'res1/'
datapath = 'E:/GitHub/SNGCCA/SNGCCA/RealData/'
scorepath = datapath + num

y = np.loadtxt(datapath + 'PAM50label664.txt').astype(int)
np.savetxt('y.csv', y.T, delimiter=',')
Exp = np.loadtxt(datapath + 'Exp664.txt')
Meth = np.loadtxt(datapath + 'Meth664.txt')
miRNA = np.loadtxt(datapath + 'miRNA664.txt')
merged_data = np.vstack((Exp, Meth, miRNA))
#plot_tsne(Exp.T,y)
#plot_tsne(Meth.T,y)
#plot_tsne(miRNA.T,y)
#plot_tsne(merged_data.T,y)

# SNGCCA
method_tsne('SNGCCA',num='res10/')
#method_tsne('DGCCA',num='resdg/')
#method_tsne('SGCCA',num='ressg/')
#method_tsne('KSSHIBA',num='resk/')


