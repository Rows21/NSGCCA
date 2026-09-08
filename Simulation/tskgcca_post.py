import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import time
from tskcca import gram_matrix

def update_alpha(z, K_tilde, tau, eps=1e-12):
    """
    z: shape (n, 1)
    K_tilde: shape (n, n)
    tau: scalar
    """

    n = K_tilde.shape[0]

    z = z.reshape(n, 1)
    I = np.eye(n)
    A = tau * I + (1 - tau) * (1 / n) * K_tilde

    # y = A^{-1} z
    y = np.linalg.solve(A, z)

    # denom = z^T K A^{-1} z
    denom = z.T @ K_tilde @ y
    denom = float(denom.item())

    if denom <= eps:
        raise ValueError(f"denom must be positive, got {denom}")

    alpha_next = y / np.sqrt(denom)

    return alpha_next
def tskcca_post(data, stage = 1, u_m = None):
    
    # Generate example data (replace with actual data)
    K = len(data)
    p = [data[i].shape[1] for i in range(K)]
    n = data[0].shape[0]
    
    X_data = {
        (k, j): data[k][:,j].reshape(-1,1) for k in range(K) for j in range(p[k])
        #(k, j): np.random.rand(n, 1) for k in range(K) for j in range(p[k])
    }

    # Compute centered Gram matrices
    K_matrices = {
        (k, j): gram_matrix(X_data[(k, j)])[1] for k in range(K) for j in range(p[k])
    }

    # Random initialization
    alpha_init = {k: np.random.rand(N, 1) for k in range(K)}
    alpha = []
    K_tilde_list = []
    for k in range(K):
        K_tilde = np.zeros((N, N))
        for j in range(p[k]):
            K_tilde += K_matrices[(k, j)] * u_m[k][j]
        H = np.eye(n) - np.ones((n, n)) / n
        K_tilde = H @ K_tilde @ H
        K_tilde = 0.5 * (K_tilde + K_tilde.T)
        K_tilde_list.append(K_tilde)
        A_k = (1 - 0.02) * (1 / N) * K_tilde + 0.02 * np.eye(N)
        denom = alpha_init[k].T @ A_k @ K_tilde @ alpha_init[k]
        alpha.append(denom ** (-0.5) * alpha_init[k])
    
    i = 0
    while i < 1000 and np.linalg.norm(alpha[0] - alpha_init[0]) < 1e-3 and np.linalg.norm(alpha[1] - alpha_init[1]) < 1e-3 and np.linalg.norm(alpha[2] - alpha_init[2]) < 1e-3:
        i += 1
        for k in range(K):
            #alpha_k = alpha[k]
            K_tilde_loop = [K_tilde_list[j] for j in range(K) if j != k]
            alpha_loop = [alpha[j] for j in range(K) if j != k]
            z_k = sum(K_tilde_loop[j] @ alpha_loop[j] for j in range(K-1))
            alpha_next = update_alpha(z_k, K_tilde_list[k], tau = 0.02)
            alpha[k] = alpha_next
            #a = 1

    u = []
    for k in range(K):
        u.append(K_tilde_list[k] @ alpha[k])
    return u

if __name__ == "__main__":

    combinations = [
        [100, 30, 5],
        [100, 50, 5],
        [100, 100, 5],
        [100, 200, 5],
        [200, 100, 5],
        [400, 100, 5],
        [100, 100, 10],
        [100, 100, 20]
    ]

    for mode in [1,2]:
        for params in combinations:
            
            t = []
            u1 = []
            u2 = []
            u3 = []
            
            N = params[0]
            P = params[1]
            S = params[2]
            root = '/Users/rongwu/Desktop/res/SNGCCA/SNGCCA/'
            if mode == 1:
                    folder = 'Linear/'
            else:
                    folder = 'Nonlinear/'
            data_path = root + 'newData/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/'
            print(params)
            
            for r in range(100):

                path_u = '/Users/rongwu/Desktop/res/SNGCCA/SNGCCA/newnewSimulation/pair1/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/'
                u1_m = np.genfromtxt(path_u + 'snr25tskcca_u1.csv', delimiter=',')
                u2_m = np.genfromtxt(path_u + 'snr25tskcca_u2.csv', delimiter=',')
                u3_m = np.genfromtxt(path_u + 'snr25tskcca_u3.csv', delimiter=',')
                u1_m[u1_m < 0.05] = 0
                u2_m[u2_m < 0.05] = 0
                u3_m[u3_m < 0.05] = 0
                u_m = [u1_m[r:r+1,].T, u2_m[r:r+1,].T, u3_m[r:r+1,].T]
                
                #views = create_synthData_new(v=5,N=N,mode=1,F=30)
                view1 = np.genfromtxt(data_path + 'snr25data' + str(1) + '_' + str(r) + '.csv', delimiter=',')
                view2 = np.genfromtxt(data_path + 'snr25data' + str(2) + '_' + str(r) + '.csv', delimiter=',')
                view3 = np.genfromtxt(data_path + 'snr25data' + str(3) + '_' + str(r) + '.csv', delimiter=',')
                views = [view1, view2, view3]
            
                start_time = time.time()
                #s_k, u = tskcca(views, stage = stage)
                u = tskcca_post(views, u_m = u_m)
                end_time = time.time()       
                t.append(end_time - start_time)
                u1.append(u[0])
                u2.append(u[1])
                u3.append(u[2])
            

                