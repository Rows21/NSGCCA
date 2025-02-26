import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import time

def gram_matrix(X):
    """Compute centered Gaussian kernel Gram matrix."""
    # Pairwise squared Euclidean distances
    pairwise_dist = pairwise_distances(X, metric="euclidean")**2
    upper_off_diag = pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)]
    sigma2 = np.median(upper_off_diag)
    K = np.exp(-pairwise_dist / (2 * sigma2))
    # Parameters
    return K

def l2n(vec):
    """ computes "safe" l2 norm """
    norm = np.sqrt(np.sum(vec**2))
    if norm == 0:
        norm = 0.05
    return norm

def binary_search(argu, sumabs):
    """ 
    """
    if l2n(argu) == 0 or np.sum(np.abs(argu/l2n(argu))) <= sumabs:
        return 0 

    lam1 = 0
    lam2 = np.max(np.abs(argu)) - 1e-5

    for idx in range(150):
        su = soft(argu, (lam1 + lam2) / 2)
        if np.sum(np.abs(su/l2n(su))) < sumabs:
            lam2 = (lam1 + lam2) / 2
        else:
            lam1 = (lam1 + lam2) / 2
        if lam2 - lam1 < 1e-6:
            return (lam1 + lam2) / 2

    print("Warning. Binary search did not quite converge..")
    return (lam1 + lam2) / 2

def soft(x_, d_):
    """ soft-thresholding operator """
    return np.sign(x_) * (np.abs(x_)-d_).clip(min=0)

def soft_thresholding(x, delta):
    """
    Apply soft-thresholding to a vector x with threshold delta.
    """
    return np.sign(x) * np.maximum(np.abs(x) - delta, 0)

def find_delta_scaled(Xv, c1, tol=1e-5):
    """
    Find the exact Delta_1 such that ||u||_1 = c1, where
    u = soft(Xv, Delta_1) / ||soft(Xv, Delta_1)||_2.

    Parameters:
    - Xv: Input vector.
    - c1: Target L1 norm for u.
    - tol: Convergence tolerance.

    Returns:
    - delta: Threshold value Delta_1.
    """
    delta_low = 0
    delta_high = np.max(np.abs(Xv))  # Upper bound for Delta_1

    while delta_high - delta_low > tol:
        delta = (delta_low + delta_high) / 2

        # Apply soft-thresholding
        u_soft = soft_thresholding(Xv, delta)

        # Normalize to have L2 norm = 1 (if non-zero)
        if np.linalg.norm(u_soft, 2) > 0:
            u_normalized = u_soft / np.linalg.norm(u_soft, 2)
        else:
            u_normalized = np.zeros_like(Xv)

        # Compute L1 norm of the normalized vector
        l1_norm = np.sum(np.abs(u_normalized))

        # Adjust delta based on the scaled L1 norm
        if l1_norm > c1:
            delta_low = delta  # Increase Delta_1
        else:
            delta_high = delta  # Decrease Delta_1

    return (delta_low + delta_high) / 2

# subproblem 
def _subproblem(M_matrices, c, pk, k, u):
    sum_term = np.zeros((pk, 1))
    for key, matrix in M_matrices.items():
        if k in key:
            other_value = key[0] if key[1] == k else key[1]
            if key[1] == k:  # 如果 2 在第二位，则转置矩阵
                matrix = matrix.T
            sum_term += matrix @ u[other_value]
                        
    delta = binary_search(sum_term, c)
    u_new = soft(sum_term, delta)
    u_new /= l2n(u_new)
    #print(np.linalg.norm(u_new))
    return u_new
        
def _hyper_tuning(M_matrices, s_k_range, p, K, u, max_iter=1000):
    best_s_k = {}
    best_objective = -np.inf
    diff_list = [+np.inf] * K
    diff = np.inf
    i = 0

    while diff > 5e-2 and i < max_iter:
        i += 1
        for s_k in s_k_range:
            #print(f"Iteration {i}")
            for k in range(K):
                diff_old = diff
                u_new = _subproblem(M_matrices, s_k, p[k], k, u)
                diff_list[k] = np.max(np.abs(u_new - u[k]))
                
                diff = max(diff_list)
                if diff < diff_old:
                    u[k] = u_new
                #print(f"View {k}: {diff}")

            objective_value = sum([u[s].T @ M_matrices[(s,t)] @ u[t] for s in range(K) for t in range(s+1, K)]).item()
            #print(objective_value)
            
            if objective_value > best_objective:
                best_objective = objective_value
                best_s_k[k] = s_k
                best_u = u
    
    return best_s_k, best_u

def tskgcca(data):
    
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
        (k, j): gram_matrix(X_data[(k, j)]) for k in range(K) for j in range(p[k])
    }

    # Random initialization
    u = {
        k: np.random.rand(p[k], 1) for k in range(K)
    }
    # normalization
    for k in range(K):
        u[k] = u[k] / l2n(u[k])

    # Compute M matrices
    M_matrices = {}
    H = np.eye(n) - np.ones((n, n)) / n
    for s in range(K):
        for t in range(s + 1, K):
            # Precompute H-transformed matrices for view s and view t
            K_s_transformed = [K_matrices[(s, i)] @ H for i in range(p[s])]
            K_t_transformed = [K_matrices[(t, j)] @ H for j in range(p[t])]

            # Calculate M_st using broadcasting
            M_st = np.array([
                [np.trace(K_s_transformed[i] @ K_t_transformed[j]) / (n ** 2) for j in range(p[t])]
                for i in range(p[s])
            ])
            
            # Store the result in the dictionary
            M_matrices[(s, t)] = M_st
    
    for _ in range(5):
        s_k_range = np.linspace(1, np.sqrt(p[k]), 10)
        #print('start')
        best_s_k, best_u = _hyper_tuning(M_matrices, s_k_range, p, K, u)
    return best_s_k, best_u

if __name__ == "__main__":

    combinations = [
        [100, 30, 5],
        [100, 50, 5],
        [100, 100, 5],
        [100, 200, 5],
        #[200, 100, 5],
        #[400, 100, 5],
        [100, 100, 10],
        [100, 100, 20]
    ]
    
    for mode in [1]:
        for params in combinations:
            
            t = []
            u1 = []
            u2 = []
            u3 = []
            
            N = params[0]
            P = params[1]
            S = params[2]
            root = 'E:/res/SNGCCA/SNGCCA/'
            if mode == 1:
                    folder = 'Linear/'
            else:
                    folder = 'Nonlinear/'
            data_path = root + 'Data/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/'
            print(params)
            for r in range(100):
                #views = create_synthData_new(v=5,N=N,mode=1,F=30)
                view1 = np.genfromtxt(data_path + 'data' + str(1) + '_' + str(r) + '.csv', delimiter=',')
                view2 = np.genfromtxt(data_path + 'data' + str(2) + '_' + str(r) + '.csv', delimiter=',')
                view3 = np.genfromtxt(data_path + 'data' + str(3) + '_' + str(r) + '.csv', delimiter=',')
                views = [view1, view2, view3]
                #print(f'input views shape :')
                #for i, view in enumerate(views):
                #    print(f'view_{i} :  {view.shape}')
                start_time = time.time()
                
                s_k, u = tskcca(views)
                end_time = time.time()       
                t.append(end_time - start_time)
                u1.append(u[0])
                u2.append(u[1])
                u3.append(u[2])
            
            merged_array = merged_array = np.empty((100,P))
            path = 'E:/res/SNGCCA/SNGCCA/Simulation/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/'
        
            for i, arr in enumerate(u1):
                merged_array[i] = u1[i].flatten()
            np.savetxt(path + 'tskcca_u1.csv', merged_array, delimiter=',')
            for i, arr in enumerate(u2):
                merged_array[i] = u2[i].flatten()
            np.savetxt(path + 'tskcca_u2.csv', merged_array, delimiter=',')
            for i, arr in enumerate(u3):
                merged_array[i] = u3[i].flatten()
            np.savetxt(path + 'tskcca_u3.csv', merged_array, delimiter=',')
            np.savetxt(path + 'tskcca_t.csv', t, delimiter=',')
                