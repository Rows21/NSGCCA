import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.optimize import minimize

# Generate synthetic data
np.random.seed(42)
data = [np.random.randn(100, 80) for _ in range(3)]  # 3 datasets of size 100x100

# Kernel matrices and empirical HSIC calculation
def compute_hsic_matrices(data, sigma="median"):
    K_matrices = []
    n = data[0].shape[0]
    
    for d in data:
        if sigma == "median":
            pairwise_distances = np.sqrt(((d[:, None] - d) ** 2).sum(-1))
            sigma_value = np.median(pairwise_distances)
        else:
            sigma_value = sigma
        K_matrices.append(pairwise_kernels(d, metric="rbf", gamma=1 / (2 * sigma_value ** 2)))
    
    H = np.eye(n) - np.ones((n, n)) / n
    M_matrices = []
    
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            M = np.einsum("ij,jk,kl->il", K_matrices[i], H, K_matrices[j])
            M = np.einsum("ij,jk->ik", M, H) / (n ** 2)
            M_matrices.append(M)
    return M_matrices

# Soft-thresholding for L1 norm constraint
def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

# Solve subproblem for a single u_k
def solve_subproblem(M_kt, u_t, s_k, max_iter=100, tol=1e-6):
    p = M_kt.shape[0]
    u_k = np.random.randn(p)
    u_k /= np.linalg.norm(u_k, 2)  # Initialize with L2 norm constraint
    
    for _ in range(max_iter):
        gradient = 2 * M_kt @ u_t
        u_k_new = soft_threshold(u_k + gradient, 1.0 / s_k)
        u_k_new /= max(1, np.linalg.norm(u_k_new, 2))  # Project to L2 ball
        
        if np.linalg.norm(u_k_new - u_k, 2) < tol:
            break
        u_k = u_k_new
    
    return u_k

# Block Coordinate Descent
def optimize_hsic(data, s_k_list, max_iter=50, tol=1e-5):
    K = len(data)
    M_matrices = compute_hsic_matrices(data)
    
    u_list = [np.random.randn(data[k].shape[1]) for k in range(K)]
    u_list = [u / np.linalg.norm(u, 2) for u in u_list]
    
    for _ in range(max_iter):
        for k in range(K):
            u_t_sum = np.zeros_like(u_list[k])
            for t in range(K):
                if t != k:
                    M_kt = M_matrices[min(k, t)][max(k, t)]
                    u_t_sum += M_kt @ u_list[t]
            
            u_list[k] = solve_subproblem(u_t_sum, u_list[k], s_k_list[k])
    
    return u_list

# Parameters
s_k_list = [0.1, 0.1, 0.1]  # Example values within [1, sqrt(p_k)]
u_list = optimize_hsic(data, s_k_list)

print("Optimized u_k vectors:")
for k, u_k in enumerate(u_list):
    print(f"u_{k + 1}: {u_k}")
