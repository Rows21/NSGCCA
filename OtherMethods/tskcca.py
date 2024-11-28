import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from utils import rbf_kl

n = 100  # Number of samples
K = 3    # Number of views
p = [5, 6, 4]  # Number of components in each view

def gram_matrix(X):
    """Compute centered Gaussian kernel Gram matrix."""
    # Pairwise squared Euclidean distances
    pairwise_dist = pairwise_distances(X, metric="euclidean")**2
    upper_off_diag = pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)]
    sigma2 = np.median(upper_off_diag)
    K = np.exp(-pairwise_dist / (2 * sigma2))
    # Parameters
    return K

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


# Generate example data (replace with actual data)
X_data = {
    (k, j): np.random.rand(n, 1) for k in range(K) for j in range(p[k])
}

# Compute centered Gram matrices
K_matrices = {
    (k, j): gram_matrix(X_data[(k, j)])[0] for k in range(K) for j in range(p[k])
}
#K_matrices = [gram_matrix(d.T) for d in data]
#cK_matrices = [rbf_kl(K) for K in K_matrices]
#u = [np.random.randn(80) for _ in range(3)]


M_matrices = []
H = np.eye(n) - np.ones((n, n)) / n
for i in range(K):
        for j in range(i + 1, K):
            M = np.einsum("ij,jk,kl->il", K_matrices[i], H, K_matrices[j])
            M = np.einsum("ij,jk->ik", M, H) / (n ** 2)
            M_matrices.append(M)
a = 1