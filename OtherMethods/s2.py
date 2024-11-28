import numpy as np
import cvxpy as cp
from sklearn.metrics.pairwise import pairwise_distances

# Problem setup
n = 100  # Number of samples
K = 3    # Number of views
p = [5, 6, 4]  # Number of components in each view

# Function to compute centered Gram matrix with Gaussian kernel
def centered_gram_matrix(X):
    """Compute centered Gaussian kernel Gram matrix."""
    # Pairwise squared Euclidean distances
    pairwise_dist = pairwise_distances(X, metric="euclidean")**2
    # Extract the upper off-diagonal elements for median computation
    upper_off_diag = pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)]
    # Compute sigma^2 as the median of the upper off-diagonal distances
    sigma2 = np.median(upper_off_diag)
    # Gaussian kernel
    K = np.exp(-pairwise_dist / (2 * sigma2))
    # Center the kernel matrix
    ones = np.ones((n, n)) / n
    K_centered = K - ones @ K - K @ ones + ones @ K @ ones
    return K_centered

# Generate example data (replace with actual data)
X_data = {
    (k, j): np.random.rand(n, 1) for k in range(K) for j in range(p[k])
}

# Compute centered Gram matrices
K_matrices = {
    (k, j): centered_gram_matrix(X_data[(k, j)]) for k in range(K) for j in range(p[k])
}

# Parameters
epsilon_k = 0.02  # Regularization parameter for each view
s_k_range = {k: np.linspace(1, np.sqrt(p[k]), 10) for k in range(K)}  # Grid of s_k values

# Variables
alpha = {
    (k, j): cp.Variable(n) for k in range(K) for j in range(p[k])
}  # Decision variables
t1 = {k: cp.Variable() for k in range(K)}  # Auxiliary variable for SOC in Constraint 1
t2 = {k: [cp.Variable() for _ in range(p[k])] for k in range(K)}  # Auxiliary SOC variables for Constraint 2

# Constraints
constraints = []

# Iterate over views and components to build constraints
for k in range(K):
    # Constraint 1: || sum_j K_{k,j} alpha_{k,j} ||_2^2 + epsilon_k * quad_form(alpha) <= 1
    summation_term = sum(K_matrices[(k, j)] @ alpha[(k, j)] for j in range(p[k]))
    constraints.append(cp.SOC(t1[k], summation_term))  # SOC: ||summation_term||_2 <= t1
    constraints.append(t1[k]**2 + epsilon_k * sum(cp.quad_form(alpha[(k, j)], K_matrices[(k, j)]) for j in range(p[k])) <= 1)

    # Constraint 2: Sum over SOCs for each component
    for j in range(p[k]):
        constraints.append(cp.SOC(t2[k][j], (1 / np.sqrt(n)) * K_matrices[(k, j)] @ alpha[(k, j)]))
    constraints.append(cp.sum(t2[k]) <= s_k_range[k][-1])  # SOC summation constraint

# Objective: Example objective (maximize sum of ||summation_term||)
objective = cp.Maximize(sum(cp.norm(summation_term, 2) for k in range(K)))

# Problem definition
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Output results
print("Optimal value:", problem.value)
for k in range(K):
    for j in range(p[k]):
        print(f"alpha[{k},{j}]:", alpha[(k, j)].value)
