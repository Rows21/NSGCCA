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
    upper_off_diag = pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)]
    sigma2 = np.median(upper_off_diag)
    K = np.exp(-pairwise_dist / (2 * sigma2))
    ones = np.ones((n, n)) / n
    #K_centered = K - ones @ K - K @ ones + ones @ K @ ones
    K_centered  =  (np.eye(n) -ones) @ K @ (np.eye(n)-ones)
    #P = K_matrices[(k, j)].T @ K_matrices[(k, j)] + regularization * np.eye(K_matrices[(k, j)].shape[0])
    P = (K_centered + K_centered.T) / 2  # Symmetrize
    U, S, Vt = np.linalg.svd(P)
    S_psd = np.maximum(S, 0)
    K_centered = U @ np.diag(S_psd) @ U.T

    S_sqrt = np.sqrt(S_psd)
    K_half = U @ np.diag(S_sqrt) @ U.T
    # Parameters
    return K_centered, K_half#P_psd #K_centered

# Generate example data (replace with actual data)
X_data = {
    (k, j): np.random.rand(n, 1) for k in range(K) for j in range(p[k])
}

# Compute centered Gram matrices
K_matrices = {
    (k, j): centered_gram_matrix(X_data[(k, j)])[0] for k in range(K) for j in range(p[k])
}

K_matrices_half = {
    (k, j): centered_gram_matrix(X_data[(k, j)])[1] for k in range(K) for j in range(p[k])
}
      
epsilon_k = 0.02  # Regularization parameter for each view
s_k_range = {k: np.linspace(1, np.sqrt(p[k]), 10) for k in range(K)}  # Grid of s_k values

# Initialize variables for alpha
alpha = {
    (k, j): np.random.rand(n, 1) for k in range(K) for j in range(p[k])
}  # Random initialization

# Function to solve for all alpha_{kj} (simultaneously update for view k)
def solve_alpha_block(k, alpha_fixed, s_k):
    """Solve for all alpha_{kj} (j=1 to p_k) together for view k."""
    # Define variables
    alpha_vars = [cp.Variable((n, 1)) for j in range(p[k])]
    #alpha_vars = {
    #    (k, j): cp.Variable(n) for k in range(K) for j in range(p[k])
    #}
    # Compute contributions from other views (k' != k)
    cross_view_terms = sum(
        sum(
            K_matrices[(k_prime, j)] @ alpha_fixed[(k_prime, j)]
            for j in range(p[k_prime])
        )
        for k_prime in range(K) if k_prime != k
    )

    # Define the objective function
    alpha_sum_k = sum(K_matrices[(k, j)] @ alpha_vars[j] for j in range(p[k]))
    objective = cp.Maximize(
        (1 / n) * cp.matmul(cross_view_terms.T, alpha_sum_k)
    )
    # Define constraints
    # Constraints
    constraints = []
    constraints.append((1 / n) * cp.norm(alpha_sum_k, "fro")**2 + epsilon_k * sum(cp.norm(K_matrices_half[(k,j)] @ alpha_vars[j],"fro")**2 for j in range(p[k])) <= 1)
    constraints.append(cp.sqrt(1/n) * sum(cp.norm(K_matrices[(k, j)] @ alpha_vars[j], "fro") for j in range(p[k])) <= s_k)  # SOC: ||summation_term||_2 <= t1

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, ignore_dpp = True)

    # Return the updated values for alpha_{kj} and the objective value
    return [alpha_var.value for alpha_var in alpha_vars], problem.value

# Tuning s_k for each view
best_s_k = {}
best_alpha = {}
best_objective = -np.inf

for k in range(K):
    print("View:", k)
    
    for s_k in s_k_range[k]:
        # Reset alpha to initial state before tuning
        alpha_tuned = alpha.copy()
        max_iter = 50
        tolerance = 1e-5

        # Block coordinate descent for the current s_k
        for iteration in range(max_iter):
            prev_alpha = alpha_tuned.copy()

            # Update all alpha_{kj} for view k
            updated_alphas, objective_value = solve_alpha_block(k, alpha_tuned, s_k)
            for j in range(p[k]):
                alpha_tuned[(k, j)] = updated_alphas[j]

            # Check convergence
            diff = max(
                np.max(np.abs(alpha_tuned[(k, j)] - prev_alpha[(k, j)]))
                for j in range(p[k])
            )
            if diff < tolerance:
                break

            # Track the best s_k and alpha based on the objective value
        if objective_value > best_objective:
            best_objective = objective_value
            best_s_k[k] = s_k
            best_alpha[k] = alpha_tuned

print("Best s_k values:", best_s_k)
print("Best objective value:", best_objective)

# selection 
for k in range(K):
    for j in range(p[k]):
        l2 = (1/np.sqrt(n)) * np.linalg.norm(K_matrices[(k,j)] @ best_alpha[0][(k, j)], 'fro')
        if l2 > 1e-6:
            print(f"alpha[{k},{j}]:")

            

        
            
