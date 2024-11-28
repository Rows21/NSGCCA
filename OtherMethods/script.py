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

# Initialize variables for alpha
alpha = {
    (k, j): np.random.rand(n, 1) for k in range(K) for j in range(p[k])
}  # Random initialization

# Function to solve for all alpha_{kj} (simultaneously update for view k)
def solve_alpha_block(k, alpha_fixed, s_k):
    """Solve for all alpha_{kj} (j=1 to p_k) together for view k."""
    # Define variables
    alpha_vars = [cp.Variable((n, 1)) for j in range(p[k])]

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
    constraints = [
        # Quadratic constraint for all components in view k
        (1 / n) * cp.norm(alpha_sum_k, 2)**2
        + epsilon_k * sum(cp.quad_form(alpha_vars[j], K_matrices[(k, j)]) for j in range(p[k]))
        <= 1,
        
        # Sum-of-norms constraint with K^2 and s_k
        sum(cp.sqrt((1 / n) * cp.quad_form(alpha_vars[j], np.dot(K_matrices[(k, j)].T, K_matrices[(k, j)])))
            for j in range(p[k]))
        <= s_k,
        cp.SOC(s_k, sum(cp.sqrt((1 / n) * cp.quad_form(alpha_vars[j], K_matrices[(k, j)])) for j in range(p[k])))
    ]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    # Return the updated values for alpha_{kj} and the objective value
    return [alpha_var.value for alpha_var in alpha_vars], problem.value

# Tuning s_k for each view
best_s_k = {}
best_alpha = {}
best_objective = -np.inf

for k in range(K):
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