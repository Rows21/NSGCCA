import numpy as np
import cvxpy as cp
from sklearn.metrics.pairwise import pairwise_distances
import time

# Function to compute centered Gram matrix with Gaussian kernel
def centered_gram_matrix(X, n):
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

# Function to solve for all alpha_{kj} (simultaneously update for view k)
def solve_alpha_block(n, p, k, K, alpha_fixed, K_matrices, K_matrices_half, s_k, epsilon_k):
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
    problem.solve(solver=cp.SCS, ignore_dpp=True, use_indirect=True, max_iters=50)

    # Return the updated values for alpha_{kj} and the objective value
    return [alpha_var.value for alpha_var in alpha_vars], problem.value

def sakgcca(data, epsilon_k=0.02, max_iter=5, tol=1e-5, r=0, best_alpha=None, best_s_k=None):
    
    K = len(data)
    print(K)
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)
    p = [data[i].shape[1] for i in range(K)]
    n = data[0].shape[0]
    
    X_data = {
        (k, j): data[k][:,j].reshape(-1,1) for k in range(K) for j in range(p[k])
        #(k, j): np.random.rand(n, 1) for k in range(K) for j in range(p[k])
    }

    # Compute centered Gram matrices
    K_matrices = {
        (k, j): centered_gram_matrix(X_data[(k, j)], n)[0] for k in range(K) for j in range(p[k])
    }

    K_matrices_half = {
        (k, j): centered_gram_matrix(X_data[(k, j)], n)[1] for k in range(K) for j in range(p[k])
    }

    s_k_range = {k: np.linspace(1, np.sqrt(p[k]), 10) for k in range(K)}  # Grid of s_k values

    # Initialize variables for alpha
    alpha = {
        (k, j): np.random.rand(n, 1) for k in range(K) for j in range(p[k])
    } # Random initialization can you restrict l2 norm to 1

    # Verify the norm for one example
    for (k, j), vec in alpha.items():
        alpha[(k, j)] = vec / np.linalg.norm(vec)

    # Tuning s_k for each view
    #if r == 0:
    a = 0
    if a == 1:
        best_s_k = {}
        best_alpha = alpha
        best_objective = -np.inf
        for s_k in s_k_range[1]:
            for k in range(K):
            #print("View:", k)
            
                alpha_tuned = alpha.copy()
                for _ in range(max_iter):
                    prev_alpha = alpha_tuned.copy()
                    start_time = time.time()
                    updated_alphas, objective_value = solve_alpha_block(n, p, k, K, alpha_tuned, K_matrices, K_matrices_half, s_k, epsilon_k)
                    if None in updated_alphas[1]:
                        break
                    for j in range(p[k]):
                        alpha_tuned[(k, j)] = updated_alphas[j] / np.linalg.norm(updated_alphas[j])

                    # Check convergence
                    diff = max(
                        np.max(np.abs(alpha_tuned[(k, j)] - prev_alpha[(k, j)]))
                        for j in range(p[k])
                    )
                    if diff < tol:
                        break

                if objective_value > best_objective:
                    best_objective = objective_value
                    best_s_k[k] = s_k
                    for j in range(p[k]):
                        best_alpha[(k, j)] = alpha_tuned[(k, j)]

    #print("Best s_k values:", best_s_k)
    #print("Best objective value:", best_objective)

    # selection 
    #select_alpha = best_alpha.copy()
    #best_alpha = {}
    s_k = s_k_range[1][-1]
    start_time = time.time()
    #for s_k in best_s_k.values():
    
    best_alpha = alpha
    
    for k in range(K):
            for _ in range(max_iter):
                #prev_alpha = best_alpha.copy()
                
                prev_alpha = alpha.copy()
                #for j in range(p[k]):
                #    best_alpha[(k, j)] = prev_alpha[j] / np.linalg.norm(prev_alpha[j])
                    
                updated_alphas, objective_value = solve_alpha_block(n, p, k, K, best_alpha, K_matrices, K_matrices_half, s_k, epsilon_k)
                
                for j in range(p[k]):
                    best_alpha[(k, j)] = updated_alphas[j] / np.linalg.norm(updated_alphas[j])

                # Check convergence
                diff = max(
                    np.max(np.abs(best_alpha[(k, j)] - prev_alpha[(k, j)]))
                    for j in range(p[k])
                )
                if diff < tol:
                    break
    
    u = [np.zeros((p[k])) for k in range(K)]
    for k in range(K):
        for j in range(p[k]):
            l2 = (1 / np.sqrt(n)) * np.linalg.norm(K_matrices[(k, j)] @ best_alpha[(k, j)], 'fro')
            u[k][j] = l2
    end_time = time.time()   
    delta = end_time - start_time         
    return u, delta, best_alpha, best_s_k

if __name__ == "__main__":
    combinations = [
        #[100, 30, 5],
        [100, 50, 5],
        #[100, 100, 5],
        #[100, 100, 10],
        #[100, 100, 20],
        [100, 200, 5],
        [200, 100, 5],
        [400, 100, 5]
    ]
    
    for params in combinations:
        for mode in [1,2]:
            print(params)
            t = []
            u1 = []
            u2 = []
            u3 = []
            
            N = params[0]
            P = params[1]
            S = params[2]
            root = 'E:/Github/SNGCCA/SNGCCA/'
            if mode == 1:
                    folder = 'Linear/'
            else:
                    folder = 'Nonlinear/'
            data_path = root + 'Data/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/'

            for r in range(50):
                print(f'Iteration : {r}')
                #views = create_synthData_new(v=5,N=N,mode=1,F=30)
                view1 = np.genfromtxt(data_path + 'data' + str(1) + '_' + str(r) + '.csv', delimiter=',')
                view2 = np.genfromtxt(data_path + 'data' + str(2) + '_' + str(r) + '.csv', delimiter=',')
                view3 = np.genfromtxt(data_path + 'data' + str(3) + '_' + str(r) + '.csv', delimiter=',')
                views = [view1, view2, view3]
                #print(f'input views shape :')
                #for i, view in enumerate(views):
                #    print(f'view_{i} :  {view.shape}')
                if r == 0:
                    u, delta, best_alpha, best_s_k = sakgcca(views)
                else:
                    u, delta, best_alpha, best_s_k = sakgcca(views, best_alpha=best_alpha, best_s_k=best_s_k)
                t.append(delta)
                u1.append(u[0])
                u2.append(u[1])
                u3.append(u[2])
                    
            merged_array = merged_array = np.empty((100,P))
            path = 'E:/GitHub/SNGCCA/SNGCCA/Simulation/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/'
                
            for i, arr in enumerate(u1):
                merged_array[i] = u1[i].flatten()
            np.savetxt(path + 'sakcca_u1.csv', merged_array, delimiter=',')
            for i, arr in enumerate(u2):
                merged_array[i] = u2[i].flatten()
            np.savetxt(path + 'sakcca_u2.csv', merged_array, delimiter=',')
            for i, arr in enumerate(u3):
                merged_array[i] = u3[i].flatten()
            np.savetxt(path + 'sakcca_u3.csv', merged_array, delimiter=',')
            np.savetxt(path + 'sakcca_t.csv', t, delimiter=',')