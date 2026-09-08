import numpy as np
import cvxpy as cp
from sklearn.metrics.pairwise import pairwise_distances
from itertools import product, combinations_with_replacement
import time
import sys

# Function to compute centered Gram matrix with Gaussian kernel
def centered_gram_matrix(X, n):
    """Compute centered Gaussian kernel Gram matrix."""
    # Pairwise squared Euclidean distances
    pairwise_dist = pairwise_distances(X, metric="euclidean")**2
    upper_off_diag = pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)]
    sigma2 = np.median(upper_off_diag)
    K = np.exp(-pairwise_dist / (2 * sigma2))

    row_means = np.mean(K, axis=1, keepdims=True) 
    col_means = np.mean(K, axis=0, keepdims=True) 
    grand_mean = K.mean()
    K_centered = K - row_means - col_means+ grand_mean

    K_centered = (K_centered + K_centered.T) / 2  # Symmetrize

    # Parameters
    return K_centered #, K_half#P_psd #K_centered

def make_psd_for_cvxpy(K, tol=1e-12):
    K = np.asarray(K, dtype=np.float64)
    K = np.squeeze(K)

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be a square matrix, got shape {K.shape}")

    # 强制对称
    K = (K + K.T) / 2

    # 特征值分解
    eigvals, eigvecs = np.linalg.eigh(K)

    # 只裁掉数值误差导致的负特征值
    eigvals[eigvals < tol] = 0.0

    # 重构 PSD 矩阵
    K_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    K_psd = (K_psd + K_psd.T) / 2

    return K_psd

# Function to solve for all alpha_{kj} (simultaneously update for view k)
def solve_alpha_block(n, p, k, K, alpha_fixed, K_matrices, s_k, epsilon_k, stage = 1, alpha_m = None):
    """Solve for all alpha_{kj} (j=1 to p_k) together for view k."""
    # Define variables
    alpha_vars = [cp.Variable((n, 1)) for j in range(p[k])]
    if stage > 1:
        alpha_vars_0 = [cp.Constant(alpha_m[k,j]) for j in range(p[k])]

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
        (1 / n) * cp.sum(cp.multiply(cross_view_terms, alpha_sum_k))
    )
    # Define constraints
    # Constraints
    constraints = []
    p_k = p[k]
    K_block = np.hstack([
        K_matrices[(k, j)]
        for j in range(p_k)
    ])

    # 拼 alpha block: [alpha_0; alpha_1; ...; alpha_{p_k-1}]
    alpha_block = cp.vstack([
        alpha_vars[j]
        for j in range(p_k)
    ])

    # shape: (n, 1)
    alpha_sum_k = K_block @ alpha_block

    variance_term = (1 / n) * cp.sum_squares(alpha_sum_k)

    rkhs_term = epsilon_k * cp.sum(cp.hstack([
        cp.quad_form(
            alpha_vars[j],
            cp.psd_wrap(K_matrices[(k, j)])
        )
        for j in range(p_k)
    ]))

    # sparsity 仍然需要 group-wise norm，所以保留 hstack
    F_k = cp.hstack([
        K_matrices[(k, j)] @ alpha_vars[j]
        for j in range(p_k)
    ])

    sparsity_term = (1 / np.sqrt(n)) * cp.sum(
        cp.norm(F_k, 2, axis=0)
    )

    #alpha_sum_k = sum(
    #    (K_matrices[(k, j)] @ alpha_vars[j] for j in range(p[k])),
    #    np.zeros((n, 1))
    #)

    #variance_term = (1 / n) * cp.sum_squares(alpha_sum_k)

    #rkhs_term = epsilon_k * sum(
    #    cp.quad_form(alpha_vars[j], cp.psd_wrap(K_matrices[(k, j)]))
    #    for j in range(p[k])
    #)

    #sparsity_term = sum(
    #    (1 / np.sqrt(n)) * cp.norm(K_matrices[(k, j)] @ alpha_vars[j], 2)
    #    for j in range(p[k])
    #)

    constraints = [
        variance_term + rkhs_term <= 1,
        sparsity_term <= s_k
    ]
    #constraints.append((1 / n) * cp.norm(alpha_sum_k, "fro")**2 + epsilon_k * sum(cp.norm(K_matrices[(k,j)] @ alpha_vars[j],"fro")**2 for j in range(p[k])) <= 1)
    #constraints.append(cp.sqrt(1/n) * sum(cp.norm(K_matrices[(k, j)] @ alpha_vars[j], "fro") for j in range(p[k])) <= s_k)  # SOC: ||summation_term||_2 <= t1
    if stage > 1:
        #constraints.append((1 / n) * sum(K_matrices[(k,j)] @ alpha_vars[j] for j in range(p[k])).T @  sum(K_matrices[(k,j)] @ alpha_vars_0[j] for j in range(p[k])) == 0)  # Non-negativity constraints
        alpha_block_current = cp.vstack([
                alpha_vars[j]
                for j in range(p_k)
            ])

        alpha_block_previous = cp.vstack([
                alpha_vars_0[j]
                for j in range(p_k)
            ])

        alpha_sum_current = K_block @ alpha_block_current
        alpha_sum_previous = K_block @ alpha_block_previous

        orth_constraint = (
            (1 / n) * cp.sum(cp.multiply(alpha_sum_current, alpha_sum_previous)) == 0
        )
        constraints.append(orth_constraint)
        #constraints.append((1 / n) * sum(K_matrices_half[(k,j)] @ alpha_vars[j] for j in range(p[k])).T @  sum(K_matrices_half[(k,j)] @ alpha_vars_0[j] for j in range(p[k])) == 0)  # Non-negativity constraints

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, ignore_dpp=True, use_indirect=True, max_iters=5)

    # Return the updated values for alpha_{kj} and the objective value
    return [alpha_var.value for alpha_var in alpha_vars], problem.value

def sakgcca(data, epsilon_k=0.02, max_iter=20, tol=5e-3, r=0, best_alpha=None, best_s_k=None, stage=1,res=None):
    
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
        (k, j): centered_gram_matrix(X_data[(k, j)], n)for k in range(K) for j in range(p[k])
    }
    K_matrices = {
        key: (M + M.T) / 2 + 1e-10 * np.eye(M.shape[0])
        for key, M in K_matrices.items()
    }

    #K_matrices_half = {
    #    (k, j): centered_gram_matrix(X_data[(k, j)], n)[1] for k in range(K) for j in range(p[k])
    #}

    s_k_range = {k: np.linspace(1, np.sqrt(p[k]), 10) for k in range(K)}  # Grid of s_k values
    combinations = list(product(*(s_k_range[k] for k in range(K))))
    # Initialize variables for alpha
    alpha = {
        (k, j): np.random.rand(n, 1) for k in range(K) for j in range(p[k])
    } # Random initialization can you restrict l2 norm to 1

    # Verify the norm for one example
    for (k, j), vec in alpha.items():
        alpha[(k, j)] = vec / np.linalg.norm(vec)

    # Tuning s_k for each view
    if best_s_k is None:
    #a = 0
    #if a == 1:
        best_s_k = {}
        best_alpha = alpha
        best_objective = -np.inf
        for s_k in combinations:
            for k in range(K):
            #print("View:", k)
            
                alpha_tuned = alpha.copy()
                
                prev_alpha = alpha_tuned.copy()
                start_time = time.time()
                updated_alphas, objective_value = solve_alpha_block(n, p, k, K, alpha_tuned, K_matrices, s_k[k], epsilon_k)
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

    print("Best s_k values:", best_s_k)
    #print("Best objective value:", best_objective)

    # selection 
    #select_alpha = best_alpha.copy()
    #best_alpha = {}
    #s_k = s_k_range[1][-1]
    start_time = time.time()
    #for s_k in best_s_k.values():
    
    best_alpha = alpha
    for _ in range(max_iter):
        for k in range(K):
            #prev_alpha = best_alpha.copy()
                
            prev_alpha = alpha.copy()
            #for j in range(p[k]):
            #    best_alpha[(k, j)] = prev_alpha[j] / np.linalg.norm(prev_alpha[j])
                
            updated_alphas, objective_value = solve_alpha_block(n, p, k, K, best_alpha, K_matrices, best_s_k[k], epsilon_k)
                
            for j in range(p[k]):
                best_alpha[(k, j)] = updated_alphas[j] / np.linalg.norm(updated_alphas[j])
            # Check convergence
            diff = max(
                np.max(np.abs(best_alpha[(k, j)] - prev_alpha[(k, j)]))
                for j in range(p[k])
            )
            if diff < tol:
                break
            
    u1 = [np.zeros((p[k])) for k in range(K)]
    for k in range(K):
        for j in range(p[k]):
            l2 = (1 / np.sqrt(n)) * np.linalg.norm(K_matrices[(k, j)] @ best_alpha[(k, j)], 'fro')
            u1[k][j] = l2
    end_time = time.time()   
    #np.savetxt(res+'/snr25sakgcca_u1'+str(sys.argv[1])+'.csv', u1[0], delimiter=',')
    #np.savetxt(res+'/snr25sakgcca_u2'+str(sys.argv[1])+'.csv', u1[1], delimiter=',')
    #np.savetxt(res+'/snr25sakgcca_u3'+str(sys.argv[1])+'.csv', u1[2], delimiter=',')
    
    for k in range(K):
        for _ in range(max_iter):
            #prev_alpha = best_alpha.copy()
                
            prev_alpha = alpha.copy()
            #for j in range(p[k]):
            #    best_alpha[(k, j)] = prev_alpha[j] / np.linalg.norm(prev_alpha[j])
            updated_alphas, objective_value = solve_alpha_block(n, p, k, K, best_alpha, K_matrices, best_s_k[k], epsilon_k, stage = stage, alpha_m = best_alpha)
            for j in range(p[k]):
                best_alpha[(k, j)] = updated_alphas[j] / np.linalg.norm(updated_alphas[j])
            # Check convergence
            diff = max(
                np.max(np.abs(best_alpha[(k, j)] - prev_alpha[(k, j)]))
                for j in range(p[k])
            )
            if diff < tol:
                break        
            
    u2 = [np.zeros((p[k])) for k in range(K)]
    for k in range(K):
        for j in range(p[k]):
            l2 = (1 / np.sqrt(n)) * np.linalg.norm(K_matrices[(k, j)] @ best_alpha[(k, j)], 'fro')
            u2[k][j] = l2
    
    delta = end_time - start_time         
    return u1, u2, delta, best_alpha, best_s_k

if __name__ == "__main__":
    combinations = [
        [100, 30, 5],
        [100, 50, 5],
        [100, 100, 5],
        [100, 100, 10],
        [100, 100, 20],
        [100, 200, 5],
        [200, 100, 5],
        [400, 100, 5]
    ]
    stage = 1
    for params in combinations:
        for mode in [1,2]:
            print(params)
            t = []
            u1 = []
            u2 = []
            u3 = []
            u1_2 = []
            u2_2 = []
            u3_2 = []
            
            N = params[0]
            P = params[1]
            S = params[2]
            root = '/Users/rongwu/Desktop/res/SNGCCA/SNGCCA/'
            if mode == 1:
                    folder = 'Linear/'
            else:
                    folder = 'Nonlinear/'
            data_path = root + 'newData/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/'
            if stage > 1:
                path_u = '/Users/rongwu/Desktop/res/SNGCCA/SNGCCA/Simulation/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/'
                u1_m = np.genfromtxt(path_u + 'sakgcca_u1.csv', delimiter=',')
                u2_m = np.genfromtxt(path_u + 'sakgcca_u2.csv', delimiter=',')
                u3_m = np.genfromtxt(path_u + 'sakgcca_u3.csv', delimiter=',')
                
            for r in range(100):
                print(f'Iteration : {r}')
                #views = create_synthData_new(v=5,N=N,mode=1,F=30)
                view1 = np.genfromtxt(data_path + 'snr1data' + str(1) + '_' + str(r) + '.csv', delimiter=',')
                view2 = np.genfromtxt(data_path + 'snr1data' + str(2) + '_' + str(r) + '.csv', delimiter=',')
                view3 = np.genfromtxt(data_path + 'snr1data' + str(3) + '_' + str(r) + '.csv', delimiter=',')
                views = [view1, view2, view3]
                #print(f'input views shape :')
                #for i, view in enumerate(views):
                #    print(f'view_{i} :  {view.shape}')
                if stage == 1:
                    if r == 0:
                        u, u_2, delta, best_alpha, best_s_k = sakgcca(views)
                    else:
                        u, u_2, delta, best_alpha, best_s_k = sakgcca(views, best_alpha=best_alpha, best_s_k=best_s_k)
                else:
                    u_m = [u1_m[r:r+1,].T, u2_m[r:r+1,].T, u3_m[r:r+1,].T]
                    if r == 0:
                        u, u_2, delta, best_alpha, best_s_k = sakgcca(views, stage=2, res='E:/GitHub/res/SNGCCA/SNGCCA/Simulation/pair2/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/')
                    else:
                        u, u_2, delta, best_alpha, best_s_k = sakgcca(views, best_alpha=best_alpha, best_s_k=best_s_k, stage=2)
                    
                t.append(delta)
                u1.append(u[0])
                u2.append(u[1])
                u3.append(u[2])
                u1_2.append(u_2[0])
                u2_2.append(u_2[1])
                u3_2.append(u_2[2])

                    
            merged_array = np.empty((100,P))
            path = 'E:/GitHub/res/SNGCCA/SNGCCA/newnewSimulation/pair1/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/'
                
            for i, arr in enumerate(u1):
                merged_array[i] = u1[i].flatten()
            np.savetxt(path + 'snr1sakgcca_u1.csv', merged_array, delimiter=',')
            for i, arr in enumerate(u2):
                merged_array[i] = u2[i].flatten()
            np.savetxt(path + 'snr1sakgcca_u2.csv', merged_array, delimiter=',')
            for i, arr in enumerate(u3):
                merged_array[i] = u3[i].flatten()
            np.savetxt(path + 'snr1sakgcca_u3.csv', merged_array, delimiter=',')
            np.savetxt(path + 'snr1sakgcca_t.csv', t, delimiter=',')

            merged_array = np.empty((100,P))
            path = 'E:/GitHub/res/SNGCCA/SNGCCA/newnewSimulation/pair2/' + folder + '/' + str(N) + '_' + str(P) + '_' + str(S) + '/'
                
            for i, arr in enumerate(u1):
                merged_array[i] = u1_2[i].flatten()
            np.savetxt(path + 'snr1sakgcca_u1.csv', merged_array, delimiter=',')
            for i, arr in enumerate(u2):
                merged_array[i] = u2_2[i].flatten()
            np.savetxt(path + 'snr1sakgcca_u2.csv', merged_array, delimiter=',')
            for i, arr in enumerate(u3):
                merged_array[i] = u3_2[i].flatten()
            np.savetxt(path + 'snr1sakgcca_u3.csv', merged_array, delimiter=',')
