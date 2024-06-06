import numpy as np
import torch

def create_synthData_new(v=2, N=400, outDir='./', device='cpu', mode=1, F=20):
    '''
    creating Main paper Synth data,
    N : number of data
    F$ : number of features in view $ 
    '''
    #np.random.seed(1)
    #torch.manual_seed(0)

    random_seeds = np.random.randint(0, 2**16 - 1, size=3)
    rng1 = np.random.RandomState(random_seeds[0])
    E1 = rng1.normal(loc=0, scale=np.sqrt(0.2), size=(N, F))
    rng2 = np.random.RandomState(random_seeds[1])
    E2 = rng2.normal(loc=0, scale=np.sqrt(0.2), size=(N, F))
    rng3 = np.random.RandomState(random_seeds[2])
    E3 = rng3.normal(loc=0, scale=np.sqrt(0.2), size=(N, F))
    
    if mode == 1:
    # Set the dimension of the multivariate normal distribution
        dimension = 3  # Change this to the desired dimension

        # Generate random data from a multivariate normal distribution with mean=0 and covariance=identity matrix
        data = np.random.multivariate_normal(mean=np.zeros(dimension), cov=np.eye(dimension), size=N)

        # Desired correlation matrix
        desired_corr1 = 0.7
        desired_corr2 = 0.0

        # Generate the covariance matrix from the desired correlation
        cov_matrix = np.array([[1, desired_corr1,desired_corr1], [desired_corr1,1, desired_corr2] , [desired_corr1,desired_corr2, 1]])

        # Compute the Cholesky decomposition of the covariance matrix
        L = np.linalg.cholesky(cov_matrix)

        # Transform the data to have the desired covariance matrix
        data = np.dot(data, L.T)

        # Verify the covariance matrix and correlation
        print("Covariance matrix:")
        print(np.cov(data, rowvar=False))
        print("Correlation matrix:")
        print(np.corrcoef(data, rowvar=False))
        
        # Define the dimensions of the vectors
        w1_dim = F
        w2_dim = F
        w3_dim = F

        # Define the number of nonzero elements
        nonzero_elements = v

        # Generate nonzero elements from a uniform distribution
        nonzero_values = np.random.uniform(low=-1, high=-0.9, size=nonzero_elements)
        nonzero_values = np.concatenate([nonzero_values, np.random.uniform(low=0.9, high=1, size=nonzero_elements)])
        # nonzero_values = np.ones(1000)

        # Shuffle the nonzero values
        np.random.shuffle(nonzero_values)

        # Initialize w1, w2, and w3 as zero vectors
        w1 = np.zeros(w1_dim)
        w2 = np.zeros(w2_dim)
        w3 = np.zeros(w3_dim)

        # Assign nonzero values to the first 75 elements of w1, w2, and w3
        w1[:nonzero_elements] = nonzero_values[:nonzero_elements]
        np.random.shuffle(nonzero_values)
        w2[:nonzero_elements] = nonzero_values[:nonzero_elements]
        np.random.shuffle(nonzero_values)
        w3[:nonzero_elements] = nonzero_values[:nonzero_elements]

        # Shuffle the vectors to ensure randomness
        #np.random.shuffle(w1)
        #np.random.shuffle(w2)
        #np.random.shuffle(w3)

        # Print the first 10 elements of w1, w2, and w3 for verification
        #print("w1:", w1[:10])
        #print("w2:", w2[:10])
        #print("w3:", w3[:10])
    
        V1 = data[:,0].reshape(N,1) @ w1.reshape(1,F) + E1
        V2 = data[:,1].reshape(N,1) @ w2.reshape(1,F) + E2
        V3 = data[:,2].reshape(N,1) @ w3.reshape(1,F) + E3
    elif mode == 2:
        # Create a random set
        #Ej = np.random.normal(loc=0, scale=np.sqrt(0.2), size=(N, F))
        V1 = E1
        V2 = E2
        V3 = E3
        # Generate 10 random seeds
        random_seeds = np.random.randint(0, 2**16 - 1, size=v)

        for i, seed in enumerate(random_seeds):
            rng = np.random.RandomState(seed)
            # Generate random samples from a normal distribution with mean 0 and standard deviation 1
            samples = rng.normal(0, 1, N)
            # Scale and shift the samples to fit the desired range [-10π, 10π]
            v1 = samples * (2 * np.pi)
            v2 = np.cos(v1)
            v3 = v1 * np.cos(v1)
            scaled_v1 = ((v1 - np.mean(v1)) / np.std(v1)) * 2
            scaled_v2 = ((v2 - np.mean(v2)) / np.std(v2)) * 2
            scaled_v3 = ((v3 - np.mean(v2)) / np.std(v3)) * 2

            V1[:,i] = scaled_v1 + V1[:,i]
            V2[:,i] = scaled_v2 + V2[:,i]
            V3[:,i] = scaled_v3 + V3[:,i]

    views  = []
    views.append(V1)
    views.append(V2)
    views.append(V3)
    
    #print(np.sum(V1[:,0:v].T @ V2[:,0:v]))
    #print(np.sum(V1.T @ V2))
    
    #correlation_matrix = np.corrcoef(V1.T, V2.T)
    #print(correlation_matrix)
    print("------------------")
    #views = [torch.tensor(view).to(device) for view in views]
    return views

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    views = create_synthData_new(5,100, mode=1, F=100)
    x = views[0].numpy()[:,0]
    #x = np.sum(x, axis=1)
    y = views[1].numpy()[:,0]
    #y = np.sum(y, axis=1)
    z = views[2].numpy()[:,0]
    #z = np.sum(z, axis=1)
    plt.plot(x, y, 'bo', label='Data 1')
    plt.show()
    plt.plot(x, z, 'bo', label='Data 2')
    plt.show()
    plt.plot(y, z, 'bo', label='Data 3')
    plt.show()
    a=1