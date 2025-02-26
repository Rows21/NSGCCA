import numpy as np

def _linear(N):
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
    return data

def create_synthData(v=2, N=400, outDir='./', device='cpu', mode=1, F=20):
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
    # Create a random set
    #Ej = np.random.normal(loc=0, scale=np.sqrt(0.2), size=(N, F))
    V1 = E1
    V2 = E2
    V3 = E3
    
    if mode == 1:
    # Set the dimension of the multivariate normal distribution
        

        # Verify the covariance matrix and correlation
        #print("Covariance matrix:")
        #print(np.cov(data, rowvar=False))
        #print("Correlation matrix:")
        #print(np.corrcoef(data, rowvar=False))
        data = _linear(N)
        for i in range(v):
            
            V1[:,i] += data[:,0]
            V2[:,i] += data[:,1]
            V3[:,i] += data[:,2]
        # Shuffle the vectors to ensure randomness
        #np.random.shuffle(w1)
        #np.random.shuffle(w2)
        #np.random.shuffle(w3)

        # Print the first 10 elements of w1, w2, and w3 for verification
        #print("w1:", w1[:10])
        #print("w2:", w2[:10])
        #print("w3:", w3[:10])

    elif mode == 2:
        
        # Generate 10 random seeds
        random_seeds = np.random.randint(0, 2**16 - 1, size=v)
        rng = np.random.RandomState(random_seeds[0])
        # Generate random samples from a normal distribution with mean 0 and standard deviation 1
        samples = rng.uniform(0, 2 * np.pi, N)
        # Scale and shift the samples to fit the desired range [-10π, 10π]
        v1 = samples
        v2 = 0.5 * samples ** 2
        v3 = samples * np.cos(samples)
        for i in range(v):
            
            V1[:,i] = v1
            V2[:,i] = v2
            V3[:,i] = v3

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
    views = create_synthData(5,100, mode=1, F=100)
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