# This is an implementation of PCA and Dual PCA
## Part 1. Implement PCA and Dual PCA algorithm from scratch
def my_pca(X):
    data = X
    data_mean = np.mean(X) # get the mean matrix
    A = data - data_mean  # data centralisation
    S = np.cov(A)  # get the covariance matrix
    eig_val,eig_vec= np.linalg.eig(S)  # get eigenvalue and eigenvecotr
    eig_val=np.sort(eig_val)[::-1] #sort the eigvalue descend
    return eig_val,eig_vec
def my_dual_pca(X):
    data = X
    data_mean = np.mean(X) # get the mean matrix
    A = data - data_mean  # data centralisation
    U, Sigma, Vh = np.linalg.svd(A, 
    full_matrices=False, 
    compute_uv=True)
    eig_vec=U#get eigenvector
    eig_val=np.square(Sigma) / (len(X.T) - 1)   # get eigenvalue
    return eig_val,eig_vec
## Part 2. Visualisation the projection results
## Part 3. Use PCA and Dual PCA for image compresssion
This is part also shows how to display the k eigenface images(k is the optimal number for eigenvectors)
## Part 4. Application: face recognition
Estalish SVM face recognition system with PCA
