import time
import numpy as np
from numpy import linalg as la

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    #TODO: Implement PCA by extracting eigenvector#
    ###############################################
    arr = np.array(X)
    cov = np.corrcoef(arr, rowvar=0)
    eig_val, eig_vec = np.linalg.eig(cov)
    P = eig_vec[:,:K].T
    T = eig_val[:K]
    ###############################################
    #              End of your code               #
    ###############################################
    
    return (P, T)
