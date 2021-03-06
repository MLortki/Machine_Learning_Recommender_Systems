""" Linear blending helper functions """

import numpy as np
from helpers import load_data
import scipy.sparse as sp

def read_submission_files(files):
    """ read files that have been saved in submission format,
        with first column Id "cX_rY" and second column Prediction.

        input:  files           -list of file names to read.

        output: sparse_matrices -list of scipy sparse matrices read from files.
    """
    sparse_matrices = []
    for i,file_ in enumerate(files):
        sparse_matrix = load_data(file_)
        sparse_matrices.append(sparse_matrix.copy())
    return sparse_matrices

def read_numpy_files(files, indices):
    """read data from numpy files and apply indices to create sparse matrices. 

        input:  files           -list of file names to read.
                indices         -list of indices corresponding to nnz-elements of interest.

        output: sparse_matrices -list of scipy sparse matrices read from files.
    """
    sparse_matrices = []
    for i,file_ in enumerate(files):
        dense_matrix = np.load(file_)
        print('matrix',i)
        print('dense matrix shape:',dense_matrix.shape)
        sparse_matrix = apply_indices_matrix(dense_matrix, indices)
        sparse_matrices.append(sparse_matrix.copy())
        print('number of non-zero elements in matrix:',sparse_matrix.nnz)
    return sparse_matrices

def apply_indices_matrix(matrix, indices):
    """extract indices from dense or sparse matrix. 

        input:  matrix          -sparse or dense matrix
                indices         -list of indices corresponding to nnz-elements of interest.

        output: sparse_matrix   -sparse matrices corresponding to indices of matrix.
    """
    sparse_matrix = sp.lil_matrix(matrix.shape)
    for (i,j) in indices:
        sparse_matrix[i,j] = matrix[i,j] 
    return sparse_matrix

def apply_indices(sparse_matrices, dataset='train'):
    """extract indices from multiple sparse matrices.

        input:  sparse_matrices -list of sparse matrices
                dataset         -string defining which indices to use.
                                 ('test','train','validation' or 'submisison'

        output: sparse_reduced  -list of sparse matrices corresponding to
                                 indices of elements of sparse_matrices.
    """ 
    # Load indices.
    if dataset == 'train':
        file = '../data/blending_train.csv'
    elif dataset == 'test':
        file = '../data/blending_test.csv'
    elif dataset == 'validation':
        file = '../data/blending_validation.csv'
    elif dataset == 'submission':
        file = '../data/sampleSubmission.csv'
    indices_matrix = load_data(file)
    rows, cols, __ = sp.find(indices_matrix)
    print('number of rows, cols:',len(cols))
    indices = list(zip(rows, cols))
    
    sparse_reduced = []
    for i,matrix in enumerate(sparse_matrices):
        print('treating matrix', i)
        sparse_matrix = apply_indices_matrix(matrix, indices)
        sparse_reduced.append(sparse_matrix.copy())
        print('number of non-zero elements in matrix {}: {}'.format(i, sparse_matrix.nnz))
    return sparse_reduced

def get_all_indices(train_true, submission_true): 
    """ combine indices of dataset and submission dataset

        input:  train_true          -sparse ratings matrix
                submission_true     -spare submissions matrix

        output: indices             -combined list of non-zero indices of both input matrices.
    """
    rows_train, cols_train, __ = sp.find(train_true)
    rows_submission, cols_submission, __ = sp.find(submission_true)
    rows_full = np.hstack((rows_train, rows_submission)) 
    cols_full = np.hstack((cols_train, cols_submission)) 
    print('number of rows, cols to read out:',len(cols_full))
    indices = list(zip(rows_full, cols_full))
    return indices

def linear_blending(test_est, submission_est, test_true):
    """ do linear blending of different methods as described in:
    /http://www.commendo.at/UserFiles/commendo/File/Presentation_GrandPrize.pdf

        input:  test_est            - list of sparse matrices containing the T predictions of M methods on probe dataset. (trained on whole dataset)
                submission_est      - list of sprase matrices containing the N predictions for test (kaggle) dataset.
                test_true           - list of sparse matrices containing   
        output: q_hat               - vector of N optimal predictions for test set
                x                   -weights vector of M weights corresponding to methods.  
    """
    def create_matrix(matrix_list): 
        P_or_Q = np.empty((matrix_list[0].nnz, len(matrix_list)))
        for i,matrix in enumerate(matrix_list):
            __,__,ratings = sp.find(matrix)
            P_or_Q[:,i] = ratings
        return P_or_Q
        
    def get_weights(P,r):
        x = np.linalg.solve(np.dot(P.T, P),np.dot(P.T,r))
        print('x',x)
        return x
    
    def get_predictions(Q, x):
        q_hat = np.dot(Q, x)
        return q_hat 
    
    P = create_matrix(test_est)
    Q = create_matrix(submission_est)
    __, __, r = sp.find(test_true)
    x = get_weights(P, r)
    q_hat = get_predictions(Q, x)
    return q_hat, x
