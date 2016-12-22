import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

def matrix_plot(dense_matrix, fname, cutoff=0, title=''):
    if cutoff > 0:
        ax = plt.matshow(dense_matrix[:cutoff,:])
    else:
        ax = plt.matshow(dense_matrix)
    if title != '':
        plt.title(title)
    plt.axis('off')
    plt.savefig(fname)

def create_sparse_matrix_plot(sparse_matrix, fname='',cutoff=1000, title=''):
    ''' Create a dense matrix plot, squeezing all entries along specified axis.
    input:
        sparse_matrix    - scipy sparse matrix to visualize
        axis             - axis along which all nnz-elements will be squeezed.
    '''
    rows,cols,ratings = sp.find(sparse_matrix) 
    ratings_dense = np.zeros(sparse_matrix.shape)
    i_dense = 0
    i_total = []
    last_index = 0
    for i, rating in enumerate(ratings):
        index = cols[i]
        if index > last_index:
            i_total.append(i_dense)
            i_dense = 0
        ratings_dense[i_dense,index] = rating
        last_index = index
        i_dense += 1
    i_max=np.max(i_total)
    matrix_plot(ratings_dense, fname, min(i_max,cutoff), title)
    print('user min, max, mean:',np.min(i_total), np.max(i_total), np.mean(i_total))
    return i_total, ratings_dense[:i_max,:]
