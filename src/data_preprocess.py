""" Functions to preprocess ratings matrix by bias correction etc. """

from helpers import build_index_groups
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def get_user_means(train, nz_col_rowindices):
    """returns mean for every user.
    
       input:   train               -data matrix (D x N)
                nz_col_rowindices   -indices of nonzero movie ratings of each user
                
       output:  user_means          -mean of ratings for every user
    """
    
    #initialize user_means array
    user_means = np.zeros((train.shape[1],1))
    
    #for every user, estimate mean of their ratings
    for j, rowindices in nz_col_rowindices:
        user_means[j,0] = np.mean(train[rowindices,j])
        
    return user_means

def get_item_means(train, nz_row_colindices):
    """returns mean rating for every movie.
    
       input:   train               -data matrix (D x N)
                nz_col_rowindices   -indices of users that rated each movie 
                
       output:  item_means          -mean of ratings for every movie
    """
    
    #initialize item_means array
    item_means = np.zeros((train.shape[0],1))
    
    #for every movie, estimate average rating
    for i, colindices in nz_row_colindices:
        item_means[i,0] = np.mean(train[i,colindices])
        
    return item_means

def get_global_means(train, nz_train):
    """returns mean of ratings.
    
       input:   train               -data matrix (D x N)
                nz_train            -indices of users that rated each movie 
                
       output:  means               -matrix of means(itendical elements)
                mean                -mean
    """
    
    means = sp.lil_matrix(train.shape)
    
    rows, cols, ratings = sp.find(train)
    mean = np.mean(ratings)
    means[rows,cols] = mean
    
    return means, mean

def get_unbiased_matrix(train, user_means, item_means, means, m):
    """returns unbiased matrix
    
       input:   train               -data matrix (D x N)
                user_means          -average of ratings of each user
                item_means          -average rating of each movie
                means               -mean of ratings(matrix)
                m                   -{'no','global','item','user'}
                                      'no'        -no bias
                                      'global'    -global bias
                                      'item'      -item bias
                                      'user'      -user bias
                                      'combined'  -combination of global, item and user bias
                                      
                
       output:  train_normalized    -dataset without specified bias
                train_means         -bias deducted from data
    """
    
    #find nonzero rows, columns and elements of train
    rows, cols, ratings = sp.find(train)
    
    #initialize unbiased matrix and total bias deducted from the data
    train_normalized = sp.lil_matrix(train.shape)
    train_means = sp.lil_matrix(train.shape)
    
    #no bias
    if m=='no':
        print('no method ok')
        return train, sp.lil_matrix(train.shape)
    
    #global bias
    elif m=='global':
        train_normalized[rows,cols] = train[rows,cols] - means[rows,cols]
        train_means[rows,cols] = means[rows,cols]
        assert train_normalized.mean() < 1e-14
        print('global ok')
        
    #item bias    
    elif m=='item':     
        train_means[rows,cols] = item_means[rows].T
        train_normalized[rows,cols] = train[rows,cols] - item_means[rows].T
        assert np.mean(train_normalized.mean(axis=1)) < 1e-14
        print('item ok')
        
    #user bias    
    elif m=='user':
        train_means[rows,cols] = user_means[cols].T
        train_normalized[rows,cols] = train[rows,cols] - user_means[cols].T
        assert np.mean(train_normalized.mean(axis=0)) < 1e-14
        print('user ok')
        
    #combined bias    
    elif m=='combined':
        train_means[rows,cols] = user_means[cols].T + item_means[rows].T - means[rows,cols]
        train_normalized[rows,cols] = train[rows,cols] - train_means[rows,cols]
        print('combined ok')
        
    return train_normalized, train_means

def get_predictions(ratings, user_means, item_means, mean, m):
    """add back biases to the prediction.
    
       input:   ratings               -data matrix (D x N)
                user_means            -average of ratings of each user 
                item_means            -average rating of each movie
                mean                  -mean of ratings
                m                     -{'global','item','user'}
                                        'global'    -add back global bias 
                                        'item'      -add back movie bias
                                        'user'      -add back user bias
                                        'combined'  -add back combined bias
                
       output:  predictions           -predictions with biases 
    """
    
    #initialization
    predictions = np.empty(ratings.shape)
    item_matrix = np.tile(item_means, (1,predictions.shape[1]))
    user_matrix = np.tile(user_means.T, (predictions.shape[0],1))   
    means_matrix = np.ones(ratings.shape)*mean
    
    #adding back combined bias
    if m=='combined':
        predictions = ratings + item_matrix + user_matrix - means_matrix
        
    #adding back item bias
    elif m=='item':
        predictions = ratings + item_matrix 
        
    #adding back user bias
    elif m=='user':
        predictions = ratings + user_matrix
        
    #adding back global bias
    elif m=='global':
        predictions = ratings + mean
        
    return predictions

def get_statistics(sparse_matrix):
    """print out useful statistics of sparse prediction matrix.
    
       input:   sparse_matrix       -matrix 
    """
    rows, cols, ratings = sp.find(sparse_matrix)
    user_ratings = np.zeros(1000)
    item_ratings = np.zeros(10000)
    for i in range(10000):
        item_ratings[i] = np.count_nonzero(rows==i)
    for i in range(1000):
        user_ratings[i] = np.count_nonzero(cols==i)
    print('item min={},max={},mean={}'.format(np.min(item_ratings),np.max(item_ratings),np.mean(item_ratings)))
    print('user min={},max={},mean={}'.format(np.min(user_ratings),np.max(user_ratings),np.mean(user_ratings)))
    return item_ratings, user_ratings

if __name__=="__main__":
""" Creates visualization of above function when run from command line."""
    scale = np.arange(0,6)
    scale = np.vstack((scale,scale))

    train = sp.lil_matrix((5,5))
    train[0,3] = 4.0
    train[0,1] = 4.0
    train[0,4] = 5.0
    train[1,3] = 1.0
    train[1,2] = 2.0
    train[1,4] = 3.0
    train[2,0] = 3.0
    train[2,3] = 3.0
    train[3,0] = 4.0
    train[3,1] = 3.0
    train[4,3] = 1.0
    train[4,2] = 1.0
    methods=['global','user','item','combined','no']
    fig, (ax00, ax01) = plt.subplots(1,2)
    ax00.set_title('train')
    ax00.matshow(train.todense())
    ax01.set_title('scale')
    ax01.matshow(scale)
    ax01.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    fig.savefig('../results/biases_train_scale')

    nz_train, nz_row_colindices, nz_col_rowindices = build_index_groups(train)
    user_means = get_user_means(train, nz_col_rowindices) 
    item_means = get_item_means(train, nz_row_colindices)
    means = get_global_means(train, nz_train) 
    rows,cols,ratings = sp.find(train)
    for m in methods:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
        train_normalized, train_mean = get_unbiased_matrix(train.copy(),user_means,item_means,means,m)
        print('error:',np.sqrt(compute_error2(train_mean,train,zip(rows,cols))))
        plt.title('test')
        ax1.set_title('biases')
        ax1.matshow(train_normalized.todense())
        ax1.axis('off')
        ax2.set_title('means')
        ax2.matshow(train_mean.todense())
        ax2.axis('off')
        ax3.set_title('sum')
        ax3.matshow((train_normalized+train_mean).todense())
        ax3.axis('off')
        ax4.set_title('original')
        ax4.matshow(train.todense())
        ax4.axis('off')
        fig.savefig('../results/biases_{}'.format(m))
