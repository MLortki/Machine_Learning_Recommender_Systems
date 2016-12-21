from helpers import build_index_groups
from costs import compute_error2
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def get_user_means(train, nz_col_rowindices):
    user_means = np.zeros((train.shape[1],1))
    for j, rowindices in nz_col_rowindices:
        user_means[j,0] = np.mean(train[rowindices,j])
    return user_means
def get_item_means(train, nz_row_colindices):
    item_means = np.zeros((train.shape[0],1))
    for i, colindices in nz_row_colindices:
        item_means[i,0] = np.mean(train[i,colindices])
    return item_means
def get_global_means(train, nz_train):
    means = sp.lil_matrix(train.shape)
    rows, cols, ratings = sp.find(train)
    mean = np.mean(ratings)
    means[rows,cols] = mean
    return means, mean

def get_unbiased_matrix(train, user_means, item_means, means, m):
    """ 
    return bias and mean matrix where user, item or combined means
    have been subtracted.
    """
    rows, cols, ratings = sp.find(train)
    train_normalized = sp.lil_matrix(train.shape)
    train_means = sp.lil_matrix(train.shape)
    if m=='no':
        print('no method ok')
        return train, sp.lil_matrix(train.shape)
    elif m=='global':
        train_normalized[rows,cols] = train[rows,cols] - means[rows,cols]
        train_means[rows,cols] = means[rows,cols]
        assert train_normalized.mean() < 1e-14
        print('global ok')
    elif m=='item':
            #train_normalized[i,colindices] = train[i,colindices]-item_means[i,0]
            #train_means[i,colindices] = item_means[i,0]
        train_means[rows,cols] = item_means[rows].T
        train_normalized[rows,cols] = train[rows,cols] - item_means[rows].T
        assert np.mean(train_normalized.mean(axis=1)) < 1e-14
        print('item ok')
    elif m=='user':
        train_means[rows,cols] = user_means[cols].T
        train_normalized[rows,cols] = train[rows,cols] - user_means[cols].T
        assert np.mean(train_normalized.mean(axis=0)) < 1e-14
        print('user ok')
    elif m=='combined':
        train_means[rows,cols] = user_means[cols].T + item_means[rows].T - means[rows,cols]
        train_normalized[rows,cols] = train[rows,cols] - train_means[rows,cols]
        print('combined ok')
    return train_normalized, train_means

def get_predictions(ratings, user_means, item_means, mean, m):
    predictions = np.empty(ratings.shape)
    item_matrix = np.tile(item_means, (1,predictions.shape[1]))
    user_matrix = np.tile(user_means.T, (predictions.shape[0],1))
    print(user_matrix.shape)
    print(item_matrix.shape)
    means_matrix = np.ones(ratings.shape)*mean
    if m=='combined':
        predictions = ratings + item_matrix + user_matrix - means_matrix
    elif m=='item':
        predictions = ratings + item_matrix 
    elif m=='user':
        predictions = ratings + user_matrix
    elif m=='global':
        predictions = ratings + mean
    return predictions

if __name__=="__main__":
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
