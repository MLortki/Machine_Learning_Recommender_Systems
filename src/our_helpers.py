# coding: utf-8
import numpy as np
import scipy.sparse as sp
import csv
from helpers import calculate_mse, load_data
from helpers import build_index_groups
import plots as pl
import sklearn.model_selection as skm

def create_submission(path_output, ratings):
    path_sample = "../data/sampleSubmission.csv"
    ratings_nonzero  = load_data(path_sample)
    (rows, cols, data) = sp.find(ratings_nonzero)
    fieldnames = ['Id', 'Prediction']
    with open(path_output, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i,row in enumerate(rows):
            valid_rating = min(max(ratings[row,cols[i]],1),5)
            valid_rating = round(valid_rating)
            _id = "r{0}_c{1}".format(row+1,cols[i]+1)
            writer.writerow({'Id': _id, 'Prediction': valid_rating})
            
            


def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings:
            all users and items we keep must have at least min_num_ratings per user and per item.
    """
    # set seed
    np.random.seed(988)

    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][:, valid_users]

    count_users = valid_ratings.shape[1]
    count_items = valid_ratings.shape[0]
    count_total = count_items * count_users

    # this is supposedly the fastest way of doing this.
    cx = sp.coo_matrix(valid_ratings)
    test = sp.lil_matrix(ratings.shape)
    train = sp.lil_matrix(ratings.shape)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        # put the element with probability 0.1 in test set.
        if (np.random.uniform()<p_test):
            test[i,j] = v
        # put the element with probability 0.9 in train set.
        else:
            train[i,j] = v

    (rows, cols, datas) = sp.find(valid_ratings)

    #  print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    #  print("Train shape:{s}, num:{n}".format(s=train.shape, n=train.shape[0]*train.shape[1]))
    #  print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    #  print("Test shape:{s}, num:{n}".format(s=test.shape, n=test.shape[0]*test.shape[1]))
    #  print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    print("Percentage of nz train data: % 2.4f, percentage of nz test data: % \
            2.4f" % (train.nnz/valid_ratings.nnz, test.nnz/valid_ratings.nnz))
    assert (train.nnz + test.nnz) == valid_ratings.nnz, "Number of nnz elements in test and train test doesn't sum up!"
    return valid_ratings, train, test

def baseline_global_mean(train, test):
    """baseline method: use the global mean."""
    mean = (train!=0).mean()
    # sum over differences between actual values and global mean.
    sum_mse = 0
    num = 0
    # this is supposedly the fastest way of doing this.
    cx = sp.coo_matrix(test)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        sum_mse += (v-mean)**2
        num += 1
    return sum_mse/(num*2.0)

# TODO: Aida's version only working for dense matrices?
def baseline_global_mean_aida(train, test):
    """baseline method: use the global mean."""
    global_predict_train=(train!=0).mean()
    num_test=np.count_nonzero(test)
    test_real_values=test.ravel()[np.flatnonzero(test)]
    test_mse=1/(2.0*num_test)*calculate_mse(test_real_values, np.full(num_test, global_predict_train))
    return test_mse

def baseline_user_mean(train, test):
    """baseline method: use the user means as the prediction."""
    user_mean = (train!=0).mean(axis=0).reshape(-1,1)
    # sum over differences between actual values and global mean.
    sum_mse = 0
    num = 0
    # this is supposedly the fastest way of doing this.
    cx = sp.coo_matrix(test)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        sum_mse += (v-user_mean[j,0])**2
        num += 1
    return sum_mse/(num*2.0)

def baseline_item_mean(train, test):
    """baseline method: use item means as the prediction."""
    item_mean = (train!=0).mean(axis=1).reshape(-1,1)
    # sum over differences between actual values and global mean.
    sum_mse = 0
    num = 0
    # this is supposedly the fastest way of doing this.
    cx = sp.coo_matrix(test)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        sum_mse += (v-item_mean[i,0])**2
        num += 1
    return sum_mse/(num*2.0)

def baseline_combined(train, test):
    mean = (train!=0).mean()
    item_mean = (train!=0).mean(axis=1).reshape(-1,1)
    user_mean = (train!=0).mean(axis=0).reshape(-1,1)

    # this is supposedly the fastest way of doing this.
    sum_mse = 0
    num = 0
    cx = sp.coo_matrix(test)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        b_user = user_mean[j,0] - mean
        b_item = item_mean[i,0] - mean
        prediction = mean + b_user + b_item 
        sum_mse += (v - prediction)**2
        num += 1
    return sum_mse/(num*2.0)


def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    
    
    user_features = 3 * np.random.rand( num_features, train.shape[1])
    item_features = 3 * np.random.rand( num_features, train.shape[0])
    
    item_features[0, :] = train[train != 0].mean(axis = 1)
    
    return user_features, item_features


def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # calculate rmse (we only consider nonzero entries.)
    # ***************************************************
    X = np.dot ( np.transpose(item_features), user_features )
    
    rmse = 0
    counter = 0
    for i, j in nz:
        
        rmse += (data[i,j] - X[i,j])**2
        counter += 1
    
    rmse = rmse/(counter)
    return rmse

def compute_error2(data, pred, nz):
    
    rmse = 0
    counter = 0
    for i, j in nz:
        
        rmse += (data[i,j] - pred[i,j])**2
        counter += 1
    
    rmse = rmse/(counter)
    return rmse

def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user):   #, nz_user_itemindices
    """update user feature matrix."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # update and return user feature.
    # ***************************************************
    N = train.shape[1]
    D = train.shape[0]
    K = item_features.shape[0]
    
    new_user_features = np.zeros((K, N))
    
    #print("UPDATE USER")
    for g , value in nnz_items_per_user:
    
        #print("column index = ", g)
        #print("elementebis raodenoba = ", value.shape)
        #scipy lil sparse matrixisdan ro gadaviyvanot numpy arrayshi
        #print(g)
        X = train[value, g].toarray()
        #print("shape of new train data = ", X.shape)
        Wt = item_features[: , value]
        #print("shape of Wt = ", Wt.shape)
        #print(type(Wt))
        #print(type(X))
        new_user_features[:, g] = np.linalg.solve( 
            np.dot(Wt, np.transpose(Wt))+ lambda_user* value.shape[0]* np.eye(K) ,
            np.dot( Wt, X)).flatten()
    
    
    return new_user_features

def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item):   #, nz_item_userindices
    """update item feature matrix."""
    N = train.shape[1]
    D = train.shape[0]
    K = user_features.shape[0]
        
    new_item_features = np.zeros((K, D))
    #print("UPDATE ITEM")
    for g, value  in nnz_users_per_item:
        
        #print("column index = ", g)
        #print("elementebis raodenoba = ", value.shape)
        
        X = train[g,value].toarray()
        #print("shape of new train data = ", X.shape)
        Zt = user_features[: , value]
        #print("shape of Zt = ", Zt.shape)
        new_item_features[:, g] = np.linalg.solve( 
            np.dot(Zt, np.transpose(Zt))+ lambda_item * value.shape[0]* np.eye(K)  ,
            np.dot( Zt, np.transpose(X))).flatten()
        
    
    return new_item_features

def ALS (train ,test, num_features,lambda_, stop_criterion,rng = None):
    
    # initialize user and movies latent matrices
    user_features, item_features = init_MF(train, num_features)
    
    
    #indices of nonzero elements
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    nz_train, nz_row_colindices, nz_col_rowindices = build_index_groups(train)   
   
  
    #print(nz_train)
    #print(nz_row_colindices )
    #print("Original data shape: ", train.shape)
    #print(nz_train.shape)
    #print(nz_row_colindices.shape)
    #print(nz_col_rowindices)
    
   
    i = 0
    rmses_te = []
    rmses_tr = []
                
    while True:
    

    
    #for i in range(rng):

        user_features_new = update_user_feature(
            train, item_features, lambda_, nz_col_rowindices)
        if( user_features_new.shape != user_features.shape):
            print("AAAAA")
        item_features_new = update_item_feature(
            train, user_features_new, lambda_, nz_row_colindices)

        rmse_te = compute_error(test, user_features_new, item_features_new , nz_test)
        rmse_tr = compute_error(train, user_features_new, item_features_new , nz_train)
        
     
        
        #if i % 5 == 0:
            #train_errors.append(rmse_tr)
            #test_errors.append(rmse_te)
        print("iter: {}, RMSE on training set: {}.".format(i, rmse_tr))
        print("iter: {}, RMSE on test set: {}.".format(i, rmse_te))
        #i +=1
     
            
        if i > 0 and rmse_te > rmses_te[len(rmses_te)-1]:
            break
            
        if i > 0 and rmses_te[-1] - rmse_te < stop_criterion:
            break
        
        rmses_tr.append(rmse_tr)
        rmses_te.append(rmse_te)

        item_features = item_features_new
        user_features = user_features_new
        
        i += 1
    
    
    rmse_te = compute_error(test, user_features, item_features, nz_test)
    rmse_tr = compute_error(train, user_features, item_features, nz_train)
    print("Final RMSE on test data: {}.".format(rmse_te))

    return item_features, user_features,  rmses_tr, rmses_te 

def gen_train_test(ratings, nz_ratings, i_ind , j_ind):
    
    #estimating shape
    d = ratings.shape[0]
    n = ratings.shape[1] 
    
    
    train_index = [ nz_ratings[i] for i in i_ind]
    test_index = [ nz_ratings[j] for j in j_ind]

    #print("train index size = ", len(train_index))
    #print("test index size = ", len(test_index))

    
    if len(train_index) + len(test_index) != len(nz_ratings):
        print("Wrong !!")

    train = np.zeros((d,n )) 
    test  = np.zeros((d,n))

    for ind in train_index:
        i = ind[0]
        j = ind[1]
        train[i,j] = ratings[i,j]
        
    for ind in test_index:
        i = ind[0]
        j = ind[1]
        test[i,j] = ratings[i,j]
        
    train = sp.lil_matrix(train)
    test = sp.lil_matrix(test)
    
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    #print("nonzero elems in train: ", len(nz_train))
    #print("nonzero elems in test: ", len(nz_test))
    
    return train, test

def cross_validation(
    ratings, n_of_splits,num_features,lambdas,
    stop_criterion):

   
    #cross_validation(n_splits)
    kf = skm.KFold(n_splits=n_of_splits, shuffle = True)
    
    #creating matrix of non-zero indices
    nz_row, nz_col = ratings.nonzero()
    nz_ratings = list(zip(nz_row, nz_col))
    
    #print("Length of list of nonzero elems of ratings is : ", len(nz_ratings) )
    
    t = range(nz_row.shape[0])
    
    #t = nz_row.shape[0]
    #nz_row = nz_row.reshape(t,1)
    #nz_col = nz_col.reshape(t,1)
    #nz_ratings = np.concatenate((nz_row, nz_col), axis = 1)
    
    #print(nz_row.shape)
    #print(nz_col.shape)
    #print(nz_ratings.shape)
    
    # creating matrix where results of cross validation are stored
    test_avg_cost =  np.zeros(len(lambdas))
    train_avg_cost = np.zeros(len(lambdas))
    errors = []

     
    for ind,lambda_ in enumerate(lambdas):
            
        print(ind+1, "/",len(lambdas))
     
            
        avg_train = 0
        avg_test = 0
            
        print("lambda  = ",lambda_)

        for i, j in kf.split(t):
            
            
            
            #print("train: ",i.shape)
            #print("test: ",j.shape)
            #print(j)
            #print("Percentage of train and test reps. : ", 
                  #len(i)*100/nz_row.shape[0],"% ", len(j)*100/nz_row.shape[0],"%" )
            
            train , test = gen_train_test(ratings, nz_ratings, i , j)
            itf, usf, rmses_tr, rmses_te = ALS (train , test, num_features,lambda_, 
                                               stop_criterion )
                
            avg_train += rmses_tr[-1]
            avg_test += rmses_te[-1]
                    
                
                
        avg_train /= n_of_splits
        avg_test /= n_of_splits
        print("average train error = ",avg_train)
        print("average test error = ",avg_test)
        test_avg_cost[ind] = avg_test
        train_avg_cost[ind] = avg_train
        errors.append(rmses_te)
    
    return test_avg_cost, train_avg_cost, errors
            

def bias_correction (full_ratings, test):
    
    nz_rows, nz_cols = test.nonzero()
    nz_test = list( zip(nz_rows, nz_cols))
    
    mean_te = 0
    mean_pr = 0
    for i,j in nz_test:
        mean_te += test[i,j]
        mean_pr += full_ratings[i,j]
        
    mean_te /= len(nz_test)
    mean_pr /= len(nz_test)
    
    #mean_pr= full_ratings.mean()
    full_ratings += (mean_te - mean_pr)
    
    return full_ratings


def main(ratings):
    
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    num_features = 1   # K in the lecture notes
    lambda_user = 0.3
    lambda_item = 0.3
    stop_criterion = 1e-4
    n_splits = 2

    
    #initialization
    sratings = sp.lil_matrix(ratings)
    train_errors = []
    test_errors = []
    
    lambdas = np.linspace( 0.001, 1, 2)

    
    print("number of different lambdas : ",len(lambdas))
    
    # set seed
    np.random.seed(988)
    
    
    test_avg_cost, train_avg_cost , errors = cross_validation(
        sratings, n_splits, num_features, lambdas, stop_criterion)
    
    #generating plot
    path = "K%d/l%d_nsp%d.jpg"%(num_features, len(lambdas),n_splits )
    
    pl.plot_cv_errors(errors, lambdas, num_features, path)
    
    ind = np.argmin(test_avg_cost)
    print("smallest avg error: ",test_avg_cost[ind])
    
    lambda_ = lambdas[ind]
    
    
    vl, train, test = split_data(ratings, num_items_per_user, num_users_per_item,0)
    
    item_features , user_features , rmse_tr, rmse_te = ALS (
        train , test, num_features,lambda_[0], 
        lambda_[1], stop_criterion,error_list, 250 )
    
    ratings_full = np.dot(np.transpose(item_features),user_features)
    
    return ratings_full, train_errors, test_errors

