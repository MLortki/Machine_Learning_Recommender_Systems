import numpy as np
from our_helpers import init_MF, 
from helpers import build_index_groups
from costs import compute_error


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




def ALS (train ,test, num_features,lambda_, stop_criterion,rng = None, test):
    
    # initialize user and movies latent matrices
    user_features, item_features = init_MF(train, num_features)
    
    
    #indices of nonzero elements
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    nz_train, nz_row_colindices, nz_col_rowindices = build_index_groups(train)   
   
    
   
    i = 0
    rmses_te = []
    rmses_tr = []
                
    while True:
    
        
        if rng != None and i >= rng:
            break
    
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
        
     
            
        #if   i > 0 and abs(rmses_tr[-1] - rmse_tr) < stop_criterion : #or i > 0 and rmse_te > rmses_te[len(rmses_te)-1] :
        if   i > 0 and abs(rmses_te[-1] - rmse_te) < stop_criterion or i > 0 and rmse_te > rmses_te[-1] :
            rmses_tr.append(rmse_tr)
            rmses_te.append(rmse_te)
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