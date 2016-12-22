import numpy as np
from surprise.evaluate import evaluate
from surprise.accuracy import rmse
from surprise.dump import dump
from surprise.prediction_algorithms import BaselineOnly
from surprise.prediction_algorithms import KNNBasic
from surprise.prediction_algorithms import KNNWithMeans
from surprise.prediction_algorithms import KNNBaseline
from surprise.prediction_algorithms import SVD


def cross_validation(algo, fold_data):
    """
        Cross validation on folded data by specified algorithm.
    
        input:   algo           -Learning algorithm method
                fold_data      -List of (train_data, test_data) to do cross validation on it
                
        output:  rmse_mean      -The mean of rmse on all (train_data, test_data) in fold_data
                prediction     -The prediction on test_data  
    """
    
    rmse_list = []
    for trainset, testset in fold_data.folds():
        #train the learning model on trainset using given algorithm
        algo.train(trainset)
        #predcit the result on testset using the trained model
        prediction = algo.test(testset)
        
        #compute rmse
        rmse_k = rmse(prediction, verbose=False)
        print(rmse_k)
        rmse_list.append(rmse_k)
        #dump('../results/dump_algo', prediction, trainset, algo)
    
    
    rmse_mean=np.mean(rmse_list)
    print(rmse_mean)
    return rmse_mean, prediction



#params = {'n_factors':12,'n_epochs':20,'lr_all':0.005,'reg_all':0.0359,'biased':True}
#params = {'n_factors':100,'n_epochs':20,'lr_all':0.005,'reg_all':0.0359,'biased':True}
def my_SVD(n_factors, n_epochs, lr_all, reg_all, biased):
    """
        SVD method
    
        input:  n_factors    - The number of factors
                n_epochs     - The number of iteration of the SGD procedure
                lr_all       - The learning rate for all parameters.
                reg_all      - The regularization term for all parameters
                biased       - Whether to use baselines (or biases)
                
        output: algo         - SVD algorithm based on specified parameters 
    """
    algo = SVD(n_factors=n_factors,n_epochs=n_epochs,lr_all=lr_all,reg_all=reg_all)
    algo.bsl_options['biased'] = biased
    return algo

def ALS_BaselineOnly():
    """
        BaselineOnly method using ALS
        
        input:  --            --
        output: algorithm     -BaselineOnly method using ALS
    """
    print("ALS_BaselineOnly")
    bsl_options = {'method': 'als',
               'n_epochs': 5,
               'reg_u': 15,
               'reg_i': 10
               }
    return BaselineOnly(bsl_options=bsl_options)

def SGD_BaselineOnly():
    """
        BaselineOnly method using SGD
        
        input:  --            --
        output: algorithm     -BaselineOnly method using SGD
    """
    print("SGD_BaselineOnly")
    bsl_options = {'method': 'sgd',
               'learning_rate': .005,
               'reg':0.02
                }
    return BaselineOnly(bsl_options=bsl_options)


def KNNBasic_ALS_pearson_baseline_user_based():
    """
        KNNBasic method based on pearson baseline similarity of users using ALS
        
        input:  --           --
        output: algorithm    -KNNBasic method based on pearson baseline similarity of users using ALS
    """
    print("KNNBasic_ALS_pearson_baseline_user_based")
    bsl_options = {'method': 'als',
                    'n_epochs': 20,
                    'user_based': True  # compute  similarities between users
                   } 
    sim_options = {'name': 'pearson_baseline'}
    return KNNBasic(bsl_options=bsl_options, sim_options=sim_options)


def KNNBasic_ALS_pearson_baseline_item_based():
    """
        KNNBasic method based on pearson baseline similarity of items using ALS
        
        input:  --           --
        output: algorithm    -KNNBasic method based on pearson baseline similarity of items using ALS
    """
    print("KNNBasic_ALS_pearson_baseline_item_based")
    bsl_options = {'method': 'als',
               'n_epochs': 20,
                'user_based': False  # compute  similarities between users
               } 
    sim_options = {'name': 'pearson_baseline'}
    return KNNBasic(bsl_options=bsl_options, sim_options=sim_options)

def KNNBasic_pearson_baseline_user_based():
    """
        KNNBasic method based on baseline similarity of users
        
        input:  --           --
        output: algorithm    -KNNBasic method based on baseline similarity of users
    """
    
    print("KNNBasic_pearson_baseline_user_based")
    sim_options = {'name': 'pearson_baseline',
               'shrinkage': 0,  # no shrinkage
                'user_based': True  # compute  similarities between users
               } 
    return KNNBasic(sim_options=sim_options)

def KNNBasic_pearson_baseline_item_based():
    """
        KNNBasic method based on baseline similarity of items
        
        input:  --           --
        output: algorithm    -KNNBasic method based on baseline similarity of items
    """
    
    print("KNNBasic_pearson_baseline_item_based")
    sim_options = {'name': 'pearson_baseline',
               'shrinkage': 0,  # no shrinkage
                'user_based': False  # compute  similarities between items
               }
    return KNNBasic(sim_options=sim_options)


def KNNBasic_cosine_user_based():
    """
        KNNBasic method based on cosine similarity of users
        
        input:  --           --
        output: algorithm    -KNNBasic method based on cosine similarity of users
    """
    
    print("KNNBasic_cosine_user_based")
    sim_options = {'name': 'cosine',
                   'user_based': True  # compute  similarities between users
                   }
    return KNNBasic(sim_options=sim_options)

def KNNBasic_cosine_item_based():
    """
        KNNBasic method based on cosine similarity of items
        
        input:  --           --
        output: algorithm    -KNNBasic method based on cosine similarity of items
    """
    
    print("KNNBasic_cosine_item_based")
    sim_options = {'name': 'cosine',
                   'user_based': False  # compute  similarities between items
                   }
    return KNNBasic(sim_options=sim_options)

def KNNBasic_pearson_user_based():
    """
        KNNBasic method based on pearson similarity of users
        
        input:  --           --
        output: algorithm    -KNNBasic method based on pearson similarity of users
    """
    
    print("KNNBasic_pearson_user_based")
    sim_options = {'name': 'pearson',
                'user_based': True  # compute  similarities between users
               } 
    return KNNBasic(sim_options=sim_options)

def KNNBasic_pearson_item_based():
    """
        KNNBasic method based on pearson similarity of items
    
        input:  --           --
        output: algorithm    -KNNBasic method based on pearson similarity of items
    """
    
    print("KNNBasic_pearson_item_based")
    sim_options = {'name': 'pearson',
                'user_based': False  # compute  similarities between items
               }
    return KNNBasic(sim_options=sim_options)

def KNNBasic_msd_user_based():
    """
        KNNBasic method based on msd similarity of users
    
        input:  --           --
        output: algorithm    -KNNBasic method based on msd similarity of users
    """
    
    print("KNNBasic_msd_user_based")
    sim_options = {'name': 'msd',
                'user_based': True  # compute  similarities between users
               } 
    return KNNBasic(sim_options=sim_options)

def KNNBasic_msd_item_based():
    """
        KNNBasic method based on msd similarity of items
    
        input:  --           --
        output: algorithm    -KNNBasic method based on msd similarity of items
    """
    
    print("KNNBasic_msd_item_based")
    sim_options = {'name': 'msd',
                'user_based': False  # compute  similarities between items
               }
    return KNNBasic(sim_options=sim_options)

def KNNBaseline_pearson_baseline_user_based():
    """
        KNNBaseline method based on pearson baseline similarity of users
        
        input:  --           --
        output: algorithm    -KNNBaseline method based on pearson baseline similarity of users
    """
    
    print("KNNBaseline_pearson_baseline_user_based")
    sim_options = {'name': 'pearson_baseline',
               'shrinkage': 0,  # no shrinkage
                'user_based': True  # compute  similarities between users
               } 
    return KNNBaseline(sim_options=sim_options)

def KNNBaseline_pearson_baseline_item_based():
    """
        KNNBaseline method based on baseline similarity of items
        
        input:  --           --
        output: algorithm    -KNNBaseline method based on baseline similarity of items
    """
    
    print("KNNBaseline_pearson_baseline_item_based")
    sim_options = {'name': 'pearson_baseline',
               'shrinkage': 0,  # no shrinkage
                'user_based': False  # compute  similarities between items
               }
    return KNNBaseline(sim_options=sim_options)


def KNNBaseline_cosine_user_based():
    """
        KNNBaseline method based on cosine similarity of users
        
        input:  --           --
        output: algorithm    -KNNBaseline method based on cosine similarity of users
    """
    
    print("KNNBaseline_cosine_user_based")
    sim_options = {'name': 'cosine',
                   'user_based': True  # compute  similarities between users
                   }
    return KNNBaseline(sim_options=sim_options)

def KNNBaseline_cosine_item_based():
    """
        KNNBaseline method based on cosine similarity of items
        
        input:  --           --
        output: algorithm    -KNNBaseline method based on cosine similarity of items
    """
    
    print("KNNBaseline_cosine_item_based")
    sim_options = {'name': 'cosine',
                   'user_based': False  # compute  similarities between items
                   }
    return KNNBaseline(sim_options=sim_options)

def KNNBaseline_pearson_user_based():
    """
        KNNBaseline method based on pearson similarity of users
        
        input:  --           --
        output: algorithm    -KNNBaseline method based on pearson similarity of users
    """
    
    print("KNNBaseline_pearson_user_based")
    sim_options = {'name': 'pearson',
                'user_based': True  # compute  similarities between users
               } 
    return KNNBaseline(sim_options=sim_options)

def KNNBaseline_pearson_item_based():
    """
        KNNBaseline method based on pearson similarity of items
    
        input:  --           --
        output: algorithm    -KNNBaseline method based on pearson similarity of items
    """
    
    print("KNNBaseline_pearson_item_based")
    sim_options = {'name': 'pearson',
                'user_based': False  # compute  similarities between items
               }
    return KNNBaseline(sim_options=sim_options)

def KNNBaseline_msd_user_based():
    """
        KNNBaseline method based on msd similarity of users
    
        input:  --           --
        output: algorithm    -KNNBaseline method based on msd similarity of users
    """
    
    
    print("KNNBaseline_msd_user_based")
    sim_options = {'name': 'msd',
                'user_based': True  # compute  similarities between users
               } 
    return KNNBaseline(sim_options=sim_options)

def KNNBaseline_msd_item_based():
    """
        KNNBaselinec method based on msd similarity of items
    
        input:  --           --
        output: algorithm    -KNNBaseline method based on msd similarity of items
    """
    
    print("KNNBaseline_msd_item_based")
    sim_options = {'name': 'msd',
                'user_based': False  # compute  similarities between items
               }
    return KNNBaseline(sim_options=sim_options)

def KNNWithMeans_pearson_baseline_user_based():
    """
        KNNWithMeans method based on pearson baseline similarity of users
        
        input:  --           --
        output: algorithm    -KNNWithMeans method based on pearson baseline similarity of users
    """
    
    print("KNNWithMeans_pearson_baseline_user_based")
    sim_options = {'name': 'pearson_baseline',
               'shrinkage': 0,  # no shrinkage
                'user_based': True  # compute  similarities between users
               } 
    return KNNWithMeans(sim_options=sim_options)

def KNNWithMeans_pearson_baseline_item_based():
    """
        KNNWithMeans method based on pearson baseline similarity of items
        
        input:  --           --
        output: algorithm    -KNNWithMeans method based on pearson baseline similarity of items
    """
    
    print("KNNWithMeans_pearson_baseline_item_based")
    sim_options = {'name': 'pearson_baseline',
               'shrinkage': 0,  # no shrinkage
                'user_based': False  # compute  similarities between items
               }
    return KNNWithMeans(sim_options=sim_options)


def KNNWithMeans_cosine_user_based():
    """
        KNNWithMeans method based on cosine similarity of users
        
        input:  --           --
        output: algorithm    -KNNWithMeans method based on cosine similarity of users
    """
    
    print("KNNWithMeans_cosine_user_based")
    sim_options = {'name': 'cosine',
                   'user_based': True  # compute  similarities between users
                   }
    return KNNWithMeans(sim_options=sim_options)

def KNNWithMeans_cosine_item_based():
    """
        KNNWithMeans method based on cosine similarity of items
        
        input:  --           --
        output: algorithm    -KNNWithMeans method based on cosine similarity of items
    """
    
    print("KNNWithMeans_cosine_item_based")
    sim_options = {'name': 'cosine',
                   'user_based': False  # compute  similarities between items
                   }
    return KNNWithMeans(sim_options=sim_options)

def KNNWithMeans_pearson_user_based():
    """
        KNNWithMeans method based on pearson similarity of users
        
        input:  --           --
        output: algorithm    -KNNWithMeans method based on pearson similarity of users
    """
    
    print("KNNWithMeans_pearson_user_based")
    sim_options = {'name': 'pearson',
                'user_based': True  # compute  similarities between users
               } 
    return KNNWithMeans(sim_options=sim_options)

def KNNWithMeans_pearson_item_based():
    """
        KNNWithMeans method based on pearson similarity of items
    
        input:  --           --
        output: algorithm    -KNNWithMeans method based on pearson similarity of items
    """
    
    print("KNNWithMeans_pearson_item_based")
    sim_options = {'name': 'pearson',
                'user_based': False  # compute  similarities between items
               }
    return KNNWithMeans(sim_options=sim_options)

def KNNWithMeans_msd_user_based():
    """
        KNNWithMeans method based on msd similarity of users
    
        input:  --           --
        output: algorithm    -KNNWithMeans method based on msd similarity of users
    """
    
    print("KNNWithMeans_msd_user_based")
    sim_options = {'name': 'msd',
                'user_based': True  # compute  similarities between users
               } 
    return KNNWithMeans(sim_options=sim_options)

def KNNWithMeans_msd_item_based():
    """
        KNNWithMeans method based on msd similarity of items
    
        input:  --           --
        output: algorithm    -KNNWithMeans method based on msd similarity of items
    """
        
    print("KNNWithMeans_msd_item_based")
    sim_options = {'name': 'msd',
                'user_based': False  # compute  similarities between items
               }
    return KNNWithMeans(sim_options=sim_options)


def train_SVD():
    """
        Find best parameters for SVD algorithm.
       
        input:   --      --
                
        output:  rmses   -The rmse list of running SVD on all parameters set 
    """
    
    #initialize parameters
    n_factors_range = np.array([10,15,20])    #number of columns
    reg_all_range = np.logspace(-1.9,-1,10)   #The regularization term for all parameters
    n_epochs_range = np.arange(10,60,10)      #The number of iteration of the SGD procedure.
    lr_all=0.005                              #The learning rate for all parameters
    biased=True                               #use baselines (or biases)
    
    
    results_path = '../results/SGD_surprise/'
    rmses = np.empty((len(n_factors_range),len(reg_all_range), len(n_epochs_range)))
    
    for i,n_factors in enumerate(n_factors_range):
        print('testing with n_factors={}'.format(n_factors))
        for j,reg_all in enumerate(reg_all_range):
            print('testing with reg_all={}'.format(reg_all))
            for k,n_epochs in enumerate(n_epochs_range):
                print('testing with n_epochs={}'.format(n_epochs))
                
                #train SVD based on given parameters 
                algo=my_SVD(int(n_factors), n_epochs, lr_all, reg_all, biased)
                
                #cross validation on train_data and compute rmse
                rmses[i,j,k],_=cross_validation(algo, train_data)
                print('rmse={}'.format(rmses[i,j,k]))
            
            results_name = 'rmse_{}_{}'.format(n_factors, reg_all)
            np.savetxt(results_path + results_name + '.csv', rmses[i,j,:], delimiter=",")
    return rmses


