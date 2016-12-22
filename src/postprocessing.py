# coding: utf-8
import numpy as np
from helpers import load_data
import scipy.sparse as sp
from blending import get_all_indices, read_numpy_files, apply_indices
from our_helpers import load_data
from our_helpers import write_predictions_csv

folder = 'third'

files_full_numpy = ['../saved/blend_0.06255_8.npy']
files_numpy = ['../saved/saved_0.98464,.npy']

file_train_true = '../data/data_train.csv'
file_submission_true = '../data/sampleSubmission.csv'

files_submission = [     '../submission/submission_KNNBasic_ALS_pearson_baseline_item_based.csv',
    '../submission/submission_KNNBasic_ALS_pearson_baseline_user_based.csv',
    '../submission/submission_KNNBasic_cosine_item_based.csv',
    '../submission/submission_KNNBasic_cosine_user_based.csv',
    '../submission/submission_KNNBasic_msd_item_based.csv',
    '../submission/submission_KNNBasic_msd_user_based.csv',
    '../submission/submission_KNNBasic_pearson_baseline_item_based.csv',
    '../submission/submission_KNNBasic_pearson_baseline_user_based.csv',
    '../submission/submission_KNNBasic_pearson_user_based.csv',
    '../submission/submission_SGD_BaselineOnly.csv',
    '../submission/submission_surprise_unrounded_0.99256.csv'
                         ]
files_full_submission = [ '../submission/training_prediction_ALS_BaselineOnly.csv',
'../submission/training_prediction_KNNBasic_ALS_pearson_baseline_item_based.csv',
'../submission/training_prediction_KNNBasic_ALS_pearson_baseline_user_based.csv',
'../submission/training_prediction_KNNBasic_cosine_item_based.csv',
'../submission/training_prediction_KNNBasic_cosine_user_based.csv',
'../submission/training_prediction_KNNBasic_msd_item_based.csv',
'../submission/training_prediction_KNNBasic_msd_user_based.csv',
'../submission/training_prediction_KNNBasic_pearson_baseline_item_based.csv',
'../submission/training_prediction_KNNBasic_pearson_baseline_user_based.csv',
'../submission/training_prediction_KNNBasic_pearson_user_based.csv',
'../submission/training_prediction_SGD_BaselineOnly.csv'
                         ]
names = ['ALS_BaselineOnly',
'KNN_ALS_pearson_baseline_item',
'KNN_ALS_pearson_baseline_user',
'KNN_cosine_item',
'KNN_cosine_user',
'KNN_msd_item',
'KNN_msd_user',
'KNN_pearson_baseline_item',
'KNN_pearson_baseline_user',
'KNN_pearson_user',
'SGD_BaselineOnly'
         ]

# Create true matrices.
train_true = load_data(file_train_true)
training_true = apply_indices([train_true], 'train')[0]
test_true = apply_indices([train_true], 'test')[0]
validation_true = apply_indices([train_true], 'validation')[0]

submission_true = load_data(file_submission_true)
indices = get_all_indices(train_true, submission_true)
trains_full = read_numpy_files(files_full_numpy, indices)
trains_submission = read_numpy_files(files_numpy, indices)


train_est = apply_indices(trains_full, 'train') 
test_est = apply_indices(trains_full, 'test') 
submission_est = apply_indices(trains_submission, 'submission') 
validation_est = apply_indices(trains_full, 'validation') 


# In[7]:

# Add matrices saved in submission format.
from blending import read_submission_files
surprise_full = read_submission_files(files_full_submission)
surprise_train_est = apply_indices(surprise_full, 'train') 
surprise_test_est = apply_indices(surprise_full, 'test') 
surprise_validation_est = apply_indices(surprise_full, 'validation') 
surprise_submission = read_submission_files(files_submission)

for i in range(len(surprise_train_est)):
    validation_est.append(surprise_validation_est[i])
    train_est.append(surprise_train_est[i])
    test_est.append(surprise_test_est[i])
    submission_est.append(surprise_submission[i])

# Add new method to the whole matrix.
#files_full_submission = [ \
#    '../submission/training_prediction_KNNBasic_ALS_pearson_baseline_item_based.csv',\
#    '../submission/training_prediction_KNNBasic_ALS_pearson_baseline_user_based.csv',\
#    ]
#files_submission = [ \
#    '../submission/submission_KNNBasic_ALS_pearson_baseline_item_based.csv',\
#    '../submission/submission_KNNBasic_ALS_pearson_baseline_user_based.csv',\
#                    ]
#surprise_full = read_submission_files(files_full_submission)
#surprise_train_est = apply_indices(surprise_full, 'train') 
#surprise_test_est = apply_indices(surprise_full, 'test') 
#surprise_validation_est = apply_indices(surprise_full, 'validation') 
#surprise_submission = read_submission_files(files_submission)
#    
#validation_est[2] = surprise_validation_est[0]
#validation_est[3] = surprise_validation_est[1]
#train_est[2] = surprise_train_est[0]
#train_est[3] = surprise_train_est[1]
#test_est[2] = surprise_test_est[0]
#test_est[3] = surprise_test_est[1]
#submission_est[2] = surprise_submission[0]
#submission_est[3] = surprise_submission[1]
test_est_new = test_est
submission_est_new = submission_est
validation_est_new = validation_est



# Visualize
from data_postprocess import create_sparse_matrix_plot
# Visualize matrices
i_total, ratings_dense = create_sparse_matrix_plot(training_true, '../results/Blending/{}/matrix_train.png'.format(folder))
dense_matrices=[]
dense_matrices.append(ratings_dense.copy())
#j_total, _ = create_matrix_plot(train_true, axis=1)
for i,matrix_est in enumerate(train_est):
    __, ratings_dense = create_sparse_matrix_plot(matrix_est, '../results/Blending/{}/matrix_{}'.format(folder,i+1))
    print(ratings_dense[:5,:5])
    dense_matrices.append(ratings_dense.copy())

from data_postprocess import matrix_plot
for i in range(1,len(dense_matrices)):
    diff_matrix = dense_matrices[i]-dense_matrices[0]
    print('diff goes from {} to {}'.format(np.max(diff_matrix),np.min(diff_matrix)))
    matrix_plot(diff_matrix, '../results/Blending/{}/matrix_diff{}.png'.format(folder,i), 1000)

# Do Blending
q_hat,x = linear_blending(test_est_new, submission_est_new, test_true)

# Visualize results
cutoff = 1000
rows,cols,true_values = sp.find(validation_true)
file_name = '../results/Blending/{}/matrix_validation.png'.format(folder)
title = 'Validation data'
i_total, validation_dense = create_sparse_matrix_plot(validation_true, file_name ,cutoff, title)
Q = np.empty((len(true_values),len(validation_est_new)))
for i,validation_prediction in enumerate(validation_est_new):
    __,__,predictions = sp.find(validation_prediction)
    Q[:,i] = predictions
    rmse = np.sqrt(np.sum(np.power(predictions-true_values, 2)) / len(true_values))
    print('rmse method {}: {}'.format(i,rmse))
    file_name = '../results/Blending/{}/matrix_validation_{}.png'.format(folder, i)
    title = 'Method {}, validation error: {:1.5f}'.format(names[i], rmse)
    create_sparse_matrix_plot(validation_prediction, file_name, cutoff, title)
    print(predictions - true_values)

blending_prediction = np.dot(Q,x) 
rmse = np.sqrt(np.sum(np.power(blending_prediction-true_values, 2)) / len(true_values))
print('blending method: {}'.format(i,rmse))

# create sparse matrix for visualization
validation_blending = sp.lil_matrix(validation_true.shape)
for k, (i,j) in enumerate(zip(rows, cols)):
    validation_blending[i,j] = blending_prediction[k]
file_name = '../results/Blending/{}/matrix_blending.png'.format(folder)
title = 'Method Blending, validation error: {:1.5f}'.format(rmse)
test = create_sparse_matrix_plot(validation_blending, file_name, cutoff, title)

# create sparse matrix for submission
rows, cols, __ = sp.find(indices_matrix)
submission_blending = sp.lil_matrix(validation_true.shape)
k = 0
for (i,j) in zip(rows, cols):
    submission_blending[i,j] = q_hat[k]
    k += 1
write_predictions_csv('../results/Blending/{}/submission.csv'.format(folder), submission_blending)
