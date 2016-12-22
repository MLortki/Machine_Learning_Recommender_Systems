import scipy.sparse as sp
import numpy as np
import scipy
import scipy.io
from helpers import read_txt
from our_helpers import write_predictions 
from our_helpers import create_submission
import graphlab as gl	
from graphlab import SArray


def make_train_sf(path_dataset):
	"""
	Read csv file and make Sframe data
    
       		input:   path_dataset   -input file path
                
       		output:  sFrame         -sFrame data format of readen input file
	"""

	def deal_line(line):
		pos, rating = line.split(',')
		row, col = pos.split("_")
		row = row.replace("r", "")
		col = col.replace("c", "")
        	return int(row), int(col), float(rating)
	
	data = read_txt(path_dataset)[1:]
	users=[]
	items=[]
	ratings=[]	
	for line in data:		
		row,col,rating=deal_line(line)		
		items.append(row)
		users.append(col)
		ratings.append(rating)
	return gl.SFrame({'user_id': users,'item_id': items,'rating': ratings})

def make_test_sf(num_items, num_users):
	"""
	make sFrame data format for whole matrix
    
       		input:   num_items    -number of items
			 num_users    -number of users
                
       		output:  sFrame       -sFrame data format of whole matrix
	"""
	users=[]
	items=[]
	for i in range(num_items):
		for u in range(num_users):
			users.append(u)
			items.append(i)
	return gl.SFrame({'user_id': users,'item_id': items})

def make_ratings_mat(results, num_items, num_users):
	ratings=np.zeros((num_items, num_users))
	for i in range(num_items):
		for u in range(num_users):
			ratings[i][u]=results[i*num_users+u]
	return ratings


def run_save(model, num_items, num_users):
	test_data=make_test_sf(num_items, num_users)
	results = model.predict(test_data).to_numpy()
	ratings=make_ratings_mat(results, num_items, num_users)
	create_submission("../results/graphlab/SGD.csv", ratings)
	return ratings

def cross_validation(k, data):
	rmse_list=[]
	for train, valid in folds:
		model=gl.recommender.factorization_recommender.create(train, target='rating', regularization=0.00005)	
		prediction = model.predict(valid)
		rmse_list.append(gl.evaluation.rmse(valid['rating'], prediction))
	return np.mean(rmse_list)
		

def main():
	num_users, num_items, k_fold=1000, 10000,10
	train_data = gl.cross_validation.shuffle(make_train_sf("../data/data_train.csv"))
	
	#evaluate rmse error of sgd algorithm
	params = dict([('target','rating'), ('regularization', 0.00005), ('num_factors',8), ('solver', 'sgd')])
	folds = gl.cross_validation.KFold(train_data, k_fold)
	job = gl.cross_validation.cross_val_score(folds, gl.recommender.factorization_recommender.create, params)
	print job.get_results()


	#learn the model
	model=gl.recommender.factorization_recommender.create(train_data, target='rating', regularization=0.00005, solver='sgd', num_factors=8)
	#model=gl.popularity_recommender.create(train_data, target='rating', solver='als')
	#model=gl.item_similarity_recommender.create(train_data, target="rating", similarity_type='cosine')	
	

	#save submition results
	run_save(model, num_items, num_users)

main()

		

