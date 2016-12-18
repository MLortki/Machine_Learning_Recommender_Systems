import scipy.sparse as sp
import numpy as np
import scipy
import scipy.io
from helpers import read_txt
from our_helpers import create_submission
import graphlab as gl	


def make_train_sf(path_dataset):
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
		if(i%10==0):
			print(i)
		for u in range(num_users):
			ratings[i][u]=results[i*num_users+u]
	return ratings

def run(model, num_items, num_users):
	test_data=make_test_sf(num_items, num_users)
	results = model.predict(test_data)
	ratings=make_ratings_mat(results, num_items, num_users)
	return ratings


def main():
	num_users, num_items=1000, 10000
	train_data = gl.cross_validation.shuffle(make_train_sf("../data/data_train.csv"))
	
	"""	
	gl.recommender.factorization_recommender.create(train_data, target='rating')	
	gl.popularity_recommender.create(train_data, target='rating', solver='als')
	"""	
	model=gl.recommender.factorization_recommender.create(train_data, target='rating')	
	
	folds = gl.cross_validation.KFold(train_data, 10)
	params = dict([('target', 'rating'), ('solver','als')])
	job = gl.cross_validation.cross_val_score(folds, gl.recommender.factorization_recommender.create,params,gl.evaluation.rmse)
	print job.get_results()
	
	ratings=run(model, num_items, num_users)	
	print(ratings)
	create_submission("../newsubmition.csv", ratings)


main()
