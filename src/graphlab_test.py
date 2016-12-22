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
		for u in range(num_users):
			ratings[i][u]=results[i*num_users+u]
	return ratings


def run(model, num_items, num_users):
	test_data=make_test_sf(num_items, num_users)
	results = model.predict(test_data).to_numpy()
	ratings=make_ratings_mat(results, num_items, num_users)
	return ratings

'''
 regularization=0.00005, linear_regularization=0.00000001-> 0.986837158816
 regularization=0.00005, linear_regularization=0.00000001 ->0.9878
m=gl.recommender.factorization_recommender.create(train, target='rating', regularization=0.00005, linear_regularization=0.00000001, num_factors=20 ->9862
'''
#0.0001 (0.000051  52:9873 51:9871 (00005: 9868 solver=auto)	
def main():
	num_users, num_items=1000, 10000
	#results=SArray(np.zeros(num_items*num_users), int)	
	#res=results.	
	#simple_test(res, num_items, num_users)
	#print("done")
	#train_data = gl.cross_validation.shuffle(make_train_sf("../data/data_train.csv"))
	
	"""	
	gl.recommender.factorization_recommender.create(train_data, target='rating')	
	gl.popularity_recommender.create(train_data, target='rating', solver='als')
	gl.item_similarity_recommender.create(train, target="rating", similarity_type='cosine')	
	"""	
	sf = gl.cross_validation.shuffle(make_train_sf("../data/data_train.csv"))
	sf_train, sf_test = sf.random_split(.1, seed=5)
	
	m=gl.recommender.factorization_recommender.create(sf_train, target='rating', regularization=0.00005, solver='sgd', num_factors=8) #nmf=True, sgd_step_size=0.0000003)
 #sgd_step_size=0.001, nmf=True)
	prediction = m.predict(sf_test)
	rmse=gl.evaluation.rmse(sf_test['rating'], prediction)
	print(rmse)
"""
	#k_folds=5
	#folds = gl.cross_validation.KFold(train_data, k)	
	#mean=0

	for train, valid in folds:
		
		model=gl.recommender.factorization_recommender.create(train, target='rating', regularization=0.02, solver='sgd', num_factors=380, sgd_step_size=0.001, nmf=True)
		m = gl.popularity_recommender.create(train, target='rating')		
		prediction = m.predict(valid)
		#print(prediction)
		print(i)
		i=i+1
		rmse=gl.evaluation.rmse(valid['rating'], prediction)
		print("rms=")
		print(rmse)
		#print(m.evaluate_rmse(valid, target='rating'))
		mean+=rmse
	mean/=k
	print("mean=")
	print(mean)
		
	#params = dict([('target','rating'), ('regularization', 0.00005), ('linear_regularization',0.00000001), ('num_factors',25)])
	#job = gl.cross_validation.cross_val_score(folds, gl.recommender.factorization_recommender.create,params)
	#print job.get_results()	
	
	#m=gl.recommender.factorization_recommender.create(train_data, target='rating', regularization=0.00005, linear_regularization=0.00000001, num_factors=25)	
	#ratings=run(m, num_items, num_users)	
	#create_submission("../results/graphlab/SGD.csv", ratings)
"""
main()

#m=gl.recommender.factorization_recommender.create(train, target='rating', regularization=0.045, solver='als', num_factors=128, sgd_step_size=0.004, nmf=True) #linear_regularization=0.0000001, num_factors=1, solver='sgd') #,linear_regularization=0.00007
#als 0.00007	sgd:0.0005	00010																																	)#, num_factors=25) 0.001 sgd, 
		#reg= 0.005
		#m = gl.item_similarity_recommender.create(train, target="rating", similarity_type='pearson', threshold=2)
#, only_top_k=1)
		

