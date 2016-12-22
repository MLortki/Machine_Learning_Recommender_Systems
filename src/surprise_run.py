"""
1- pip install surprise
2- pthon surprise_run.py

"""

from surprise_helpers import run_all_algorithm

def main():
	#learn all models on blending train set
	fold_files=[('../data/blending_train_surprise.csv', '../data/data_train_surprise.csv')]
	output_prefix='training_prediction_'
	rmse_list=run_all_algorithm(fold_files, output_prefix)
	print("rmse_list : ")
	print(rmse_list)

	#learn all models on whole train set and test on submition set
	fold_files=[('../data/data_train_surprise.csv', '../data/sampleSubmission_surprise.csv')]
	output_prefix='submission_'
	rmse_list = run_all_algorithm(fold_files, output_prefix)
	print("rmse_list : ")
	print(rmse_list)

main()
