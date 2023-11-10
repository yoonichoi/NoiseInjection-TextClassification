
from config import * 
from methods import *
import random
import pickle
import argparse
import os

###############################
#### run model and get acc ####
###############################

def run_model(train_file, test_file, num_classes, percent_dataset, analyze_path=None):

	#initialize model
	model = build_model(input_size, word2vec_len, num_classes)

	#load data
	train_x, train_y = get_x_y(train_file, num_classes, word2vec_len, input_size, word2vec, percent_dataset)
	test_x, test_y = get_x_y(test_file, num_classes, word2vec_len, input_size, word2vec, 1)

	#implement early stopping
	callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

	#train model
	model.fit(	train_x, 
				train_y, 
				epochs=100000, 
				callbacks=callbacks,
				validation_split=0.1, 
				batch_size=1024, 
				shuffle=True, 
				verbose=0)
	#model.save('checkpoints/lol')
	#model = load_model('checkpoints/lol')

	#evaluate model
	y_pred = model.predict(test_x)
	test_y_cat = one_hot_to_categorical(test_y)
	y_pred_cat = one_hot_to_categorical(y_pred)
	acc = accuracy_score(test_y_cat, y_pred_cat)

	if analyze_path is not None:
		os.mkdir(analyze_path)
		save_pickle(analyze_path + '/y_pred.pkl', y_pred)
		save_pickle(analyze_path + '/test_x.pkl', test_x)
		save_pickle(analyze_path + '/test_y.pkl', test_y)
		print("pickle files saved to", analyze_path)

	#clean memory???
	train_x, train_y = None, None
	gc.collect()

	#return the accuracy
	#print("data with shape:", train_x.shape, train_y.shape, 'train=', train_file, 'test=', test_file, 'with fraction', percent_dataset, 'had acc', acc)
	return acc

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--analyze', action='store_true')
	args = parser.parse_args()

	sd = args.seed
	analyze = args.analyze

	analyze_mode = True if analyze else False
	print("analyze mode:", analyze_mode)
	random.seed(sd)

	#get the accuracy at each increment
	orig_accs = {dataset:{} for dataset in datasets}
	eda_accs = {dataset:{} for dataset in datasets}
	aeda_accs = {dataset:{} for dataset in datasets}

	# make folder named with timenow_results
	nowstr = get_now_str()
	os.mkdir(f'reproduce_fig2/outputs/{nowstr}')
	basepath = f'reproduce_fig2/outputs/{nowstr}'

	writer = open(f'{basepath}/result_{sd}.csv', 'w')

	#for each dataset
	for i, dataset_folder in enumerate(dataset_folders):

		dataset = datasets[i]
		num_classes = num_classes_list[i]
		input_size = input_size_list[i]

		train_orig = dataset_folder + '/train_orig.txt'
		train_eda = dataset_folder + '/train_eda.txt'
		train_aeda = dataset_folder + '/train_aeda.txt'

		test_path = dataset_folder + '/test.txt'
		word2vec_pickle = dataset_folder + '/word2vec.pkl'
		word2vec = load_pickle(word2vec_pickle)

		for increment in increments:
			
			#calculate aeda accuracy
			aeda_acc = run_model(train_aeda, test_path, num_classes, increment, analyze_path=f'{basepath}/{dataset}/{increment}/aeda' if analyze_mode else None)
			aeda_accs[dataset][increment] = aeda_acc

			#calculate eda accuracy
			eda_acc = run_model(train_eda, test_path, num_classes, increment, analyze_path=f'{basepath}/{dataset}/{increment}/eda' if analyze_mode else None)
			eda_accs[dataset][increment] = eda_acc

			#calculate original accuracy
			orig_acc = run_model(train_orig, test_path, num_classes, increment, analyze_path=f'{basepath}/{dataset}/{increment}/orig' if analyze_mode else None)
			orig_accs[dataset][increment] = orig_acc

			print(dataset, increment, orig_acc, eda_acc, aeda_acc)
			writer.write(dataset + ',' + str(increment) + ',' + str(orig_acc) + ',' + str(eda_acc) + ',' + str(aeda_acc) + '\n')
			writer.flush()

			gc.collect()
		

	print(orig_accs, eda_accs, aeda_accs)
