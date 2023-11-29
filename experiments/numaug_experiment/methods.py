

from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import math
import time
import numpy as np
import random
from random import randint
random.seed(3)
import datetime, re, operator
from random import shuffle
from time import gmtime, strftime
import gc

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #get rid of warnings
from os import listdir
from os.path import isfile, join, isdir
import pickle

#import data augmentation methods
from eda_aug import *
from add_aug import *

###################################################
######### loading folders and txt files ###########
###################################################

#loading a pickle file
def load_pickle(file):
	return pickle.load(open(file, 'rb'))

def save_pickle(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

#create an output folder if it does not already exist
def confirm_output_folder(output_folder):
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

#get full image paths
def get_txt_paths(folder):
	txt_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and '.txt' in f]
	if join(folder, '.DS_Store') in txt_paths:
		txt_paths.remove(join(folder, '.DS_Store'))
	txt_paths = sorted(txt_paths)
	return txt_paths

#get subfolders
def get_subfolder_paths(folder):
	subfolder_paths = [join(folder, f) for f in listdir(folder) if (isdir(join(folder, f)) and '.DS_Store' not in f)]
	if join(folder, '.DS_Store') in subfolder_paths:
		subfolder_paths.remove(join(folder, '.DS_Store'))
	subfolder_paths = sorted(subfolder_paths)
	return subfolder_paths

#get all image paths
def get_all_txt_paths(master_folder):

	all_paths = []
	subfolders = get_subfolder_paths(master_folder)
	if len(subfolders) > 1:
		for subfolder in subfolders:
			all_paths += get_txt_paths(subfolder)
	else:
		all_paths = get_txt_paths(master_folder)
	return all_paths

###################################################
################ data processing ##################
###################################################

#get the pickle file for the word2vec so you don't have to load the entire huge file each time
def gen_vocab_dicts(folder, output_pickle_path, huge_word2vec):

	vocab = set()
	text_embeddings = open(huge_word2vec, 'r').readlines()
	word2vec = {}

	#get all the vocab
	all_txt_paths = get_all_txt_paths(folder)
	print(all_txt_paths)

	#loop through each text file
	for txt_path in all_txt_paths:

		# get all the words
		try:
			all_lines = open(txt_path, "r").readlines()
			for line in all_lines:
				words = line[:-1].split(' ')
				for word in words:
					vocab.add(word)
		except:
			print(txt_path, "has an error")
	
	print(len(vocab), "unique words found")

	# load the word embeddings, and only add the word to the dictionary if we need it
	for line in text_embeddings:
		items = line.split(' ')
		word = items[0]
		if word in vocab:
			vec = items[1:]
			word2vec[word] = np.asarray(vec, dtype = 'float32')
	print(len(word2vec), "matches between unique words and word2vec dictionary")
		
	pickle.dump(word2vec, open(output_pickle_path, 'wb'))
	print("dictionaries outputted to", output_pickle_path)

#getting the x and y inputs in numpy array form from the text file
def get_x_y(train_txt, num_classes, word2vec_len, input_size, word2vec, percent_dataset):

	#read in lines
	train_lines = open(train_txt, 'r').readlines()
	shuffle(train_lines)
	train_lines = train_lines[:int(percent_dataset*len(train_lines))]
	num_lines = len(train_lines)

	#initialize x and y matrix
	x_matrix = None
	y_matrix = None

	try:
		x_matrix = np.zeros((num_lines, input_size, word2vec_len))
	except:
		print("Error!", num_lines, input_size, word2vec_len)
	y_matrix = np.zeros((num_lines, num_classes))

	#insert values
	for i, line in enumerate(train_lines):

		parts = line[:-1].split('\t')
		label = int(parts[0])
		sentence = parts[1]	

		#insert x
		words = sentence.split(' ')
		words = words[:x_matrix.shape[1]] #cut off if too long
		for j, word in enumerate(words):
			if word in word2vec:
				x_matrix[i, j, :] = word2vec[word]

		#insert y
		y_matrix[i][label] = 1.0

	return x_matrix, y_matrix

###################################################
############### AEDA #################
###################################################

def gen_punc_aug(train_orig, output_file, num_aug=9):
	lines = open(train_orig, 'r').readlines()
	data_aug = []
	for line in lines:
		line1 = line.split('\t')
		label = line1[0]
		sentence = line1[1]
		for i in range(num_aug):
			sentence_aug = insert_punctuation_marks(sentence)
			line_aug = '\t'.join([label, sentence_aug])
			data_aug.append(line_aug)
		data_aug.append(line)	
	with open(output_file, 'w') as aeda:
		aeda.writelines(data_aug)
	print("finished punc_aug for", train_orig, "to", output_file)

###################################################
############### Adding Numbers #################
###################################################

def gen_num_aug(train_orig, output_file, num_aug=9):
	lines = open(train_orig, 'r').readlines()
	data_aug = []
	for line in lines:
		line1 = line.split('\t')
		label = line1[0]
		sentence = line1[1]
		for i in range(num_aug):
			sentence_aug = insert_numbers(sentence)
			line_aug = '\t'.join([label, sentence_aug])
			data_aug.append(line_aug)
		data_aug.append(line)	
	with open(output_file, 'w') as aeda:
		aeda.writelines(data_aug)
	print("finished num_aug for", train_orig, "to", output_file)


###################################################
############### Adding Alphabets #################
###################################################

def gen_alpha_aug(train_orig, output_file, num_aug=9):
	lines = open(train_orig, 'r').readlines()
	data_aug = []
	for line in lines:
		line1 = line.split('\t')
		label = line1[0]
		sentence = line1[1]
		for i in range(num_aug):
			sentence_aug = insert_alphabets(sentence)
			line_aug = '\t'.join([label, sentence_aug])
			data_aug.append(line_aug)
		data_aug.append(line)	
	with open(output_file, 'w') as aeda:
		aeda.writelines(data_aug)
	print("finished alpha_aug for", train_orig, "to", output_file)



###################################################
############### Hybrid aug #################
###################################################


def gen_hybrid_noise_aug(train_orig, output_file, num_aug=9):
	lines = open(train_orig, 'r').readlines()
	data_aug = []
	for line in lines:
		line1 = line.split('\t')
		label = line1[0]
		sentence = line1[1]
		sentence_aug = noise_3(sentence, 0.3 , num_aug)
		for i in range(len(sentence_aug)):
			line_aug = '\t'.join([label, sentence_aug[i]])
			data_aug.append(line_aug)
	with open(output_file, 'w') as aeda:
		aeda.writelines(data_aug)
	print("finished hybrid noise aug for", train_orig, "to", output_file)

def gen_hybrid_noise_aug_inplace(train_orig, output_file, num_aug=9):
	lines = open(train_orig, 'r').readlines()
	data_aug = []
	for line in lines:
		line1 = line.split('\t')
		label = line1[0]
		sentence = line1[1]
		sentence_aug = noise_3_inplace(sentence, 0.1, num_aug)
		for i in range(len(sentence_aug)):
			line_aug = '\t'.join([label, sentence_aug[i]])
			data_aug.append(line_aug)
	with open(output_file, 'w') as aeda:
		aeda.writelines(data_aug)
	print("finished hybrid noise aug for", train_orig, "to", output_file)




###################################################
############### EDA #################
###################################################

def gen_tsne_aug(train_orig, output_file):

	writer = open(output_file, 'w')
	lines = open(train_orig, 'r').readlines()
	for i, line in enumerate(lines):
		parts = line[:-1].split('\t')
		label = parts[0]
		sentence = parts[1]
		writer.write(line)
		for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
			aug_sentence = eda_4(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=2)[0]
			writer.write(label + "\t" + aug_sentence + '\n')
	writer.close()
	print("finished eda for tsne for", train_orig, "to", output_file)




#generate more data with standard augmentation
def gen_eda_aug(train_orig, output_file, num_aug=9):
	writer = open(output_file, 'w')
	lines = open(train_orig, 'r').readlines()
	for i, line in enumerate(lines):
		parts = line[:-1].split('\t')
		label = parts[0]
		sentence = parts[1]
		aug_sentences = eda_4(sentence, num_aug=num_aug)
		for aug_sentence in aug_sentences:
			writer.write(label + "\t" + aug_sentence + '\n')
	writer.close()
	print("finished eda for", train_orig, "to", output_file)

#generate more data with only synonym replacement (SR)
def gen_sr_aug(train_orig, output_file, alpha_sr, n_aug):
	writer = open(output_file, 'w')
	lines = open(train_orig, 'r').readlines()
	for i, line in enumerate(lines):
		parts = line[:-1].split('\t')
		label = parts[0]
		sentence = parts[1]
		aug_sentences = SR(sentence, alpha_sr=alpha_sr, n_aug=n_aug)
		for aug_sentence in aug_sentences:
			writer.write(label + "\t" + aug_sentence + '\n')
	writer.close()
	print("finished SR for", train_orig, "to", output_file, "with alpha", alpha_sr)

#generate more data with only random insertion (RI)
def gen_ri_aug(train_orig, output_file, alpha_ri, n_aug):
	writer = open(output_file, 'w')
	lines = open(train_orig, 'r').readlines()
	for i, line in enumerate(lines):
		parts = line[:-1].split('\t')
		label = parts[0]
		sentence = parts[1]
		aug_sentences = RI(sentence, alpha_ri=alpha_ri, n_aug=n_aug)
		for aug_sentence in aug_sentences:
			writer.write(label + "\t" + aug_sentence + '\n')
	writer.close()
	print("finished RI for", train_orig, "to", output_file, "with alpha", alpha_ri)

#generate more data with only random swap (RS)
def gen_rs_aug(train_orig, output_file, alpha_rs, n_aug):
	writer = open(output_file, 'w')
	lines = open(train_orig, 'r').readlines()
	for i, line in enumerate(lines):
		parts = line[:-1].split('\t')
		label = parts[0]
		sentence = parts[1]
		aug_sentences = RS(sentence, alpha_rs=alpha_rs, n_aug=n_aug)
		for aug_sentence in aug_sentences:
			writer.write(label + "\t" + aug_sentence + '\n')
	writer.close()
	print("finished RS for", train_orig, "to", output_file, "with alpha", alpha_rs)

#generate more data with only random deletion (RD)
def gen_rd_aug(train_orig, output_file, alpha_rd, n_aug):
	writer = open(output_file, 'w')
	lines = open(train_orig, 'r').readlines()
	for i, line in enumerate(lines):
		parts = line[:-1].split('\t')
		label = parts[0]
		sentence = parts[1]
		aug_sentences = RD(sentence, alpha_rd=alpha_rd, n_aug=n_aug)
		for aug_sentence in aug_sentences:
			writer.write(label + "\t" + aug_sentence + '\n')
	writer.close()
	print("finished RD for", train_orig, "to", output_file, "with alpha", alpha_rd)

###################################################
##################### model #######################
###################################################

#building the model in keras
def build_model(sentence_length, word2vec_len, num_classes):
	model = None
	model = Sequential()
	model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(sentence_length, word2vec_len)))
	model.add(Dropout(0.5))
	model.add(Bidirectional(LSTM(32, return_sequences=False)))
	model.add(Dropout(0.5))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#print(model.summary())
	return model

#building the cnn in keras
def build_cnn(sentence_length, word2vec_len, num_classes):
	model = None
	model = Sequential()
	model.add(layers.Conv1D(128, 5, activation='relu', input_shape=(sentence_length, word2vec_len)))
	model.add(layers.GlobalMaxPooling1D())
	model.add(Dense(20, activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#one hot to categorical
def one_hot_to_categorical(y):
	assert len(y.shape) == 2
	return np.argmax(y, axis=1)

def get_now_str():
	return str(strftime("%Y-%m-%d_%H:%M:%S", gmtime()))



