# AEDA: An Easier Data Augmentation Technique for Text classification
# Akbar Karimi, Leonardo Rossi, Andrea Prati

import random
from sklearn.utils import shuffle

random.seed(0)

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
DATASETS = ['cr', 'sst2', 'subj', 'pc', 'trec']
NUM_AUGS = [1, 2, 4, 8]
ADD_RATIO = 0.3
LENGTH = 5

####################################
###########		AEDA	############
####################################

# Insert punction words into a given sentence with the given ratio "add_ratio"
def insert_punctuation_marks(sentence, add_ratio=ADD_RATIO):
	words = sentence.split(' ')
	new_line = []
	q = random.randint(1, int(add_ratio * len(words) + 1))
	qs = random.sample(range(0, len(words)), q)

	for j, word in enumerate(words):
		if j in qs:
			new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
			new_line.append(word)
		else:
			new_line.append(word)
	new_line = ' '.join(new_line)
	return new_line

####################################
######## Adding Numbers	############
####################################

def generate_random_number(length=LENGTH):
	# given length, generate a random number of that length
	number = ''
	for i in range(length):
		number += str(random.randint(0, 9))
	return number

# Insert random number words of specified length into a given sentence with the given ratio "add_ratio"
def insert_numbers(sentence, add_ratio=ADD_RATIO):
	words = sentence.split(' ')
	new_line = []
	q = random.randint(1, int(add_ratio * len(words) + 1))	# number of numbers to add
	qs = random.sample(range(0, len(words)), q)	# indices to add numbers to

	for j, word in enumerate(words):
		if j in qs:
			new_line.append(generate_random_number())
			new_line.append(word)
		else:
			new_line.append(word)
	new_line = ' '.join(new_line)
	return new_line


####################################
######## Adding Alphabets	########
####################################

def generate_random_alphabets(length=LENGTH):
	# given length, generate a random alphabet of that length
	alphabet = ''
	for i in range(length):
		alphabet += chr(random.randint(97, 122))
	return alphabet

# Insert random number words of specified length into a given sentence with the given ratio "add_ratio"
def insert_alphabets(sentence, add_ratio=ADD_RATIO):
	words = sentence.split(' ')
	new_line = []
	q = random.randint(1, int(add_ratio * len(words) + 1))	# number of alphabets to add
	qs = random.sample(range(0, len(words)), q)	# indices to add alphabets to

	for j, word in enumerate(words):
		if j in qs:
			new_line.append(generate_random_alphabets())
			new_line.append(word)
		else:
			new_line.append(word)
	new_line = ' '.join(new_line)
	return new_line

########################################################################
# hybrid noise injection data augmentation function
########################################################################


def noise_3(sentence, num_aug=9):

	augmented_sentences = []
	num_new_per_technique = int(num_aug / 3)

	# punc
	for _ in range(num_new_per_technique):
		augmented_sentence = insert_punctuation_marks(sentence)
		augmented_sentences.append(augmented_sentence)

	# char
	for _ in range(num_new_per_technique):
		augmented_sentence = insert_alphabets(sentence)
		augmented_sentences.append(augmented_sentence)

	# num
	for _ in range(num_new_per_technique):
		augmented_sentence = insert_numbers(sentence)
		augmented_sentences.append(augmented_sentence)

	shuffle(augmented_sentences)

	# trim so that we have the desired number of augmented sentences
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	# append the original sentence
	augmented_sentences.append(sentence)

	return augmented_sentences




def main(dataset, aug_type):
	outputfile = None
	for aug in NUM_AUGS:
		data_aug = []
		with open(dataset + '/train.txt', 'r') as train_orig:
			for line in train_orig:
				line1 = line.split('\t')
				label = line1[0]
				sentence = line1[1]
				for i in range(aug):
					if aug_type == 'punctuation':
						outputfile = 'train_punc_aug'
						sentence_aug = insert_punctuation_marks(sentence)
					elif aug_type == 'numbers':
						outputfile = 'train_num_aug'
						sentence_aug = insert_numbers(sentence, random.randint(1,10))
					else:
						outputfile = 'train_alpha_aug'
						sentence_aug = insert_alphabets(sentence, random.randint(1,10))
					line_aug = '\t'.join([label, sentence_aug])
					data_aug.append(line_aug)
				data_aug.append(line)

		with open(f"{dataset}/{outputfile}_{aug}.txt", 'w') as train_orig_plus_augs:
			train_orig_plus_augs.writelines(data_aug)


if __name__ == "__main__":
	aug_type = 'alphabets'	# 'punctuation', 'numbers', 'alphabets'
	for dataset in DATASETS:
		main(dataset, aug_type)
