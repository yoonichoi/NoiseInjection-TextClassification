from methods import *
from config import *

if __name__ == "__main__":

	#generate the augmented data sets
	for dataset_folder in dataset_folders:
		#pre-existing file locations
		train_orig = dataset_folder + '/train_orig.txt'
		
		for numaug in num_augs:
			# EDA augmentation
			train_eda = f"{dataset_folder}/train_eda_{str(numaug)}.txt"
			gen_eda_aug(train_orig, train_eda, numaug)	# creates train+eda

			# AEDA augmentation
			train_aeda = f"{dataset_folder}/train_aeda_{str(numaug)}.txt"
			gen_punc_aug(train_orig, train_aeda, numaug)	# creates train+aeda

			# NUM augmentation (A4)
			train_num = f"{dataset_folder}/train_num_{str(numaug)}.txt"
			gen_num_aug(train_orig, train_num, numaug)	# creates train+num

			# APLHA augmentation (A4)
			train_alpha = f"{dataset_folder}/train_alpha_{str(numaug)}.txt"
			gen_alpha_aug(train_orig, train_alpha, numaug)	# creates train+alpha

			# HYBRID augmentation (A4)
			train_hybrid = f"{dataset_folder}/train_hybrid_{str(numaug)}.txt"
			gen_hybrid_noise_aug(train_orig, train_hybrid, numaug)	# creates train+hybrid

			train_hybrid = f"{dataset_folder}/train_hybrid_inplace_{str(numaug)}.txt"
			gen_hybrid_noise_aug_inplace(train_orig, train_hybrid, numaug)	# creates train+hybrid+inplace



		#generate the vocab dictionary
		word2vec_pickle = dataset_folder + '/word2vec.pkl' # don't want to load the huge pickle every time, so just save the words that are actually used into a smaller dictionary
		gen_vocab_dicts(dataset_folder, word2vec_pickle, huge_word2vec)
