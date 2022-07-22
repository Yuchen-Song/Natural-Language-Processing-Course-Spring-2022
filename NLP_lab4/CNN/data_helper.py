import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd
import os

MAX_SEQ_LEN = 500
MIN_FREQ = 5
def clean_tokenize(string):
	"""
	Tokenization/string cleaning for datasets.
	:param string: a doc with multiple sentences, type: str
	return a word list, type: list
	e.g.
	Input: 'It is a nice day. I am happy.'
	['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) # remove non alphabet, non digit, non punctuation
	string = re.sub(r"\'s", " \'s", string) # add a space before \'s
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string) #remove multiple spaces
	return string.strip().lower().split() #strip "\n", convert to lower case, split string

def label_vec(ind,num_classes):
	"""
	Convert label to a vector
	:param ind: label, type: str
	return a vector, type: list
	e.g.
	Input: '5'
	[0,0,0,0,1]
	"""
	ind = int(ind)
	y = [0] * num_classes
	y[ind-1] = 1
	return y

def load_data_and_labels(file_name):
	"""
	https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
	"""
	"""
	Loads data from files, splits the data into words and generates labels.
	Returns split sentences, labels, number of classes
	"""
	# Load data from files
	df = pd.read_csv(file_name)
	df['words'] = df['text'].apply(clean_tokenize) # document -> a list of words
	
	num_classes = max ( map( int, df['label'] ) ) # maximum value of labels
	df['y'] = df['label'].apply(label_vec, num_classes = num_classes) # label -> a label vector, i.e., 5->[0,0,0,0,1]

	return df['words'], df['y']


def pad_sentences(sentences, padding_word="<PAD/>",sequence_length=MAX_SEQ_LEN):
	"""
	Pads all sentences to the same length. The length is pre-defined.
	Returns padded sentences.
	"""
	padded_sentences = []
	for i in range(len(sentences)):
		sentence = sentences[i]
		if len(sentence) < sequence_length:
			num_padding = sequence_length - len(sentence)
			new_sentence = sentence + [padding_word] * num_padding
			padded_sentences.append(new_sentence)
		else:
			padded_sentences.append(sentence[:MAX_SEQ_LEN])
	return padded_sentences


def build_vocab(sentences):
	"""
	Builds a vocabulary mapping from word to index based on the sentences.
	Remove low frequency words.
	Returns vocabulary.
	"""
	vocabulary = {}
	cnter = Counter()
	for sent in sentences:
		cnter.update(sent)
	for k in list(cnter):
		if cnter[k] < MIN_FREQ:
			del cnter[k]
	for word in list(cnter.keys()):	
		vocabulary[word] = len(vocabulary)
	vocabulary['<unk>'] = len(vocabulary)
	return vocabulary

def load_embedding_weights(vocabulary):
	embeddings_index = {}
	with open ('./data/glove_50d.txt','r') as fin:
		for line in fin:
			values = line.strip().split(' ')
			word = values[0]
			emb = np.asarray(values[1:], dtype='float32')
			emb_dim = len(emb)
			embeddings_index[word] = emb
	
	embedding_matrix = np.zeros((len(vocabulary), emb_dim))
	for word in vocabulary.keys():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None: # words not found in embedding index will be all-zeros.	
			embedding_matrix[vocabulary[word]] = embedding_vector
		
	return embedding_matrix

def build_input_data(sentences, labels, vocabulary):
	"""
	Maps sentences and labels to vectors based on a vocabulary.
	"""
	unknown_token_id = vocabulary['<unk>']
	vocab = vocabulary.keys()
	x = np.array( [ [vocabulary[word] if word in vocab else unknown_token_id for word in sentence] for sentence in sentences ] )
	y = np.array(list(labels))
	return x, y


def load_data():
	"""
	Loads and preprocessed data for the dataset.
	Returns input text ids, labels, vocabulary.
	"""
	train_file = './data/train.csv'
	test_file = './data/test.csv'

	# Load and preprocess training data
	train_sentences, train_labels = load_data_and_labels(train_file)
	train_sentences_padded = pad_sentences(train_sentences)
	# Build vocabulary from training data, but not from test data.
	vocabulary = build_vocab(train_sentences_padded)
	weights = load_embedding_weights(vocabulary)
	x_train, y_train = build_input_data(train_sentences_padded, train_labels, vocabulary)
	
	# Load and preprocess test data
	test_sentences, test_labels = load_data_and_labels(test_file)
	test_sentences_padded = pad_sentences(test_sentences)
	x_test, y_test = build_input_data(test_sentences_padded, test_labels, vocabulary)

	return x_train, y_train, x_test, y_test, weights