# author: Yuchen Song
# student_id: 201830360498

import numpy
import numpy as np
import scipy.sparse
from scipy import sparse
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
from nltk.tokenize import word_tokenize

stop_words = []
with open('./stop_words.txt', 'r') as f:
    for word in f.readlines():
        stop_words.append(word)
stop_words = set(stop_words + list(string.punctuation))

# Q1-2
def tokenize(text):
    '''
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
    e.g. 
    Input: 'It is a nice day. I am happy.'
    Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
    '''
    # tokens = []
    # # YOUR CODE HERE
    # tokens = word_tokenize(text)
    return word_tokenize(text)

def remove_stopwords(word):
    if word not in stop_words:
        return word
    else:
        return ''

# Q1-3
def get_bagofwords(data, vocab_dict):
    '''
    :param data: a list of words, type: list
    :param vocab_dict: a dict from words to indices, type: dict
    return a word (sparse) matrix, type: scipy.sparse.csr_matrix
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html
    '''
    data_matrix = sparse.lil_matrix((len(data), len(vocab_dict)))
    # YOUR CODE HERE
    for i, doc in enumerate(data):
        for word in doc:
            word_idx = vocab_dict.get(word, -1)
            if word_idx != -1:
                data_matrix[i, word_idx] += 1
    data_matrix = data_matrix.tocsr()
    return data_matrix

# Q1-1 Note that, here you need to use Q1-2 and Q1-3.
def read_data(file_name, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    df['words'] = df['text'].apply(tokenize)
    # df['words'].astype('str')
    # df['words'] = df['words'].apply(remove_stopwords)
    # df['words'].apply(lambda word: word not in stop_words)
    # df['words'] = df['words'].apply(lambda stop_remove: [word.lower() for word in stop_remove.split() if word not in stop_words])
    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = dict(zip(vocab, range(len(vocab))))

    data_matrix = get_bagofwords(df['words'], vocab_dict)

    return df['id'], df['label'], data_matrix, vocab

# Q2-1
def normalize(P):
    """
    normalize P to make sure the sum of the first dimension equals to 1
    e.g.
    Input: [1,2,1,2,4]
    Output: [0.1,0.2,0.1,0.2,0.4] (without laplace smoothing) or [0.1333,0.2,0.1333,0.2,0.3333] (with laplace smoothing)

    Numpy Sum https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    """
    # YOUR CODE HERE
    # sum = np.sum(P)
    # sum = sum+len(P)
    # for i in range(len(P)):
    #     P[i] = (P[i]+1)/sum
    # return P
    return (P+1)/(sum(P)+len(P))

# Q2-2, Q2-3
def train_NB(data_label, data_matrix):
    '''
    :param data_label: [N], type: list
    :param data_matrix: [N(document_number) * V(Vocabulary number)], type:  scipy.sparse.csr_matrix
    return the P(y) (an K array), P(x|y) (a V*K matrix)
    '''
    N = data_matrix.shape[0]
    K = max(data_label)  # labels begin with 1
    # YOUR CODE HERE
    # V = data_matrix.shape[1]
    # label_matrix = np.zeros([N, K])
    # data_label = data_label.tolist()
    # data_matrix = data_matrix.todense()
    # for i in range(len(data_label)):
    #     label_matrix[i, data_label[i]-1] = 1
    # Bayes_matrix = np.matmul(numpy.transpose(data_matrix), label_matrix)  # V*K
    # P_y = np.zeros([K,])  # (# of yi)/(# of all)
    # # label_sum = np.sum(data_label)  # K labels altogether, but each doc can be predicted to a single label
    # for i in range(K):
    #     yi_sum = 0
    #     for d in data_label:
    #         if d == i+1:
    #             yi_sum += 1
    #     P_y[i] = (yi_sum+1) / (N+K)
    # P_xy = np.zeros([V, K])  #
    # col_sum = np.sum(Bayes_matrix, axis=0)
    # for v in range(V):
    #     for k in range(K):
    #         P_xy[v, k] = (Bayes_matrix[v, k]+1)/(col_sum[0, k]+V)
    # return P_y, P_xy
    data_delta = np.zeros([N,K])
    for i, l in enumerate(data_label):
        data_delta[i, l-1] = 1
    P_y = normalize(np.sum(data_delta, axis=0, keepdims=False))
    P_xy = normalize(data_matrix.transpose().dot(data_delta))
    return P_y, P_xy
# Q3
def predict_NB(data_matrix, P_y, P_xy):
    '''
    :param data_matrix: [N(document_number), V(Vocabulary number)], type:  scipy.sparse.csr_matrix
    :param P_y: [K(class number)], type: np.ndarray
    :param P_xy: [V, K], type: np.ndarray
    return data_pre (a N array)
    '''
    # compute the label probabilities using the P(y) and P(x|y) according to the naive Bayes algorithm
    # YOUR CODE HERE
    # N = data_matrix.shape[0]
    # data_matrix = data_matrix.todense()
    # data_pre = np.zeros([N,])
    # for n in range(N):
    #     label_prob = np.log(P_y)
    #     # temp = np.matmul(data_matrix[n], np.log(P_xy))
    #     # temp = np.array(temp)
    #     # temp.reshape([5,])
    #     label_prob += np.array(np.matmul(data_matrix[n], np.log(P_xy)))[0]
    #     label_prob = np.exp(label_prob)
    #     label = np.argmax(label_prob)
    #     data_pre[n] = label
    # # get labels for every document by choosing the maximum probability
    # # YOUR CODE HERE
    # return data_pre
    log_P_y = np.expand_dims(np.log(P_y), axis=0)
    log_P_xy = np.log(P_xy)
    log_P_dy = data_matrix.dot(log_P_xy)
    log_P = log_P_y + log_P_dy
    return np.argmax(log_P, axis=1) + 1

def evaluate(y_true, y_pre): 
    assert len(y_true) == len(y_pre)
    acc = accuracy_score(y_true, y_pre)
    # average='macro', calculate the average precision, recall, and F1 score of all categories.
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pre, average="macro")
    return acc, precision, recall, f1

def evaluate_each_category(y_true, y_pre): 
    assert len(y_true) == len(y_pre)
    acc = accuracy_score(y_true, y_pre)
    # average='macro', calculate the precision, recall, and F1 score of each category.
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pre, average=None)
    return acc, precision, recall, f1

if __name__ == '__main__':
    # Read train.csv
    train_id_list, train_data_label, train_data_matrix, vocab = read_data("data/train.csv")
    # Divide train.csv into train set(80%) and validation set (20%).
    X_train, X_validation, y_train, y_validation = train_test_split(train_data_matrix, train_data_label, test_size=0.2)
    print("Vocabulary Size:", len(vocab))
    print("Training Set Size:", len(y_train))
    print("Validation Set Size:", len(y_validation))
    # Read test.csv
    test_id_list, _, test_data_matrix, _ = read_data("data/test.csv", vocab)
    print("Test Set Size:", len(test_id_list))

    # training
    P_y, P_xy = train_NB(y_train, X_train)
    train_data_pre = predict_NB(X_train, P_y, P_xy)
    acc, precision, recall, f1 = evaluate(y_train, train_data_pre)
    print("Evalution in train set: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (acc, precision, recall, f1))

    # validation
    validation_data_pre = predict_NB(X_validation, P_y, P_xy)
    acc, precision, recall, f1 = evaluate(y_validation, validation_data_pre)
    print("Evalution in Validation set: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (acc, precision, recall, f1))

    acc, precision, recall, f1 = evaluate_each_category(y_validation, validation_data_pre)
    print("Evalution in test set in each category: \nAccuracy:",acc,"\nPrecision:",precision, "\nRecall:",recall, '\nF1:',f1)

    # Predict the label of each documents in the test set
    test_data_pre = predict_NB(test_data_matrix, P_y, P_xy)

    sub_df = pd.DataFrame()
    sub_df["id"] = test_id_list
    sub_df["pred"] = test_data_pre
    sub_df.to_csv("submission.csv", index=False)
    print("Predict Results are saved, please check the submission.csv")
