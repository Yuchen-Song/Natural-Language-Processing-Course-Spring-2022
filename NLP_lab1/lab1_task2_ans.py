import numpy as np
from math import sqrt, log
from itertools import chain
from collections import defaultdict


def calculate_bow(corpus):
    def vectorize(sentence, vocab):
        return [sentence.split().count(i) for i in vocab]

    vectorized_corpus = []
    vocab = sorted(set([token for doc in corpus for token in doc.lower().split()]))
    for i in corpus:
        vectorized_corpus.append((i, vectorize(i, vocab)))
    return vectorized_corpus, vocab


def calculate_tfidf(corpus, vocab):
    """
    INPUT:

    corpus = [('this is a foo bar', [1, 1, 0, 1, 1, 0, 0, 1]),
    ('foo bar bar black sheep', [0, 2, 1, 1, 0, 0, 1, 0]),
    ('this is a sentence', [1, 0, 0, 0, 1, 1, 0, 1])]

    vocab = ['a', 'bar', 'black', 'foo', 'is', 'sentence',
    'sheep', 'this']

    OUTPUT:

    [[0.300, 0.300, 0.0, 0.300, 0.300, 0.0, 0.0, 0.300],
    [0.0, 0.600, 0.600, 0.300, 0.0, 0.0, 0.600, 0.0],
    [0.375, 0.0, 0.0, 0.0, 0.375, 0.75, 0.0, 0.375]]

    """
    def termfreq(matrix, doc_id, term):
        try:
            return matrix[doc_id][term] / float(sum(matrix[doc_id].values()))
        except ZeroDivisionError:
            return 0
    # for Q1-4
    def termfreq_1(matrix, doc_id, term):
        try:
            max_num = max(matrix[doc_id].values())
            # return matrix[doc_id][term] / float(sum(matrix[doc_id].values()))
            return 0.5+0.5*(matrix[doc_id][term]) / float(max_num)
        except ZeroDivisionError:
            return 0

    def inversedocfreq(matrix, term):
        try:
            unique_docs = sum([1 for i,_ in enumerate(matrix) if matrix[i][term] > 0])
            return float(len(matrix)) / unique_docs
        except ZeroDivisionError:
            return 0
    
    #for Q1-4
    def inversedocfreq_1(matrix, term):
        try:
            unique_docs = sum([1 for i,_ in enumerate(matrix) if matrix[i][term] > 0])
            return log(len(matrix) / (1 + unique_docs))
        except ZeroDivisionError:
            return 0

    word2id = dict(zip(vocab, range(len(vocab))))
    matrix = [{k:v for k,v in zip(vocab, i[1])} for i in corpus]
    tfidf_mat  =  np.zeros((len(matrix), len(vocab)), dtype=float)
    #for Q1-4
    tfidf_mat_1  =  np.zeros((len(matrix), len(vocab)), dtype=float)

    for doc_id, doc in enumerate(matrix):
        for term in doc:
            term_id = word2id[term]
            tf = termfreq(matrix,doc_id,term) # matrix, 句子id, 词语
            idf = inversedocfreq(matrix, term)
            
            #for Q1-4
            tf_1 = termfreq_1(matrix,doc_id,term) # matrix, 句子id, 词语
            idf_1 = inversedocfreq_1(matrix, term)
            
            tfidf_mat[doc_id][term_id] = tf*idf
            #for Q1-4
            tfidf_mat_1[doc_id][term_id] = tf_1*idf_1

    all_sents = [doc[0] for doc in corpus]
    corpus_tfidf = list(zip(all_sents, tfidf_mat))
    #for Q1-4
    corpus_tfidf_1 = list(zip(all_sents, tfidf_mat_1))
    return corpus_tfidf, corpus_tfidf_1


def cosine_sim(u,v):
    return np.dot(u,v) / (sqrt(np.dot(u,u)) * sqrt(np.dot(v,v)))


def print_similarity(corpus):
    """
    Print pairwise similarities
    """
    for sentx in corpus:
        for senty in corpus:
            print("{:.4f}".format(cosine_sim(sentx[1], senty[1])), end=" ")
        print()
    print()


def q1():
    all_sents = ["this is a foo bar",
                 "foo bar bar black sheep",
                 "this is a sentence"]
    corpus_bow, vocab = calculate_bow(all_sents)
    print(corpus_bow)
    print(vocab)

    print("Test BOW cosine similarity")
    print_similarity(corpus_bow)

    print("Test tfidf cosine similarity")
    # corpus_tfidf = list(zip(all_sents, calculate_tfidf(corpus_bow, vocab)))
    corpus_tfidf, corpus_tfidf_1 = calculate_tfidf(corpus_bow, vocab)
    print('corpus_tfidf:', corpus_tfidf)
    print()
    print('corpus_tfidf_1:', corpus_tfidf_1)
    print()
    print_similarity(corpus_tfidf)
    print()
    print_similarity(corpus_tfidf_1)

if __name__ == "__main__":
    q1()
