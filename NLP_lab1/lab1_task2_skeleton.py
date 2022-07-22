# author: ‘Your name"
# student_id: "Your student ID"
import numpy as np
# from math import sqrt, log
# from itertools import chain, product
# from collections import defaultdict


def calculate_bow(corpus):
    """
    Calculate bag of words representations of corpus
    Parameters
    ----------
    corpus: list
        Documents represented as a list of string

    Returns
    ----------
    corpus_bow: list
        List of tuple, each tuple contains raw text and vectorized text
    vocab: list
    """
    # YOUR CODE HERE
    def vectorize(sentence, vocab):
        return [sentence.split().count(i) for i in vocab]

    vectorized_corpus = []
    vocab = sorted(set([token for doc in corpus for token in doc.lower().split()]))
    for i in corpus:
        vectorized_corpus.append((i, vectorize(i, vocab)))
    return vectorized_corpus, vocab
    # return corpus_bow, vocab


def calculate_tfidf_v1(corpus, vocab):
    """
    Parameters
    ----------
    corpus: list of tuple
        Output of calculate_bow()
    vocab: list
        List of words, output of calculate_bow()

    Returns
    corpus_tfidf: list
        List of tuple, each tuple contains raw text and vectorized text
    ----------

    """


    # return a List[(doc,termfreq)]
    def termfreq(doc, term):
        try:
            # YOUR CODE HERE
            return doc.split().count(term)

        except ZeroDivisionError:
            return 0

    # return a List[inversedocfreq]
    def inversedocfreq(term):
        try:
            # YOUR CODE HERE
            nt = 0
            for c in corpus:
                c = c[0]
                if c.__contains__(term):
                    nt += 1
            return nt
        except ZeroDivisionError:
            return 0

    # YOUR CODE HERE
    N = len(corpus)
    corpus_tfidf = []
    for doc in corpus:
        doc = doc[0]
        sigma = 0
        for term in vocab:
            sigma += termfreq(doc, term)
        tfidf = [termfreq(doc, term)/sigma * N/inversedocfreq(term) for term in vocab]
        corpus_tfidf.append((doc, tfidf))
    return corpus_tfidf

def calculate_tfidf_v2(corpus, vocab):
    """
    Parameters
    ----------
    corpus: list of tuple
        Output of calculate_bow()
    vocab: list
        List of words, output of calculate_bow()

    Returns
    corpus_tfidf: list
        List of tuple, each tuple contains raw text and vectorized text
    ----------

    """


    # return a List[(doc,termfreq)]
    def termfreq(doc, term):
        try:
            # YOUR CODE HERE
            return doc.split().count(term)

        except ZeroDivisionError:
            return 0

    # return a List[inversedocfreq]
    def inversedocfreq(term):
        try:
            # YOUR CODE HERE
            nt = 0
            for c in corpus:
                c = c[0]
                if c.__contains__(term):
                    nt += 1
            return nt
        except ZeroDivisionError:
            return 0

    # YOUR CODE HERE
    N = len(corpus)
    corpus_tfidf = []
    for doc in corpus:
        doc = doc[0]
        max = 0
        for term in vocab:
            max = np.maximum(termfreq(doc, term), max)
        tfidf = [(0.5 + 0.5 * termfreq(doc, term)/max) * (np.log(N/(1+inversedocfreq(term))) + 1) for term in vocab]
        corpus_tfidf.append((doc, tfidf))
    return corpus_tfidf



def cosine_sim(u,v):
    """
    Parameters
    ----------
    u: list of number
    v: list of number

    Returns
    ----------
    cosine_score: float
        cosine similarity between u and v
    """
    # YOUR CODE HERE
    cosine_score = np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v))
    return cosine_score


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

    # 2nd problem
    print("Test BOW cosine similarity")
    print_similarity(corpus_bow)

    # 3rd problem
    print("Test tfidf cosine similarity")
    # corpus_tfidf = list(zip(all_sents, calculate_tfidf(corpus_bow, vocab)))
    corpus_tfidf = calculate_tfidf_v1(corpus_bow, vocab)
    print(corpus_tfidf)
    print_similarity(corpus_tfidf)

    corpus_tfidf = calculate_tfidf_v2(corpus_bow, vocab)
    print(corpus_tfidf)
    print_similarity(corpus_tfidf)

if __name__ == "__main__":
    q1()
