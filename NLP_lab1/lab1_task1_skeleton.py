# author: ‘Yuchen Song'
# student_id: '201830360498'

def q1():
    print('q1: {:}'.format(''))
    # 1. Print the number of word tokens
    # YOUR CODE
    from nltk.corpus import gutenberg as gb
    word_list = gb.words('austen-sense.txt')
    print(len((word_list)))

    # 2. Print the number of word types
    # YOUR CODE
    word_type = set(word_list)
    print(len(word_type))

    # 3. Print all tokens in the first sentence
    # YOUR CODE
    sent_list = gb.sents('austen-emma.txt')
    print(sent_list[0])

def q2():
    print('q2: {:}'.format(''))
    # 1. Print the top 10 most common words in the 'romance' category.
    # Your Code
    import nltk
    from nltk.corpus import brown as br
    romance_word_list = br.words(categories='romance')
    romance_freqdist = nltk.FreqDist([ w.lower() for w in romance_word_list ])
    for r in romance_freqdist.most_common(10):
        print(r[0]+" ", end="")
    # print(r for r in romance_freqdist.most_common(10))
    print()
    # 2.  Print the word frequency of the following words:['ring','activities','love','sports','church'] 
    # in the 'romance'  and 'hobbies' categories respectively. 
    # Your Code
    word_list = ['ring','activities','love','sports','church']
    for w in word_list:
        print(w+" in romance: ",end="")
        print(romance_freqdist.freq(w)*romance_freqdist.N())
    hobbies_word_list = br.words(categories='hobbies')
    hobbies_freqdist = nltk.FreqDist([ w.lower() for w in hobbies_word_list ])
    for w in word_list:
        print(w+" in hobbies: ",end="")
        print(hobbies_freqdist.freq(w)*hobbies_freqdist.N())

def q3():
    print('q3: {:}'.format(''))
    # 1. Print all synonymous words (lemmas) of the word 'dictionary'
    # Your Code
    from nltk.corpus import wordnet as wn
    for synset in wn.synsets('dictionary'):
        print(' '.join(synset.lemma_names()))
    print()
    # 2. Print all hyponyms of the word 'dictionary'
    # Your Code
    for synset in wn.synsets('dictionary'):
        hyponyms_synsets = synset.hyponyms()
        for hy in hyponyms_synsets:
            print(' '.join(hy.lemma_names()))

    # 3. Use one of the predefined similarity measures to score the similarity of
    # the following pairs of synsets and rank the pairs in order of decreasing similarity.
    # (right_whale.n.01,novel.n.01)
    # (right_whale.n.01,minke_whale.n.01)
    # (right_whale.n.01,tortoise.n.01)
    # synset_names = ['right_whale.n.01', 'novel.n.01', 'minke_whale.n.01', 'tortoise.n.01']
    # Your Code
    print()
    synset_names = ['right_whale.n.01', 'novel.n.01', 'minke_whale.n.01', 'tortoise.n.01']
    synsets = [wn.synset(sn) for sn in synset_names]
    sims = [-synsets[0].lch_similarity(sn) for sn in synsets ]
    import numpy as np
    index = np.argsort(sims)
    for i in index:
        print("("+synset_names[0]+" "+synset_names[i]+"): ", end="")
        print(0-sims[i])


if __name__ == '__main__':
    q1()

    print()

    q2()

    print()
    q3()
