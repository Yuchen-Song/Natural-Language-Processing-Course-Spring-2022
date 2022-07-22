# author: 'Your name'
# student_id: 'Your student ID'

def q1():
    print('q1: {:}'.format(''))
    # 1. Print the number of word tokens
    # YOUR CODE
    from nltk.corpus import gutenberg as gb
    #if you want to print all file ids in gutenberg archive
    #print(gb.fileids())
    file_id = 'austen-sense.txt'
    word_list = gb.words(file_id)
    #print(word_list[0])
    #print(word_list[:10])
    print(len(word_list))

    # 2. Print the number of word types
    # YOUR CODE
    print(len( set( [ w.lower() for w in word_list ]) ))
    
    # 3. Print all tokens in the first sentence
    # YOUR CODE
    sent_list = gb.sents(file_id)
    print(' '.join(sent_list[0]))

    # if you want to tokenize a string
    raw = 'i have a book.'
    from nltk import word_tokenize as wt 
    word_list = wt(raw)
    #print(word_list)

def q2():
    print('q2: {:}'.format(''))

    # 1. Print the top 10 most common words in the 'romance' category.
    # Your Code
    import nltk
    from nltk.corpus import brown
    #print(brown.categories())

    romance_word_list = brown.words(categories='romance')
    romance_freqdist = nltk.FreqDist([ w.lower() for w in romance_word_list ])
    for word, freq in romance_freqdist.most_common(10):
        print(word, ':', freq)
    # 2.  Print the word frequency of the following words:['ring','activities','love','sports','church'] 
    # in the 'romance'  and 'hobbies' categories respectively. 
    # Your Code
    queries = ['ring','activities','love','sports','church']
    
    print('romance freq distribution')
    print()
    for q in queries:
        print(q,':',romance_freqdist[q])
    print()

    print('hobbies freq distribution')
    print()
    hobbies_word_list = brown.words(categories='hobbies')
    hobbies_freqdist = nltk.FreqDist([ w.lower() for w in hobbies_word_list ])
    for q in queries:
        print(q,':',hobbies_freqdist[q])

    # An easy way to do (2)
    categories=['romance','hobbies'] # categories = brown.categories()
    cfd = nltk.ConditionalFreqDist( [(c, w.lower()) for c in categories for w in brown.words(categories=c)] )
    cfd.tabulate(conditions=categories,samples=queries)
    #cfd.plot(conditions=categories,samples=queries)

def q3():
    print('q3: {:}'.format(''))
    # 1. Print all synonymous words (lemmas) of the word 'dictionary'
    # 同义词
    # Your Code
    from nltk.corpus import wordnet as wn
    for synset in wn.synsets('dictionary'):
        print(' '.join(synset.lemma_names()))
    
    # 2. Print all hyponyms of the word 'dictionary'
    # Your Code
    hyponym_synsets = []
    for synset in wn.synsets('dictionary'):
        hyponym_synsets += synset.hyponyms()
    #print(type(hyponym_synsets[0]))
    hyponyms = []
    for hy in hyponym_synsets:
        hyponyms += hy.lemma_names()
    print(' '.join(hyponyms))
    # 3. Use one of the predefined similarity measures to score the similarity of
    # the following pairs of synsets and rank the pairs in order of decreasing similarity.
    # (right_whale.n.01,novel.n.01)
    # (right_whale.n.01,minke_whale.n.01)
    # (right_whale.n.01,tortoise.n.01)
    synset_names = ['right_whale.n.01', 'novel.n.01', 'minke_whale.n.01', 'tortoise.n.01']
    synsets = [wn.synset(sn) for sn in synset_names]
    sims = [synsets[0].lch_similarity(sn) for sn in synsets ]
    print(sims)
    syn_sims = [(synset_names[i],sims[i]) for i in range(len(synsets))]
    def getkey(tup):
        return tup[1]
    print( sorted(syn_sims,key=getkey,reverse=True) )
    print( sorted(syn_sims,key=lambda tup:tup[1],reverse=True) )
    
    syn_sims.sort(key=getkey,reverse=True)
    for i in range(1,len(syn_sims)):
        print(syn_sims[i][0],':',syn_sims[i][1])
    '''
    lch_similarity
    wup_similarity
    #paper: Measuring the Semantic Similarity of Texts
    from nltk.corpus import wordnet_ic
    brown_ic = wordnet_ic.ic('ic-brown.dat')
   
    right.lin_similarity(minke,brown_ic)
    right.res_similarity(minke,brown_ic)
    '''

if __name__ == '__main__':
    
    q1()

    print()
    
    q2()

    print()
    
    q3()

