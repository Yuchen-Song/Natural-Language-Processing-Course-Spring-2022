from nltk.corpus import brown,gutenberg
from collections import Counter

import matplotlib.pyplot as plt
import math

def q1():
    puncs = set((',', '.', '--', '-', '!', '?', ':', ';', '``', "''", '(', ')', '[', ']'))
    word_list_brown = (x.lower() for x in brown.words() if x not in puncs)
    word_list_gutenberg = (x.lower() for x in gutenberg.words() if x not in puncs)

    token_counts_brown = Counter(word_list_brown)
    token_counts_gutenberg = Counter(word_list_gutenberg)

    # print(token_counts_brown.most_common(20))
    # print(token_counts_gutenberg.most_common(20))

    token_counts = token_counts_brown + token_counts_gutenberg

    # rank200
    rank_200 = token_counts.most_common(200)
    rank_num = []
    for i in range(200):
        rank_num.append(rank_200[i][1])

    x = [i for i in range(1,201)]

    # .plot()作图需指定x轴和y轴的数据
    plt.plot([math.log(i) for i in x], [math.log(y) for y in rank_num])
    plt.show()
    plt.savefig('zipf.png')

if __name__ == "__main__":
    q1()
