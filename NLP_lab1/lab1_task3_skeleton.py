from nltk.corpus import brown,gutenberg
from collections import Counter

import matplotlib.pyplot as plt
import math

def q1():
    puncs = set((',', '.', '--', '-', '!', '?', ':', ';', '``', "''", '(', ')', '[', ']'))
    word_list = (x.lower() for x in (gutenberg.words() and brown.words())  if x not in puncs)
    token_counts = Counter(word_list)
    print(token_counts.most_common(20))

    # 前两百
    freq = []
    for word in token_counts:
        freq.append(token_counts[word])
    freq.sort(reverse=True)

    freq1 = freq[:200]
    x = [i for i in range(1,201)]

    # .plot()作图需指定x轴和y轴的数据
    plt.plot([math.log(i) for i in x], [math.log(y) for y in freq1])
    plt.show()
    plt.savefig('zipf.png')

if __name__ == "__main__":
    q1()
