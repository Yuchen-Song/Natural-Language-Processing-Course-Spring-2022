import os 
from gensim.models import KeyedVectors
from gensim.downloader import base_dir


def load_data():
    path = os.path.join(base_dir, 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-200.gz')
    model = KeyedVectors.load_word2vec_format(path)
    return model
