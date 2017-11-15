# coding:utf-8

import sys
import gensim
import sklearn
import numpy as np



from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence



if __name__ == '__main__':
    #训练model
    model = Word2Vec(LineSentence("train_set.txt"), size=100, window=2, min_count=0, workers=4)
    model.wv.save_word2vec_format('address_word2vec_model')
    print('word2vec model get!')
    Model=gensim.models.KeyedVectors.load_word2vec_format('address_word2vec_model')
    print(Model.wv['霄云路'])
    print(Model.wv.similarity('霄云路','霄云路'))


