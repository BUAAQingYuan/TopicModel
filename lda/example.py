__author__ = 'PC-LiNing'

from lda import LDA
from lda import load_data

corpus, dic = load_data.load_corpus('docs.txt')
lda = LDA.LDAModel(corpus, dic=dic, n_topics=5, iter_times=600)
lda.train()
