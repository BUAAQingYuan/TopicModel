# -*- coding: utf-8 -*-
__author__ = 'PC-LiNing'

from plsa import load_data
from plsa import pLSA

# corpus = [(word_id,word_count),...]
# dic ={word_id:word}
corpus,dic = load_data.load_corpus('data.txt')
plsa = pLSA.PLSA(corpus,dic,topics=3)
plsa.train()


