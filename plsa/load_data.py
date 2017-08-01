# -*- coding: utf-8 -*-
__author__ = 'PC-LiNing'

from gensim import corpora
from collections import defaultdict
import codecs

stoplist = set('的 和 与 中 为 及 对 在 了 例'.split())


def load_texts(file):
    f = codecs.open(file,encoding='utf-8')
    texts = []
    for line in f.readlines():
        line = line.strip('\n').strip()
        words = line.split()
        # remove stop word and single word
        texts.append([word for word in words if word not in stoplist and len(word) > 1])
    return texts


def load_corpus(data_file):
    texts = load_texts(data_file)
    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus,dictionary