__author__ = 'PC-LiNing'

import numpy as np
import random
import math
import datetime
import codecs


class LDAModel(object):
    def __init__(self, data, dic, n_topics=100, alpha=0.1, beta=0.01, iter_times=2000):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.iter_times = iter_times
        self.dictionary = dic
        self.n_words = len(self.dictionary)
        self.doc_words = data
        self.n_doc = len(data)
        #
        self.nw = np.zeros(shape=(self.n_topics, self.n_words), dtype=np.int32)
        self.nwsum = np.zeros(shape=(self.n_topics,), dtype=np.int32)
        self.nd = np.zeros(shape=(self.n_doc, self.n_topics), dtype=np.int32)
        self.ndsum = np.zeros(shape=(self.n_doc,), dtype=np.int32)
        self.z = np.zeros(shape=(self.n_doc, self.n_words), dtype=np.int32)
        self.theta = np.zeros(shape=(self.n_doc, self.n_topics), dtype=np.float32)
        self.phi = np.zeros(shape=(self.n_topics, self.n_words), dtype=np.float32)

    def init_topics(self):
        # words number of each doc
        for i in range(self.n_doc):
            self.ndsum[i] = len(self.doc_words[i])
            for j in range(self.ndsum[i]):
                topic = random.randint(0,self.n_topics - 1)
                self.z[i][j] = topic
                self.nw[topic][self.doc_words[i][j]] += 1
                self.nd[i][topic] += 1
                self.nwsum[topic] += 1

    def gibbs_sample(self,i,j):
        topic = self.z[i][j]
        word = self.doc_words[i][j]
        self.nw[topic][word] -= 1
        self.nd[i][topic] -= 1
        self.nwsum[topic] -= 1
        self.ndsum[i] -= 1
        Vbeta = self.n_words * self.beta
        Kalpha = self.n_topics * self.alpha
        p = np.zeros(shape=(self.n_topics, ), dtype=np.float32)
        p = (self.nw.T[word] + self.beta)/(self.nwsum + Vbeta) *\
            (self.nd[i] + self.alpha)/(self.ndsum[i] + Kalpha)
        for topic in range(1, self.n_topics):
            p[topic] += p[topic - 1]
        u = random.uniform(0, p[self.n_topics-1])
        sample_topic = -1
        for topic in range(self.n_topics):
            if p[topic] > u:
                sample_topic = topic
                break

        self.nw[sample_topic][word] += 1
        self.nd[i][sample_topic] += 1
        self.nwsum[sample_topic] += 1
        self.ndsum[i] += 1
        return sample_topic

    def train(self):
        self.init_topics()
        for it in range(self.iter_times):
            for i in range(self.n_doc):
                for j in range(self.ndsum[i]):
                    self.z[i][j] = self.gibbs_sample(i,j)
            self.compute_parameters()
            ppl = self.compute_perplexity()
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("{}: iter {}, ppl {:g}".format(time_str, it, ppl))
        # save
        print('save model...')
        np.savetxt('theta.txt', self.theta, fmt='%.6f')
        np.savetxt('phi.txt', self.phi, fmt='%.6f')
        self.print_tassign()
        self.print_words_map()
        print("train end.")

    def compute_parameters(self):
        Vbeta = self.n_words * self.beta
        Kalpha = self.n_topics * self.alpha
        # theta
        for i in range(self.n_doc):
            self.theta[i] = (self.nd[i] + self.alpha)/(self.ndsum[i] + Kalpha)
        # phi
        for i in range(self.n_topics):
            self.phi[i] = (self.nw[i] + self.beta)/(self.nwsum[i] + Vbeta)

    def compute_p_dw(self,i,j):
        p_dw = 0.0
        word = self.doc_words[i][j]
        for topic in range(self.n_topics):
            p_dw += self.theta[i][topic] * self.phi[topic][word]
        return math.log(p_dw)

    def compute_perplexity(self):
        p_sum = 0.0
        for i in range(self.n_doc):
            for j in range(self.ndsum[i]):
                p_sum += self.compute_p_dw(i,j)
        total_words = sum(self.ndsum.tolist())
        ppl = math.exp(-(p_sum/total_words))
        return ppl

    def print_tassign(self):
        f = codecs.open('tassign.txt', 'a', encoding='utf-8')
        for i in range(self.n_doc):
            line = ''
            for j in range(self.ndsum[i]):
                word = self.doc_words[i][j]
                line += str(word)+":"+str(self.z[i][j])+"   "
            f.write(line+'\n')
        f.close()

    def print_words_map(self):
        f = codecs.open('word2id.txt', 'a', encoding='utf-8')
        for key,value in self.dictionary.items():
            line = str(key)+' '+value
            f.write(line+'\n')
        f.close()