__author__ = 'PC-LiNing'

import numpy
import math

def random_sample(x,y):
    data = numpy.random.random_sample((x,y))
    for i in range(x):
        sum = 0.0
        for j in range(y):
            sum += data[i][j]
        for j in range(y):
            data[i][j] = data[i][j] / sum
    return data


def normalize(vec):
    s = sum(vec)
    assert(abs(s) != 0.0) # the sum must not be 0
    for i in range(len(vec)):
        assert(vec[i] >= 0) # element must be >= 0
        vec[i] = vec[i] * 1.0 / s
    return vec


class PLSA:
    def __init__(self,corpus,dictionary,topics=2,max_iter=100,delta=0.001):
        self.corpus = corpus
        self.topics = topics
        self.dictionary = dictionary
        self.iter = max_iter
        self.delta = delta
        self.word_num = len(dictionary)
        self.doc_num = len(corpus)
        self.doc_word_matrix = numpy.zeros(shape=(self.doc_num,self.word_num),dtype=numpy.int32)
        self.topic_word = random_sample(self.topics, self.word_num)
        self.doc_topic = random_sample(self.doc_num, self.topics)
        self.likelihood = 0.0
        self.z_dw = numpy.zeros(shape=(self.doc_num,self.word_num,self.topics),dtype=numpy.float32)

    def compute_likelihood(self):
        likelihood = 0.0
        for doc in range(self.doc_num):
            for word in range(self.word_num):
                p_wd = 0.0
                for topic in range(self.topics):
                    p_wd = p_wd + self.doc_topic[doc][topic]*self.topic_word[topic][word]
                # the value must > 0
                # assert(p_wd > 0.0)
                if p_wd > 0.0:
                    likelihood = likelihood + self.doc_word_matrix[doc][word] * math.log(p_wd)
        return  likelihood

    def train(self):
        # build doc_word_matrix
        for doc in range(self.doc_num):
            for pair in self.corpus[doc]:
                self.doc_word_matrix[doc][pair[0]] = pair[1]
        # compute likelihood
        prelikelihood = self.compute_likelihood()
        print("Init likelihood: "+str(prelikelihood))
        # iteration
        for iteration in range(self.iter):
            print("Iteration #" + str(iteration + 1) + "...")
            print("E step...")
            # E-step
            for d_index in range(self.doc_num):
                for w_index in range(self.word_num):
                    prob = self.doc_topic[d_index, :] * self.topic_word[:, w_index]
                    if sum(prob) == 0.0:
                        prob = numpy.zeros(shape=(self.topics,),dtype=numpy.float32)
                    else:
                        prob = normalize(prob)
                    self.z_dw[d_index][w_index] = prob

            print("M step...")
            # M-step
            # update P(w | z)
            for z in range(self.topics):
                for w_index in range(self.word_num):
                    s = 0.0
                    for d_index in range(self.doc_num):
                        count = self.doc_word_matrix[d_index][w_index]
                        s = s + count * self.z_dw[d_index, w_index, z]
                    self.topic_word[z][w_index] = s
                self.topic_word[z] = normalize(self.topic_word[z])

            # update P(z | d)
            for d_index in range(self.doc_num):
                for z in range(self.topics):
                    s = 0.0
                    for w_index in range(self.word_num):
                        count = self.doc_word_matrix[d_index][w_index]
                        s = s + count * self.z_dw[d_index, w_index, z]
                    self.doc_topic[d_index][z] = s
                self.doc_topic[d_index] = normalize(self.doc_topic[d_index])

            # compute likelihood
            likelihood = self.compute_likelihood()
            delta = likelihood - prelikelihood
            print("likelihood:"+str(likelihood)+","+"delta:"+str(delta))
            prelikelihood = likelihood
            if delta < self.delta:
                break
        print("Finished likelihood: "+str(prelikelihood))
        print("Train end.")

    def inference_likelihood(self,new_doc_num,new_doc_topic,new_doc_word_matrix):
        likelihood = 0.0
        for doc in range(new_doc_num):
            for word in range(self.word_num):
                p_wd = 0.0
                for topic in range(self.topics):
                    p_wd = p_wd + new_doc_topic[doc][topic]*self.topic_word[topic][word]
                # the value must > 0
                # assert(p_wd > 0.0)
                if p_wd > 0.0:
                    likelihood = likelihood + new_doc_word_matrix[doc][word] * math.log(p_wd)
        return  likelihood

    def inference(self,new_corpus):
        # build doc_word_matrix
        new_doc_num = len(new_corpus)
        new_doc_word_matrix = numpy.zeros(shape=(new_doc_num,self.word_num),dtype=numpy.int32)
        for doc in range(new_doc_num):
            for pair in new_corpus[doc]:
                # ignore new words
                if pair[0] < self.word_num:
                    new_doc_word_matrix[doc][pair[0]] = pair[1]

        new_doc_topic = random_sample(new_doc_num, self.topics)
        new_z_dw = numpy.zeros(shape=(new_doc_num,self.word_num,self.topics),dtype=numpy.float32)
        # compute likelihood
        prelikelihood = self.inference_likelihood(new_doc_num,new_doc_topic,new_doc_word_matrix)
        print("Init likelihood: "+str(prelikelihood))
        # iteration
        for iteration in range(self.iter):
            print("Iteration #" + str(iteration + 1) + "...")
            print("E step...")
            # E-step
            for d_index in range(new_doc_num):
                for w_index in range(self.word_num):
                    prob = new_doc_topic[d_index, :] * self.topic_word[:, w_index]
                    if sum(prob) == 0.0:
                        prob = numpy.zeros(shape=(self.topics,),dtype=numpy.float32)
                    else:
                        prob = normalize(prob)
                    new_z_dw[d_index][w_index] = prob

            print("M step...")
            # M-step
            # do not update P(w | z)
            # update P(z | d)
            for d_index in range(new_doc_num):
                for z in range(self.topics):
                    s = 0.0
                    for w_index in range(self.word_num):
                        count = new_doc_word_matrix[d_index][w_index]
                        s = s + count * new_z_dw[d_index, w_index, z]
                    new_doc_topic[d_index][z] = s
                new_doc_topic[d_index] = normalize(new_doc_topic[d_index])

            # compute likelihood
            likelihood = self.inference_likelihood(new_doc_num,new_doc_topic,new_doc_word_matrix)
            delta = likelihood - prelikelihood
            print("likelihood:"+str(likelihood)+","+"delta:"+str(delta))
            prelikelihood = likelihood
            if delta < self.delta:
                break
        print("Finished likelihood: "+str(prelikelihood))
        print("Compute end.")
        return new_doc_topic