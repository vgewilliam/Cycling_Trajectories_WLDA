#-*- coding:utf-8 -*-
import logging.config
import configparser
import pandas as pd
import numpy as np
import random
import codecs
import os
from collections import OrderedDict
from dataprocess import DataPreProcessing
import pyLDAvis


path = os.getcwd()

logging.config.fileConfig("logging.conf")

logger = logging.getLogger()

conf = configparser.ConfigParser()
conf.read("setting.conf",encoding='utf-8')

#filepath
thetafile = os.path.join(path,os.path.normpath(conf.get("filepath","thetafile")))
phifile = os.path.join(path,os.path.normpath(conf.get("filepath","phifile")))
paramfile = os.path.join(path,os.path.normpath(conf.get("filepath","paramfile")))
topNfile = os.path.join(path,os.path.normpath(conf.get("filepath","topNfile")))
tassginfile = os.path.join(path,os.path.normpath(conf.get("filepath","tassginfile")))

#Initial model parameters
K = int(conf.get("model_args","K"))
alpha = float(conf.get("model_args","alpha"))
beta = float(conf.get("model_args","beta"))
iter_times = int(conf.get("model_args","iter_times"))
top_words_num = int(conf.get("model_args","top_words_num"))


class LDAModel(object):
    
    def __init__(self,dpre):
        # self.phi = None
        # self.theta = None
        self.dpre = dpre

        """
        Model Parameters
        Topic number K, iter_times,top_words_num,α(alpha), β(beta)
        """

        self.K = K
        self.beta = beta #Hyperparameter for topic-word probability distribution
        self.alpha = alpha  #Hyperparameter for track-topic probability distribution
        self.iter_times = iter_times
        self.top_words_num = top_words_num

        """
        trainfile, wordfile
        wordidmapfile, word and id file
        thetafile, document-topic distribution
        phifile, topic-word distribution 
        topNfile, topN words in the document
        tassginfile, result file
        paramfile, parameters file
        """

        self.thetafile = thetafile
        self.phifile = phifile
        self.topNfile = topNfile
        self.tassginfile = tassginfile
        self.paramfile = paramfile

        # Initialize the Parameters
        self.p = np.zeros(self.K)
        self.nw = np.zeros((self.dpre.words_count,self.K),dtype="int")
        self.nwsum = np.zeros(self.K,dtype="int")
        self.nd = np.zeros((self.dpre.docs_count,self.K),dtype="int")
        self.ndsum = np.zeros(dpre.docs_count,dtype="int")
        self.Z = np.array([ [0 for y in range(dpre.docs[x].length)] for x in range(dpre.docs_count)])

        self.nw_weight = np.zeros((self.dpre.words_count, self.K), dtype="double") #
        self.nwsum_weight = np.zeros(self.K, dtype="double")
        self.nd_weight = np.zeros((self.dpre.docs_count, self.K), dtype="double")
        self.ndsum_weight = np.zeros(dpre.docs_count, dtype="double")
        self.W = None

        random.seed(10)
        self.topicinit() #Randomly assign a topic to each word



    def topicinit(self):
        self.W = self.dpre.weights  # Initialize the weights

        for x in range(len(self.Z)):  #len(self.Z) is the number of documents

            self.ndsum[x] = self.dpre.docs[x].length
            self.ndsum_weight[x] = sum(self.W[x])
            for y in range(self.dpre.docs[x].length):
                topic = random.randint(0,self.K-1)
                self.Z[x][y] = topic

                # Assign weights

                self.nw_weight[self.dpre.docs[x].words[y]][topic] += self.W[x][y]
                self.nd_weight[x][topic] += self.W[x][y]
                self.nwsum_weight[topic] += self.W[x][y]

        self.theta = np.array([ [0.0 for y in range(self.K)] for x in range(self.dpre.docs_count)])  #e.g., 𝜃1 : {topic1 : 80%, topic2 : 10%…}
        self.phi = np.array([ [ 0.0 for y in range(self.dpre.words_count) ] for x in range(self.K)]) #e.g., φ1 : {word1 : 80%, word2 : 10%…}


    def sampling(self,i,j):

        topic = self.Z[i][j] #A word with assigned topic

        word = self.dpre.docs[i].words[j]

        #When words are assigned topics, the corresponding TOPIC statistic minus the weights
        self.nw_weight[word][topic] -= self.W[i][j]
        self.nd_weight[i][topic] -= self.W[i][j]
        self.nwsum_weight[topic] -= self.W[i][j]
        self.ndsum_weight[i] -= self.W[i][j]

        Vbeta = self.dpre.words_count * self.beta
        Kalpha = self.K * self.alpha

        self.p = (self.nw_weight[word] + self.beta) / (self.nwsum_weight + Vbeta) * \
                 (self.nd_weight[i] + self.alpha) / (self.ndsum_weight[i] + Kalpha)

        for k in range(1,self.K):
            self.p[k] += self.p[k-1]

        u = random.uniform(0,self.p[self.K-1])

        for topic in range(self.K):
            if self.p[topic]>u:  #Random sampling updates the value of the topic
                break

        self.nw_weight[word][topic] += self.W[i][j]
        self.nd_weight[i][topic] += self.W[i][j]
        self.nwsum_weight[topic] += self.W[i][j]
        self.ndsum_weight[i] += self.W[i][j]

        return topic

    # iteration
    def est(self):

        for x in range(self.iter_times): #iteration times
            for i in range(self.dpre.docs_count): #number of documents
                for j in range(self.dpre.docs[i].length): #number of words
                    topic = self.sampling(i,j) #sampling
                    self.Z[i][j] = topic

        logger.info(u"Iteration completed")
        logger.debug(u"Calculating document-topic distribution")
        self._theta()
        logger.debug(u"Calculating topic-word distribution")
        self._phi()
        logger.debug(u"Save models")
        self.save()


    def _theta(self):
        for i in range(self.dpre.docs_count):
            self.theta[i] = (self.nd_weight[i] + self.alpha) / (self.ndsum_weight[i] + self.K * self.alpha)



    def _phi(self):
        for i in range(self.K):
            self.phi[i] = (self.nw_weight.T[i] + self.beta) / (self.nwsum_weight[i] + self.dpre.words_count * self.beta)

    # Save the results
    def save(self):
        logger.info(u"document-topic distribution has been saved to %s" % self.thetafile)
        with codecs.open(self.thetafile,'w') as f:
            for x in range(self.dpre.docs_count):
                for y in range(self.K):
                    f.write(str(self.theta[x][y]) + '\t')
                f.write('\n')

        logger.info(u"topic-word distribution has been saved to %s" % self.phifile)
        with codecs.open(self.phifile,'w') as f:
            for x in range(self.K):
                for y in range(self.dpre.words_count):
                    f.write(str(self.phi[x][y]) + '\t')
                f.write('\n')

        #Save parameters
        logger.info(u"parameters have been saved to %s" % self.paramfile)
        with codecs.open(self.paramfile,'w','utf-8') as f:
            f.write('K=' + str(self.K) + '\n')
            f.write('alpha=' + str(self.alpha) + '\n')
            f.write('beta=' + str(self.beta) + '\n')
            f.write(u'Number of iterations  iter_times=' + str(self.iter_times) + '\n')
            f.write(u'Top words  top_words_num=' + str(self.top_words_num) + '\n')

        logger.info(u"topN words have been saved to %s" % self.topNfile)
        with codecs.open(self.topNfile,'w','utf-8') as f:
            self.top_words_num = min(self.top_words_num,self.dpre.words_count)
            for x in range(self.K):
                twords = []
                twords = [(n,self.phi[x][n]) for n in range(self.dpre.words_count)]
                twords.sort(key = lambda i:i[1], reverse= True)
                for y in range(self.top_words_num):
                    word = OrderedDict({value:key for key, value in self.dpre.word2id.items()})[twords[y][0]]
                    f.write('\t'*2+ word +'\t' + str(twords[y][1])+ '\n')

        #Save the results of the document's word assignment of the topic on final exit
        logger.info(u"document-word-topic assignment results have been saved to %s" % self.tassginfile)
        with codecs.open(self.tassginfile,'w') as f:
            for x in range(self.dpre.docs_count):
                for y in range(self.dpre.docs[x].length):
                    f.write(str(self.dpre.docs[x].words[y])+':'+str(self.Z[x][y])+ '\t')
                f.write('\n')
        logger.info(u"Model training completed")





def run():
    dpre = DataPreProcessing.preprocessing(object)
    lda = LDAModel(dpre)
    lda.est()



    #Write topic results to Excel
    data = pd.read_excel( r'./data/Topic_output/Topic_output.xlsx',engine='openpyxl')
    topics = lda.theta
    topic = []
    for t in topics:
        topic.append("Topic #" + str(list(t).index(np.max(t))+1))
    data['Topic number with the highest probability'] = topic
    data['Corresponding probability for each topic'] = list(topics)
    data.to_excel("./data/Topic_output/Topic_output.xlsx", index=False)
    logger.info(u"Topic_output saved!")

    # Visualize topics

    pic = pyLDAvis.prepare(lda.phi, lda.theta, dpre.doc_lengths, dpre.vacab,  dpre.term_freqs)
    pyLDAvis.save_html(pic, 'lda_pass' + str(len(lda.phi)) + '.html')


if __name__ == '__main__':
    run()
    