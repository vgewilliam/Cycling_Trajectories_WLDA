import logging.config
import configparser
from collections import OrderedDict
import codecs
import os
from collections import Counter

import numpy as np

path = os.getcwd()

logging.config.fileConfig("logging.conf")

logger = logging.getLogger()


conf = configparser.ConfigParser()
conf.read("setting.conf", encoding='utf-8')
trainfile = os.path.join(path,os.path.normpath(conf.get("filepath", "trainfile")))
wordidmapfile = os.path.join(path,os.path.normpath(conf.get("filepath","wordidmapfile")))
weightfile = os.path.join(path,os.path.normpath(conf.get("filepath","weightfile")))

class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0

class DataPreProcessing(object):

    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        self.docs = []
        self.word2id = OrderedDict()
        self.corpus = []
        self.term_freqs = []
        self.vacab = []
        self.doc_lengths = []
        self.weights = []
        self.corpusIndex=[]

    def cachewordidmap(self):
        with codecs.open(wordidmapfile, 'w','utf-8') as f:
            for word,id in self.word2id.items():
                f.write(word +"\t"+str(id)+"\n")

    def preprocessing(object):
        logger.info(u'Load text data......')
        with codecs.open(trainfile, 'r', 'utf-8') as f:
            docs = f.readlines()  # read data
        logger.debug(u"Load is complete, ready to generate the dictionary object....")
        dpre = DataPreProcessing()
        items_idx = 0
        for line in docs:
            if line != "":
                tmp = line.strip().split(',')
                # print(tmp)
                # generate a document object
                doc = Document()  
                for item in tmp:  # Number the words in the document, with each word having a unique number
                    dpre.corpus.append(item)
                    if dpre.word2id.__contains__(item):
                        doc.words.append(dpre.word2id[item])
                    else:
                        dpre.word2id[item] = items_idx
                        doc.words.append(items_idx)
                        items_idx += 1
                doc.length = len(tmp)
                dpre.doc_lengths.append(doc.length)
                dpre.docs.append(doc)
                dpre.corpusIndex.append(doc.words)

            else:
                pass
        dpre.docs_count = len(dpre.docs)  # dpre.docs
        dpre.words_count = len(dpre.word2id)  # word2id
        dpre.vacab = list(Counter(dpre.corpus).keys()) 
        dpre.term_freqs = list(Counter(dpre.corpus).values()) 
        logger.info(u"Total %s of documents" % dpre.docs_count)
        dpre.cachewordidmap()
        logger.info(u"The relationship between words and numbers has been saved to %s" % wordidmapfile)


 #####Load weights data 
        logger.info(u'Load weights data......')
        with codecs.open(weightfile, 'r', 'utf-8') as f:
            weights = f.readlines() 

        for line in weights:

            if line != "":
                tmp_wight = line.strip().split(',')
                tmp_wight = map(float,tmp_wight)

                #tmp_wight = np.array(tmp_wight).astype(np.float)
                dpre.weights.append(list(tmp_wight))

        return dpre

