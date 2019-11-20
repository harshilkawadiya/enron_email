import torch
import numpy as np
import nltk
class Loader(object):
    def __init__(self, max_length,word2idx):
        self.samples = []
        self.labels = []
        self.num_samples = 0
        self.word2idx = word2idx
        self.max_length = max_length
        self.add_data('./actions.txt',1)
        self.add_data('./non_actions.txt',0)
        self._shuffle_indices()

    def add_data(self, path_name,label=0): 
        f = open(path_name,'r+')
        for items in f:
        	self.samples.append(items.strip())
        	self.labels.append(label)
        self.num_samples = len(self.samples)

    def _shuffle_indices(self):
        self.indices = np.random.permutation(self.num_samples)

    def get_feature(self,sentence):
        word2idx = self.word2idx
        s = sentence.lower()
        tokens = nltk.tokenize.word_tokenize(s)
        sent = []
        print(tokens)
        for word in tokens:
            if word in word2idx:
                sent.append(word2idx[word])
            else:
                sent.append(word2idx['__UNK__'])
        if len(sent)>self.max_length:
        	sent = sent[:self.max_length]
        else:
        	rem =  self.max_length-len(sent)
        	for lap in range(rem):
        		sent.append(word2idx['__UNK__'])
        return sent


    def __getitem__(self,index):
        idx = self.indices[index]
        _sentence = self.samples[idx]
        sentence = torch.Tensor(self.get_feature(_sentence))
        target = torch.Tensor(self.labels[idx])
        return sentence,target



