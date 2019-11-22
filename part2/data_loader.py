import torch
import numpy as np
import nltk
class Loader(object):
    def __init__(self, max_length,vocab,subset='train'):
        self.samples = []
        self.labels = []
        self.num_samples = 0
        self.vocab = vocab
        self.max_length = max_length
        if subset=='train':
            self.add_data('./actions.txt',1)
            self.add_data('./non_actions.txt',0)
        else:
            self.add_data('./actions_val.txt',1)
            self.add_data('./non_actions_val.txt',0)            
        self._shuffle_indices()

    def add_data(self, path_name,label=0): 
        f = open(path_name,'r+')
        cnt = 0
        for items in f:
            cnt+=1
            self.samples.append(items.strip())
            self.labels.append(int(label))
        self.num_samples = len(self.samples)
        
    def _shuffle_indices(self):
        self.indices = np.random.permutation(self.num_samples)

    def get_feature(self,sentence):
        vocab = self.vocab
        s = sentence.lower()
        tokens = nltk.tokenize.word_tokenize(s)
        sent = []
        # print(tokens)
        for word in tokens:
            if word in vocab:
                sent.append(vocab[word])
            else:
                sent.append(vocab['__UNK__'])
        if len(sent)>self.max_length:
        	sent = sent[:self.max_length]
        else:
        	rem =  self.max_length-len(sent)
        	for lap in range(rem):
        		sent.append(vocab['__UNK__'])
        return sent


    def __getitem__(self,index):
        idx = self.indices[index]
        _sentence = self.samples[idx]
        sentence = torch.Tensor(self.get_feature(_sentence))
        # print(index,idx)
        target = torch.Tensor([self.labels[idx]]).long()
        # print(target)
        return sentence,target
