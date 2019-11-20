import numpy as np
class VocabBuilder(object):

	def __init__(self,vocab_path):
		self.map = {}
		self.feature_path = vocab_path

	def get_data(self):
		count = 0
		vec = []
		data = open(self.feature_path,'r+')
		for line in data:
			count+=1
			if count>1000:
				break
			key = line.split()[0]
			_feature = line.split()[1:]
			vec = list(map(lambda x:float(x),_feature))
			self.map[key] = vec
		self.map['__UNK__'] = [0.0 for k in range(len(vec))]
		return self.map



