import numpy as np
class VecsBuilder(object):

	def __init__(self,vecs_path):
		self.map = {}
		self.feature_path = vecs_path

	def get_data(self):
		count = 0
		vec = []
		data = open(self.feature_path,'r+')
		for line in data:
			count+=1
			key = line.split()[0]
			_feature = line.split()[1:]
			vec = list(map(lambda x:float(x),_feature))
			self.map[key] = vec
		self.map['__UNK__'] = [0.0 for k in range(len(vec))]
		return self.map



