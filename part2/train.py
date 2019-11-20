from vocab import VocabBuilder
from data_loader import Loader

vocab_builder = VocabBuilder(vocab_path='./glove/glove.6B.100d.txt')
vocab = vocab_builder.get_data()


max_length = 25
trainer = Loader(max_length,vocab)

for (label,target) in trainer:
	print(label,target)
	