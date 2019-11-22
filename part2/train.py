import torch
from vec_builder import VecsBuilder
from data_loader import Loader
from models import Classifier
import torch.nn as nn
import os
vecs_builder = VecsBuilder(vecs_path='./glove/glove.6B.100d.txt')
vecs = vecs_builder.get_data()

def train(model,train_loader):
	model.train()
	optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
	train_loss_epoch = 0
	steps = 0
	num_correct = 0
	for (label,target) in train_loader:
		if torch.cuda.is_available():
			label =  label.cuda()
			target = target.cuda()
		optim.zero_grad()
		pred =  model(label)
		loss = loss_ce(pred,target)
		num_correct += (torch.max(pred, 1)[1].view(target.size()).data == target.data).sum()
		loss.backward()
		optim.step()
		steps+=1
		train_loss_epoch += loss.item()
	return train_loss_epoch/train_loader.num_samples,num_correct.item()/train_loader.num_samples

def eval(model,val_loader):
	model.eval()
	optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
	val_loss_epoch = 0
	steps = 0
	num_correct = 0
	for (label,target) in val_loader:
		if torch.cuda.is_available():
			label =  label.cuda()
			target = target.cuda()

		optim.zero_grad()
		pred =  model(label)
		loss = loss_ce(pred,target)
		num_correct += (torch.max(pred, 1)[1].view(target.size()).data == target.data).sum()
		optim.step()
		steps+=1

		val_loss_epoch += loss.item()
	return val_loss_epoch/val_loader.num_samples,num_correct.item()/val_loader.num_samples

max_length = 25
num_classes = 2
hidden_dim = 200
embed_dim = 100
num_epochs = 100
model = Classifier(embed_dim, hidden_dim,num_classes)
if torch.cuda.is_available():
	model.cuda()
train_loader = Loader(max_length,vecs,'train')
val_loader = Loader(max_length,vecs,'val')
loss_ce = nn.CrossEntropyLoss()
best_acc = 0
for epoch in range(num_epochs):
	train_loss,train_acc = train(model,train_loader)
	val_loss,val_acc = eval(model,val_loader)
	print('Epoch : ',epoch)
	print('Train Loss : ',train_loss)
	print('Train Acc : ',train_acc)
	print('Validation Loss : ',val_loss)
	print('Validation Acc : ',val_acc)
	if val_acc>best_acc:
		best_acc = val_acc
		best_model = 'best.pkl'
		torch.save(model.state_dict(),best_model)
		print('Best Model Saved with Valdn Accuracy :',val_acc)
	print('--------------------------------------------------\n\n')
