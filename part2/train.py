from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from vec_builder import VecsBuilder
from data_loader import Loader
from models import Classifier
import torch.nn as nn
import os

loss_ce = nn.CrossEntropyLoss()
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
		loss = loss_ce(pred,target.view(-1))
		num_correct += (torch.max(pred, 1)[1].view(target.size()).data == target.data).sum()
		loss.backward()
		optim.step()
		steps+=1
		train_loss_epoch += loss.item()
	return train_loss_epoch/len(train_loader),num_correct.item()

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
		loss = loss_ce(pred,target.view(-1))
		num_correct += (torch.max(pred, 1)[1].view(target.size()).data == target.data).sum()
		optim.step()
		steps+=1

		val_loss_epoch += loss.item()
	return val_loss_epoch/len(val_loader),num_correct.item()


def main(args):
	vecs_builder = VecsBuilder(vecs_path='./glove/glove.6B.300d.txt')
	vecs = vecs_builder.get_data()

	train_dataset = Loader(args.max_length,vecs,'train')
	train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = 5)
	val_dataset = Loader(args.max_length,vecs,'val')
	val_loader = DataLoader(val_dataset, batch_size = args.batch_size)
	model = Classifier(args.embed_dim, args.hidden_dim,args.num_classes,args.num_hidden_layers)

	if torch.cuda.is_available():
		print('Cuda Functioning..')
		model.cuda()

	best_acc = 0
	automated_log = open('models/automated_log.txt','w+')
	automated_log.write('Epochs'+'\t'+'Train-Loss'+'\t'+'Train-Accuracy'+'\t'+'Validation Loss'+'\t'+'Validation Accuracy\n')

	for epoch in tqdm(range(args.num_epochs)):
		train_loss,train_acc = train(model,train_loader)
		val_loss,val_acc = eval(model,val_loader)
		train_acc = train_acc/train_dataset.num_samples
		val_acc = val_acc/val_dataset.num_samples
		# print('Epoch : ',epoch)
		# print('Train Loss : ',train_loss)
		# print('Train Acc : ',train_acc)
		# print('Validation Loss : ',val_loss)
		# print('Validation Acc : ',val_acc)
		automated_log.write(str(epoch)+'\t'+str(train_loss)+'\t'+str(train_acc)+'\t'+str(val_loss)+'\t'+str(val_acc)+'\n')
		if epoch%10==0:
			model_name = 'models/model_'+str(epoch)+'.pkl'
			torch.save(model.state_dict(),model_name)
		if val_acc>best_acc:
			best_acc = val_acc
			best_model = 'best.pkl'
			torch.save(model.state_dict(),best_model)
			f = open('models/best.txt','w+')
			report = 'Epoch : '+str(epoch)+'\t Validation Accuracy : '+str(best_acc)
			f.write(report)
			f.close()
			print('Best Model Saved with Valdn Accuracy :',val_acc)
	automated_log.close()
parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int,default=25,
                    help='Maximum Sentence Length')
parser.add_argument('--num_classes', type=int,default=2,
                    help='Number of Classes')
parser.add_argument('--hidden_dim', type=int,default=200,
                    help='Hidden Dimension')
parser.add_argument('-embed_dim', type=int,default=300,
                    help='Embedding Dimension')
parser.add_argument('--num_epochs', type=int,default=100,
                    help='Number of Epochs')
parser.add_argument('--num_hidden_layers', type=int,default=2,
                    help='Number of Hidden Layers')
parser.add_argument('--batch_size', type=int,default=2,
                    help='Batch Size')
args = parser.parse_args()
main(args)	