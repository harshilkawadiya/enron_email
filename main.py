import pandas as pd
import re

df = pd.read_csv('data/emails.csv',nrows=100)
splitter = re.compile('[^.!?].*[.!?]')

total_count = 0

def get_message(colln,idx):
	_msg = colln.iloc[idx]['message']
	msgs = _msg.split(':')[-1].split('/n')
	msgs_join = ' '.join(msgs)
	msg_sents = splitter.findall(msgs_join)
	return msg_sents, len(msgs)

sentences = []
for k in range(len(df)):
	sents_k,_ = get_message(df,k)
	sentences.extend(map(str.strip, sents_k))

results = '\n'.join(sentences)
save_results = open('./outputs/save_results.txt','a+')
print('Results Saved for ' + str(len(sentences)) + ' lines.')
save_results.write(results)
save_results.close()
