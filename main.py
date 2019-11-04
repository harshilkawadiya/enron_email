from tqdm import tqdm
import pandas as pd
import re
import nltk

df = pd.read_csv('data/emails.csv',nrows=1000)

splitter = re.compile('[^.!?]*[\w]\s*[.!?]')
bow = {}
sentences = []
total_count = 0
actionable_sentences = 0;
LIMIT = 1000 #for reducing ram memory consumption

def is_actionable(query):
	query = query.lower()
	words = ['?','can you ','would ','should ','please ', 'will ', 'could ', 'what ', 'what ','who ',' how ','where ','when ']
	for word in words:
		if word in query:
			if query.count(' ')>2:
				return True
	return False

# Get message content out from email body.
def get_message(colln,idx):
	_msg = colln.iloc[idx]['message']
	msgs = _msg.split(':')[-1].split('/n')
	msgs_join = ' '.join(msgs)
	msg_sents = splitter.findall(msgs_join)
	return msg_sents

# Saving actionable sentences in this list and then to a file.
sentences = []
tp = 0
fp = 0
fn = 0

pos_dict = {'MD':'1','VB':'2','VBG':'2','VBN':'2','VBP':'2','VBZ':'2','WP':'3'}
patterns = ['142','12','32','342','242']

def evaluate(idf):
	if idf[0]=='2':
		return True

	for p in patterns:
		if p in idf:
			return True
	return False

for k in tqdm(range(len(df))):
	try:
		sents_k = get_message(df,k)
		total_count+=len(sents_k)
		for s in sents_k:
			# Evaluation by findin patterns in nltk
			tags = nltk.pos_tag(s.strip().split())
			pos = [item[1] for item in tags]
			key_idf = ''
			for p in pos:
				if p in pos_dict.keys():
					key_idf+=pos_dict[p]
				else:
					key_idf+='4'
			gt_sent = evaluate(key_idf)
			if is_actionable(s.strip()):
				sentences.append(s.strip())
				if gt_sent:
					tp+=1
				else:
					fp+=1
			else:
				if gt_sent:
					fn+=1
					print(tags)
		if len(sentences) > LIMIT:
			save_results = open('save_results.txt','a+')
			actionable_sentences+=len(sentences)
			print('Results Saved for ' + str(len(sentences)) + ' lines.')
			results = '\n'.join(sentences)
			save_results.write(results)
			save_results.close()
			sentences = []
	except:
		print('Message Not Retrieved')


print('\n\n\n\n============== Summary ====================\n')
print('Precision :', (tp)/(tp+fp))
print('Recall :', (tp)/(tp+fn))

save_results = open('./outputs/save_results.txt','a+')
actionable_sentences+=len(sentences)
print('Results Saved for ' + str(len(sentences)) + ' lines.')
results = '\n'.join(sentences)
save_results.write(results)
save_results.close()


print('Number of Actionable Sentences : ', actionable_sentences)
print('Total Sentences : ', total_count)