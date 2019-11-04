from tqdm import tqdm
import pandas as pd
import re

df = pd.read_csv('data/emails.csv',nrows=5000)

splitter = re.compile('[^.!?]*[\w]\s*[.!?]')
bow = {}
sentences = []
total_count = 0
actionable_sentences = 0;
LIMIT = 1000 #for reducing ram memory consumption

def is_actionable(query):
	query = query.lower()
	words = ['?','can you ','would you','should ','please ', 'will you ', 'could you ', 'what are ', 'what is ','who ',' how do ']
	for word in words:
		if word in query:
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

for k in tqdm(range(len(df))):
	sents_k = get_message(df,k)
	total_count+=len(sents_k)
	for s in sents_k:
		if is_actionable(s.strip()):
			sentences.append(s.strip())
	if len(sentences) > LIMIT:
		save_results = open('save_results.txt','a+')
		actionable_sentences+=len(sentences)
		print('Results Saved for ' + str(len(sentences)) + ' lines.')
		results = '\n'.join(sentences)
		save_results.write(results)
		save_results.close()
		sentences = []

save_results = open('./outputs/save_results.txt','a+')
actionable_sentences+=len(sentences)
print('Results Saved for ' + str(len(sentences)) + ' lines.')
results = '\n'.join(sentences)
save_results.write(results)
save_results.close()


print('Number of Actionable Sentences : ', actionable_sentences)
print('Total Sentences : ', total_count)