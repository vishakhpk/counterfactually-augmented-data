import pickle
import pandas as pd
import torch
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


##path = 'sentiment/combined/paired/{}_paired.tsv'
#path = 'NLI/revised_combined/{}.tsv'
path = '../counterfactually-augmented-data/NLI/revised_combined/{}.tsv'
train_path = path.format('train')
val_path = path.format('dev')
test_path = path.format('test')

train_df = pd.read_table(train_path)
val_df = pd.read_table(val_path)
test_df = pd.read_table(test_path)

# build text/label fields on factual and counterfactual data
all_premises = train_df["sentence1"].tolist()
all_hypotheses = train_df["sentence2"].tolist()
all_train_texts = all_premises + all_hypotheses
#all_train_texts = train_df["Text"].tolist()
#label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
#text_field = Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
#text_field.build_vocab(all_train_texts, min_freq=3)
#text_field.build_vocab(all_train_texts, min_freq=3)

# setup tokenizer
vocab_size = 3000#VOCAB_SIZE
tokenizer = Tokenizer(num_words=vocab_size, oov_token=True)
tokenizer.fit_on_texts(all_train_texts)

# split into factual and counterfactual
# we'll do this now for simplicity, at the cost of some copy-pasting below
train_IDs = len(train_df)/2#train_df['batch_id'].values
val_IDs = len(val_df)/2#val_df['batch_id'].values
test_IDs = len(test_df)/2#test_df['batch_id'].values

indices = list(range(len(train_df)))
#print(indices)
factual_indices, counterfactual_indices = indices[::2], indices[1::2]#train_test_split(indices, test_size=0.5, stratify=train_IDs)
#print(factual_indices)
temp = factual_indices + counterfactual_indices
temp2 = counterfactual_indices + factual_indices
factual_indices = temp
counterfactual_indices = temp2

cf_train_df = train_df.iloc[counterfactual_indices]
train_df = train_df.iloc[factual_indices]
print(train_df.head())
print(cf_train_df.head())
"""
plt.hist(train_df['Sentiment'])
plt.title('Train Factuals')
plt.figure()
plt.hist(cf_train_df['Sentiment'])
plt.title('Train Counterfactuals')
plt.show()
"""
indices = list(range(len(val_df)))
factual_indices, counterfactual_indices = indices[::2], indices[1::2]#train_test_split(indices, test_size=0.5, stratify=val_IDs)
cf_val_df = val_df.iloc[counterfactual_indices]
val_df = val_df.iloc[factual_indices]

indices = list(range(len(test_df)))
factual_indices, counterfactual_indices = indices[::2], indices[1::2]#train_test_split(indices, test_size=0.5, stratify=test_IDs)
cf_test_df = test_df.iloc[counterfactual_indices]
test_df = test_df.iloc[factual_indices]


# load text, labels, and IDs
train_IDs = [i for i in range(len(train_df))]#train_df['batch_id'].tolist()
val_IDs = [i for i in range(len(val_df))]#val_df['batch_id'].tolist()
test_IDs = [i for i in range(len(test_df))]#test_df['batch_id'].tolist()

cf_train_IDs = [i for i in range(len(cf_train_df))]#cf_train_df['batch_id'].tolist()
cf_val_IDs = [i for i in range(len(cf_val_df))]#cf_val_df['batch_id'].tolist()
cf_test_IDs = [i for i in range(len(cf_test_df))]#cf_test_df['batch_id'].tolist()

train_prs = train_df['sentence1'].tolist()
val_prs = val_df['sentence1'].tolist()
test_prs = test_df['sentence1'].tolist()

train_hps = train_df['sentence2'].tolist()
val_hps = val_df['sentence2'].tolist()
test_hps = test_df['sentence2'].tolist()

cf_train_prs = cf_train_df['sentence1'].tolist()
cf_val_prs = cf_val_df['sentence1'].tolist()
cf_test_prs = cf_test_df['sentence1'].tolist()

cf_train_hps = cf_train_df['sentence2'].tolist()
cf_val_hps = cf_val_df['sentence2'].tolist()
cf_test_hps = cf_test_df['sentence2'].tolist()

#cf_train_texts = cf_train_df['Text'].tolist()
#cf_val_texts = cf_val_df['Text'].tolist()
#cf_test_texts = cf_test_df['Text'].tolist()
label_map = {'entailment':2, 'neutral':1, 'contradiction':0}
print(train_df['gold_label'].head())
train_labels = train_df['gold_label'].replace(label_map).tolist()#(train_df['Sentiment'] == 'Positive').tolist()
val_labels = val_df['gold_label'].replace(label_map).tolist()#(train_df['Sentiment'] == 'Positive').tolist()
test_labels = test_df['gold_label'].replace(label_map).tolist()#(train_df['Sentiment'] == 'Positive').tolist()
print(train_labels[:5])
print(set(train_labels))
#val_labels = (val_df['Sentiment'] == 'Positive').tolist()
#test_labels = (test_df['Sentiment'] == 'Positive').tolist()
#
# tokenize, convert to sequences, and pad
# note: using the same padding for factual/counterfactual dataset pairs 
#       not sure on this for val/test
train_pr_sequences = tokenizer.texts_to_sequences(train_prs)
cf_train_pr_sequences = tokenizer.texts_to_sequences(cf_train_prs)
train_hp_sequences = tokenizer.texts_to_sequences(train_hps)
cf_train_hp_sequences = tokenizer.texts_to_sequences(cf_train_hps)
train_padding = max([len(i) for i in train_pr_sequences] + 
                    [len(j) for j in cf_train_pr_sequences])
train_data_pr = pad_sequences(train_pr_sequences, maxlen=train_padding, padding='post')
cf_train_data_pr = pad_sequences(cf_train_pr_sequences, maxlen=train_padding, padding='post')
#train_padding = max([len(i) for i in train_hp_sequences] + 
#                    [len(j) for j in cf_train_hp_sequences])
train_data_hp = pad_sequences(train_hp_sequences, maxlen=train_padding, padding='post')
cf_train_data_hp = pad_sequences(cf_train_hp_sequences, maxlen=train_padding, padding='post')

val_pr_sequences = tokenizer.texts_to_sequences(val_prs)
cf_val_pr_sequences = tokenizer.texts_to_sequences(cf_val_prs)
val_padding = max([len(i) for i in val_pr_sequences] + 
                  [len(j) for j in cf_val_pr_sequences])
val_data_pr = pad_sequences(val_pr_sequences, maxlen=val_padding, padding='post')
cf_val_data_pr = pad_sequences(cf_val_pr_sequences, maxlen=val_padding, padding='post')
val_hp_sequences = tokenizer.texts_to_sequences(val_hps)
cf_val_hp_sequences = tokenizer.texts_to_sequences(cf_val_hps)
val_data_hp = pad_sequences(val_hp_sequences, maxlen=val_padding, padding='post')
cf_val_data_hp = pad_sequences(cf_val_hp_sequences, maxlen=val_padding, padding='post')

test_pr_sequences = tokenizer.texts_to_sequences(test_prs)
cf_test_pr_sequences = tokenizer.texts_to_sequences(cf_test_prs)
test_padding = max([len(i) for i in test_pr_sequences] + 
                   [len(j) for j in cf_test_pr_sequences])
test_data_pr = pad_sequences(test_pr_sequences, maxlen=test_padding, padding='post')
cf_test_data_pr = pad_sequences(cf_test_pr_sequences, maxlen=test_padding, padding='post')
test_hp_sequences = tokenizer.texts_to_sequences(test_hps)
cf_test_hp_sequences = tokenizer.texts_to_sequences(cf_test_hps)
test_data_hp = pad_sequences(test_hp_sequences, maxlen=test_padding, padding='post')
cf_test_data_hp = pad_sequences(cf_test_hp_sequences, maxlen=test_padding, padding='post')

# # Iterating with IDs 

# In[15]:


batch_size = 32#BSZ
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def get_cf_dataloader(data_pr, data_hp, data_IDs, cf_data_pr, cf_data_hp, cf_IDs, labels, batch_size):
    # Returns batch_size chunks of (encoded text, ID of text, label of text)
    batches = []
    for i in range(0, len(data_pr), batch_size):
        text_tensor_pr = torch.tensor(data_pr[i:i + batch_size], device=device, dtype=torch.long)
        #length_tensor_pr = torch.tensor([len(j) for j in data_pr[i:i+batch_size]], device=device)
        text_tensor_hp = torch.tensor(data_hp[i:i + batch_size], device=device, dtype=torch.long)
        #length_tensor_hp = torch.tensor([len(j) for j in data_hp[i:i+batch_size]], device=device)
        labels_tensor = torch.tensor(labels[i:i + batch_size], device=device, dtype=torch.long)
        #labels_tensor = torch.tensor(labels[i:i + batch_size], device=device, dtype=torch.float)

        cf_text_tensor_pr = torch.tensor(cf_data_pr[i:i + batch_size], device=device, dtype=torch.long)
        #cf_length_tensor_pr = torch.tensor([len(j) for j in cf_data_pr[i:i+batch_size]], device=device)
        cf_text_tensor_hp = torch.tensor(cf_data_hp[i:i + batch_size], device=device, dtype=torch.long)
        #cf_length_tensor_hp = torch.tensor([len(j) for j in cf_data_hp[i:i+batch_size]], device=device)
        cf_labels_tensor = torch.tensor(labels[i:i + batch_size], device=device, dtype=torch.float)
        #cf_indices = [cf_IDs.index(data_ID) for data_ID in data_IDs[i:i + batch_size]]
        #cf_text_tensor = torch.tensor(cf_data[cf_indices], device=device, dtype=torch.long)
        #cf_length_tensor = torch.tensor([len(j) for j in cf_data[cf_indices]], device=device)
        
        batches.append((text_tensor_pr, text_tensor_hp, cf_text_tensor_pr, cf_text_tensor_hp, labels_tensor))
        #batches.append((text_tensor_pr, length_tensor_pr, text_tensor_hp, cf_text_tensor_pr, cf_length_tensor_pr, labels_tensor))
    batches = batches[:-1]
    return batches

train_dataloader = get_cf_dataloader(train_data_pr, train_data_hp, train_IDs, cf_train_data_pr, cf_train_data_hp, cf_train_IDs, train_labels, batch_size)
val_dataloader = get_cf_dataloader(val_data_pr, val_data_hp, val_IDs, cf_val_data_pr, cf_val_data_hp, cf_val_IDs, val_labels, batch_size)
test_dataloader = get_cf_dataloader(test_data_pr, test_data_hp, test_IDs, cf_test_data_pr, cf_test_data_hp, cf_test_IDs, test_labels, batch_size)
#val_loader = get_dataloader(val_data, val_labels, batch_size)
# val_loader = get_cf_dataloader(val_data, val_IDs, cf_val_data, cf_val_IDs, val_labels, batch_size)
#test_loader = get_dataloader(test_data, test_labels, batch_size)
# test_loader = get_cf_dataloader(test_data, test_IDs, cf_test_data, cf_test_IDs, test_labels, batch_size)

for batch in train_dataloader:
    pr, hp, cf_pr, cf_hp, label = batch
    print(pr.shape)
    print(hp.shape)
    print(cf_pr.shape)
    print(cf_hp.shape)
    print(label.shape)
    break

pickle.dump(tokenizer, open("cf-both-tokenizer.pkl", "wb"))
pickle.dump(train_dataloader, open("cf-both-train.pkl", "wb"))
pickle.dump(val_dataloader, open("cf-both-val.pkl", "wb"))
pickle.dump(test_dataloader, open("cf-both-test.pkl", "wb"))

exit(0)

tokenizer = Tokenizer(num_words=3000, oov_token=True)
source_folder="../counterfactually-augmented-data/NLI/revised_combined/"
train_df = pd.read_csv(source_folder+"train.tsv", delimiter='\t')
print(train_df.keys())
#train_df = train_df.sample(frac=1)
ref={'contradiction':0, 'neutral':1, '-':1, 'entailment':2}
train_prs = [str(i) for i in list(train_df["sentence1"])]
train_hps = [str(i) for i in list(train_df["sentence2"])]
print(sum([i=="-" for i in train_df['gold_label']]))
train_labels = [ref[i] for i in list(train_df["gold_label"])]
train_all = [str(item) for item in train_prs+train_hps]
tokenizer.fit_on_texts(train_all)
train_prs_sequences = tokenizer.texts_to_sequences(train_prs)
train_hps_sequences = tokenizer.texts_to_sequences(train_hps)
max_padding = max([len(i) for i in train_prs_sequences+train_hps_sequences])
train_prs_data = pad_sequences(train_prs_sequences, maxlen=max_padding, padding='post')
train_hps_data = pad_sequences(train_hps_sequences, maxlen=max_padding, padding='post')
batch_size = 32

val_df = pd.read_csv(source_folder+"dev.tsv", delimiter='\t')
val_prs = list(val_df["sentence1"])
val_hps = list(val_df["sentence2"])
val_labels = [ref[i] for i in list(val_df["gold_label"])]
val_prs_sequences = tokenizer.texts_to_sequences(val_prs)
val_hps_sequences = tokenizer.texts_to_sequences(val_hps)
max_padding = max([len(i) for i in val_prs_sequences+val_hps_sequences])
val_prs_data = pad_sequences(val_prs_sequences, maxlen=max_padding, padding='post')
val_hps_data = pad_sequences(val_hps_sequences, maxlen=max_padding, padding='post')

test_df = pd.read_csv(source_folder+"test.tsv", delimiter='\t')
test_prs = list(test_df["sentence1"])
test_hps = list(test_df["sentence2"])
test_labels = [ref[i] for i in list(test_df["gold_label"])]
test_prs_sequences = tokenizer.texts_to_sequences(test_prs)
test_hps_sequences = tokenizer.texts_to_sequences(test_hps)
max_padding = max([len(i) for i in test_prs_sequences+test_hps_sequences])
test_prs_data = pad_sequences(test_prs_sequences, maxlen=max_padding, padding='post')
test_hps_data = pad_sequences(test_hps_sequences, maxlen=max_padding, padding='post')

print(train_prs_data.shape)
print(len(train_labels))
print(val_prs_data.shape)
print(test_prs_data.shape)
device = "cuda"

def get_cf_dataloader(data, cf_data, labels, batch_size):
    batches = []
    for i in range(0, len(data), batch_size):
        text_tensor = torch.tensor(data[i:i + batch_size], device=device, dtype=torch.long)
        length_tensor = torch.tensor([len(j) for j in data[i:i+batch_size]], device=device)
        labels_tensor = torch.tensor(labels[i:i + batch_size], device=device, dtype=torch.float)
        cf_text_tensor = torch.tensor(cf_data[i:i + batch_size], device=device, dtype=torch.long)
        cf_length_tensor = torch.tensor([len(j) for j in cf_data[i:i+batch_size]], device=device)
        batches.append((text_tensor, length_tensor, cf_text_tensor, cf_length_tensor, labels_tensor))
    return batches

def get_dataloader(data_prs, data_hps, labels, batch_size):
    # Returns batch_size chunks of (encoded texts, length of each text, label of each text)
    obj = []
    for i in range(0, len(data_prs), batch_size):
        #print(torch.tensor(data_prs[i:i+batch_size], device=device, dtype=torch.long))
        #print(torch.tensor(data_hps[i:i+batch_size], device=device, dtype=torch.long))
        #print(torch.tensor(labels[i:i+batch_size],device=device, dtype=torch.float))
        obj.append((torch.tensor(data_prs[i:i+batch_size], device=device, dtype=torch.long), torch.tensor(data_hps[i:i+batch_size], device=device, dtype=torch.long), torch.tensor(labels[i:i+batch_size],device=device, dtype=torch.long)))
        #obj.append((torch.tensor(data_prs[i:i+batch_size], device=device, dtype=torch.long), torch.tensor(data_hps[i:i+batch_size], device=device, dtype=torch.long), torch.tensor([len(j) for j in data_prs[i:i+batch_size]], device=device), torch.tensor(labels[i:i+batch_size],device=device, dtype=torch.float)))
    obj = obj[:-1]
    return obj

train_dataloader = get_dataloader(train_prs_data, train_hps_data, train_labels, batch_size)
val_dataloader = get_dataloader(val_prs_data, val_hps_data, val_labels, batch_size)
test_dataloader = get_dataloader(test_prs_data, test_hps_data, test_labels, batch_size)

pickle.dump(tokenizer, open("cf-tokenizer.pkl", "wb"))
pickle.dump(train_dataloader, open("cf-train.pkl", "wb"))
pickle.dump(val_dataloader, open("cf-val.pkl", "wb"))
pickle.dump(test_dataloader, open("cf-test.pkl", "wb"))
#for batch in train_dataloader:
#    pr, hp, labels = batch
#    print(pr.shape)
#    print(hp.shape)
#    print(labels.shape)
#    break
