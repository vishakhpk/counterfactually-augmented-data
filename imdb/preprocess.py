import pandas as pd


print('Preprocessing csvs...')

# load data
path = 'data/{}_paired.tsv'
train_path = path.format('train')
val_path = path.format('dev')
test_path = path.format('test')

train_df = pd.read_table(train_path)
val_df = pd.read_table(val_path)
test_df = pd.read_table(test_path)

# standardize columns
train_df.columns = train_df.columns.str.lower()
val_df.columns = val_df.columns.str.lower()
test_df.columns = test_df.columns.str.lower()

# convert labels to binary (1 if positive, 0 if negative)
train_df['label'] = (train_df['sentiment'] == 'Positive').astype(int)
val_df['label'] = (val_df['sentiment'] == 'Positive').astype(int)
test_df['label'] = (test_df['sentiment'] == 'Positive').astype(int)


# split into factual and counterfactual
factual_indices = list(range(0, len(train_df), 2))
counterfactual_indices = list(range(1, len(train_df), 2))
cf_train_df = train_df.iloc[counterfactual_indices]
train_df = train_df.iloc[factual_indices]

factual_indices = list(range(0, len(val_df), 2))
counterfactual_indices = list(range(1, len(val_df), 2))
cf_val_df = val_df.iloc[counterfactual_indices]
val_df = val_df.iloc[factual_indices]

factual_indices = list(range(0, len(test_df), 2))
counterfactual_indices = list(range(1, len(test_df), 2))
cf_test_df = test_df.iloc[counterfactual_indices]
test_df = test_df.iloc[factual_indices]

# join as columns
train_df['cf-text'] = cf_train_df['text'].values
train_df['cf-label'] = cf_train_df['label'].values

val_df['cf-text'] = cf_val_df['text'].values
val_df['cf-label'] = cf_val_df['label'].values

test_df['cf-text'] = cf_test_df['text'].values
test_df['cf-label'] = cf_test_df['label'].values

# remove unncessary columns
train_df.drop(columns=['sentiment', 'batch_id'], inplace=True)
val_df.drop(columns=['sentiment', 'batch_id'], inplace=True)
test_df.drop(columns=['sentiment', 'batch_id'], inplace=True)

# save factual-only set with pairs to csv
new_path = 'data/'
train_df.to_csv(new_path + 'fact_train.csv', index=False)
val_df.to_csv(new_path + 'fact_val.csv', index=False)
test_df.to_csv(new_path + 'fact_test.csv', index=False)

# make augmented set with pairs by repeating rows with flipped names
flipped_df = train_df.copy()
flipped_df.columns = ['cf-text', 'cf-label', 'text', 'label']
aug_train_df = train_df.append(flipped_df)

flipped_df = val_df.copy()
flipped_df.columns = ['cf-text', 'cf-label', 'text', 'label']
aug_val_df = val_df.append(flipped_df)

flipped_df = test_df.copy()
flipped_df.columns = ['cf-text', 'cf-label', 'text', 'label']
aug_test_df = test_df.append(flipped_df)

# save augmented set with pairs to csv
aug_train_df.to_csv(new_path + 'aug_train.csv', index=False)
aug_val_df.to_csv(new_path + 'aug_val.csv', index=False)
aug_test_df.to_csv(new_path + 'aug_test.csv', index=False)

print('Done.')
