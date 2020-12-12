import random
import argparse

import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from torchtext.data import Field
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from simple_lstm import LSTM
from simple_lstm import (save_metrics, load_metrics, save_checkpoint,
                         load_checkpoint)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epochs', type=int, default = 20, help='Num Epochs')
parser.add_argument('--lambda_coeff', type=float, default = 0.0005,
                    help='Lambda value for logit pairing')
parser.add_argument('--lr', type=float, default = 0.0005, help='Learning rate')
parser.add_argument('--batch_size', type=int, default = 32, help='Batch size')
parser.add_argument('--vocab_size', type=int, default = 3000,
                    help='Vocab size for lstm')
parser.add_argument('--output_path', type=str, default = "models",
                    help='Output path')
parser.add_argument('--aug', type=int, default=1,
                    help='Whether or not to cf-augment the train/val sets (0 or 1)')
parser.add_argument('--prepath', type=str, default=None,
                    help='Relative path *within models/* to pretrained model for warm starting')
args = parser.parse_args()

EPOCHS = args.epochs
LAMBDA = args.lambda_coeff
LR = args.lr
OUT_DIR = args.output_path
VOCAB_SIZE = args.vocab_size
BSZ = args.batch_size
AUGMENTED = args.aug
PRETRAIN_PATH = args.prepath

model_name = f'epochs={EPOCHS},lambda={LAMBDA},lr={LR},vocab={VOCAB_SIZE},' \
             f'bsz={BSZ},aug={AUGMENTED}'

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'params: {model_name}')

if PRETRAIN_PATH is not None:
    print(f'Starting with pretrained model:\n\t{PRETRAIN_PATH}')
    model_name += f'+{PRETRAIN_PATH[:-3]}'  # remove .pt


# load data ------------------------------------------------------------------ #
def load_imdb(split, augmented, random_state=123):
    # split: 'train', 'val', 'test'
    # augmented: bool

    # load requested csv
    if augmented:
        path = 'data/aug_{}.csv'
    else:
        path = 'data/fact_{}.csv'
    df = pd.read_csv(path.format(split))

    # shuffle the data
    df = df.sample(frac=1, random_state=random_state)

    return df

train_df = load_imdb(split='train', augmented=AUGMENTED)
val_df = load_imdb(split='val', augmented=AUGMENTED)
test_df = load_imdb(split='test', augmented=True)  # test data always augmented

print('Dataset size:')
print(f'{len(train_df)} train, {len(val_df)} val, {len(test_df)} test')

# format for lstm ------------------------------------------------------------ #
# build vocabulary/tokenizer from training data
if AUGMENTED:
    # no need to include counterfactual examples for vocab/tokenizer
    vocab_texts = train_df['text'].tolist()
else:
    # include counterfactual examples for vocab/tokenizer
    vocab_texts = train_df['text'].tolist() + train_df['cf-text'].tolist()

# setup tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=True)
tokenizer.fit_on_texts(vocab_texts)

# tokenize, convert to sequences, and pad
# note: using the same padding for factual/counterfactual data
def get_padded_sequences(df):
    sequences = tokenizer.texts_to_sequences(df['text'])
    cf_sequences = tokenizer.texts_to_sequences(df['cf-text'])
    padding = max([len(i) for i in sequences] +
                  [len(j) for j in cf_sequences])
    data = pad_sequences(sequences, maxlen=padding, padding='post')
    cf_data = pad_sequences(cf_sequences, maxlen=padding, padding='post')

    return data, cf_data

train_sequences, cf_train_sequences = get_padded_sequences(train_df)
val_sequences, _ = get_padded_sequences(val_df)  # no val cf metrics needed
test_sequences, cf_test_sequences = get_padded_sequences(test_df)

y_train = train_df['label'].values
y_val = val_df['label'].values
y_test = test_df['label'].values

def get_dataloader(data, labels, batch_size):
    batches = []
    for i in range(0, len(data), batch_size):
        text_tensor = torch.tensor(data[i:i + batch_size], device=device,
                                   dtype=torch.long)
        length_tensor = torch.tensor([len(j) for j in data[i:i+batch_size]],
                                     device=device)
        labels_tensor = torch.tensor(labels[i:i + batch_size], device=device,
                                     dtype=torch.float)
        batches.append((text_tensor, length_tensor, labels_tensor))
    return batches

def get_cf_dataloader(data, cf_data, labels, batch_size):
    batches = []
    for i in range(0, len(data), batch_size):
        text_tensor = torch.tensor(
            data[i:i + batch_size], device=device, dtype=torch.long)
        length_tensor = torch.tensor(
            [len(j) for j in data[i:i+batch_size]], device=device)
        labels_tensor = torch.tensor(
            labels[i:i + batch_size], device=device, dtype=torch.float)

        cf_text_tensor = torch.tensor(
            cf_data[i:i + batch_size], device=device, dtype=torch.long)
        cf_length_tensor = torch.tensor(
            [len(j) for j in cf_data[i:i+batch_size]], device=device)

        batches.append((text_tensor, length_tensor, cf_text_tensor,
                        cf_length_tensor, labels_tensor))
    return batches

train_loader = get_cf_dataloader(train_sequences, cf_train_sequences, y_train,
                                 BSZ)
val_loader = get_dataloader(val_sequences, y_val, BSZ)
# we want regular and cf test metrics
cf_test_loader = get_cf_dataloader(test_sequences, cf_test_sequences, y_test,
                                   BSZ)

# train and test ------------------------------------------------------------- #
destination_folder = OUT_DIR
lambda_coef = LAMBDA
criterion = torch.nn.BCELoss()

def clp_loss(criterion, output, labels, cf_output, lambda_coef):
    counterfactual_loss = (output - cf_output).abs().sum()
    sigmoid_out = torch.sigmoid(output)
    loss = criterion(sigmoid_out, labels) - lambda_coef * counterfactual_loss
    return loss

epochs_vs_performance = {
    'epoch': [],
    'step': [],
    'f1_score': [],
}

def train(model,
          optimizer,
          criterion = criterion,
          train_loader = train_loader,
          train_batches = len(train_loader),
          valid_loader = val_loader,
          valid_batches = len(val_loader),
          num_epochs = 5,
          eval_every = len(train_loader) // 2,
          file_path = destination_folder,
          best_valid_loss = float("Inf")):

    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for text, text_len, cf_text, cf_text_len, labels in train_loader:
            labels = labels.to(device)
            text = text.to(device)
            cf_text = cf_text.to(device)

            cf_output = model(cf_text, cf_text_len)
            output = model(text, text_len)

            loss = clp_loss(criterion, output, labels, cf_output, lambda_coef)
            # loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    y_pred_val = []
                    y_true_val = []
                    for text, text_len, labels in valid_loader:
                        labels = labels.to(device)
                        text = text.to(device)

                        output = model(text, text_len)
                        output = torch.sigmoid(output)

                        # ugly but it works
                        y_true_val.extend(labels.cpu().numpy().tolist())
                        y_pred_val.extend(output.cpu().numpy().tolist())

                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

                    epochs_vs_performance['epoch'].append(epoch)
                    epochs_vs_performance['step'].append(global_step)
                    epochs_vs_performance['f1_score'].append(
                        f1_score(y_true_val, y_pred_val))
                    print(y_true_val[:10])
                    print(y_pred_val[:10])

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / valid_batches
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step,
                              num_epochs*train_batches, average_train_loss,
                              average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + f'/model-{model_name}.pt',
                                    model, optimizer, best_valid_loss)
                    save_metrics(file_path + f'/metrics-{model_name}.pt',
                                 train_loss_list, valid_loss_list,
                                 global_steps_list)

    save_metrics(file_path + f'/metrics-{model_name}.pt', train_loss_list,
                 valid_loss_list, global_steps_list)
    print('Finished Training!')

model = LSTM(vocab_size = VOCAB_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr = LR)

if PRETRAIN_PATH is not None:
    print(f'Loading pretrained model:\n\t{PRETRAIN_PATH}')
    load_checkpoint('models/' + PRETRAIN_PATH, model, optimizer)

train(model=model, optimizer=optimizer, num_epochs = EPOCHS)
train_loss_list, valid_loss_list, global_steps_list = load_metrics(
    destination_folder + f'/metrics-{model_name}.pt')

# Evaluation Function
def evaluate(model, test_loader, version='title', threshold=0.5):
    y_true_fact = []

    y_pred_fact = []
    y_pred_cfact = []

    y_raw_fact = []
    y_raw_cfact = []

    model.eval()
    with torch.no_grad():
        for text, text_len, cf_text, cf_text_len, labels in test_loader:
            # labels
            labels = labels.to(device)
            y_true_fact.extend(labels.tolist())

            # factual predictions
            text = text.to(device)
            output = model(text, text_len)

            sigmoid_out = torch.sigmoid(output)
            y_raw_fact.extend(sigmoid_out.tolist())

            output = (sigmoid_out > threshold).int()
            y_pred_fact.extend(output.tolist())

            # cf predictions
            cf_text = cf_text.to(device)
            cf_output = model(cf_text, cf_text_len)

            cf_sigmoid_out = torch.sigmoid(cf_output)
            y_raw_cfact.extend(cf_sigmoid_out.tolist())

            cf_output = (cf_sigmoid_out > threshold).int()
            y_pred_cfact.extend(cf_output.tolist())


    print('Classification Report:')
    print(classification_report(y_true_fact, y_pred_fact, labels=[1, 0],
                                digits=4))

    # CF Consistency:
    # fraction of cf pairs that receive different predictions
    # 1 indicates consistency, 0 indicates lack of consistency
    # note all pairs are asymmetric
    print(f'CF Consistency: {np.not_equal(y_pred_fact, y_pred_cfact).mean()}')

    # CF Gap:
    # mean absolute difference in prediction
    # larger is better
    # differences = []
    # for batch_a, batch_b in zip(y_fact_out, y_cfact_out):
    #     batch_diff = (batch_a - batch_b).abs().tolist()
    #     differences.extend(batch_diff)
    mean_difference = np.abs(np.subtract(y_raw_fact, y_raw_cfact)).mean()
    print(f'CF Gap: {mean_difference}')

    # save output
    results_df = pd.DataFrame({
        'y_true_fact': y_true_fact,
        'y_pred_fact': y_pred_fact,
        'y_pred_cfact': y_pred_cfact,
        'y_raw_fact': y_raw_fact,
        'y_raw_cfact': y_raw_cfact,
    })

    results_df.to_csv(f'results/{model_name}.csv', index=False)

best_model = LSTM(vocab_size=VOCAB_SIZE).to(device)
optimizer = optim.Adam(best_model.parameters(), lr=LR)

load_checkpoint(destination_folder + f'/model-{model_name}.pt', best_model,
                optimizer)
evaluate(best_model, cf_test_loader)

pd.DataFrame(epochs_vs_performance).to_csv(f'results/f1-{model_name}.csv',
                                           index=False)
