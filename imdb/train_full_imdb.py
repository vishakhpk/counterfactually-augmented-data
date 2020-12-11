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
                             confusion_matrix)

from simple_lstm import LSTM
from simple_lstm import (save_metrics, load_metrics, save_checkpoint,
                         load_checkpoint)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epochs', type=int, default = 20, help='Num Epochs')
parser.add_argument('--lr', type=float, default = 0.0005, help='Learning rate')
parser.add_argument('--batch_size', type=int, default = 32, help='Batch size')
parser.add_argument('--vocab_size', type=int, default = 3000,
                    help='Vocab size for lstm')
parser.add_argument('--output_path', type=str, default = "./models",
                    help='Output path')
parser.add_argument('--prepath', type=str, default=None,
                    help='Path to pretrained model for warm starting')
args = parser.parse_args()

EPOCHS = args.epochs
LR = args.lr
OUT_DIR = args.output_path
VOCAB_SIZE = args.vocab_size
BSZ = args.batch_size
PRETRAIN_PATH = args.prepath

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = f'epochs={EPOCHS},lr={LR},vocab={VOCAB_SIZE},bsz={BSZ}'
print(f'params: {params}')
model_name = 'imdb-pretrain'

# load full dataset
df = pd.read_csv('data/imdb_full.csv')
X_all, y_all = df.Text, df.Sentiment
y_all = (y_all == 'Positive').astype(int).tolist()

# split into train, val, test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=123, shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=123, shuffle=True)

print('Dataset size:')
print(f'{len(y_train)} train, {len(y_val)} val, {len(y_test)} test')


# setup tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=True)
tokenizer.fit_on_texts(X_train)

# tokenize, convert to sequences, and pad
# note: using the same padding for factual/counterfactual data
def get_padded_sequences(text):
    sequences = tokenizer.texts_to_sequences(text)
    padding = max([len(i) for i in sequences])
    data = pad_sequences(sequences, maxlen=padding, padding='post')
    return data

train_sequences = get_padded_sequences(X_train.tolist())
val_sequences = get_padded_sequences(X_val.tolist())
test_sequences = get_padded_sequences(X_test.tolist())


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


train_loader = get_dataloader(train_sequences, y_train, BSZ)
val_loader = get_dataloader(val_sequences, y_val, BSZ)
test_loader = get_dataloader(test_sequences, y_test, BSZ)


# train and test ------------------------------------------------------------- #
destination_folder = OUT_DIR
lambda_coef = LAMBDA
criterion = torch.nn.BCELoss()

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
        for text, text_len, labels in train_loader:
            labels = labels.to(device)
            text = text.to(device)

            output = model(text, text_len)
            output = torch.sigmoid(output)
            loss = criterion(output, labels)

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
                    for text, text_len, labels in valid_loader:
                        labels = labels.to(device)
                        text = text.to(device)

                        output = model(text, text_len)
                        output = torch.sigmoid(output)

                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

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

# Evaluation Function
def evaluate(model, test_loader, version='title', threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for text, text_len, labels in test_loader:
            # labels
            labels = labels.to(device)
            y_true.extend(labels.tolist())

            # factual predictions
            text = text.to(device)
            output = model(text, text_len)

            sigmoid_out = torch.sigmoid(output)
            output = (sigmoid_out > threshold).int()

            y_pred.extend(output.tolist())


    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))


model = LSTM(vocab_size = VOCAB_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr = LR)

train(model=model, optimizer=optimizer, num_epochs = EPOCHS)
train_loss_list, valid_loss_list, global_steps_list = load_metrics(
    destination_folder + f'/metrics-{model_name}.pt')

best_model = LSTM(vocab_size=VOCAB_SIZE).to(device)
optimizer = optim.Adam(best_model.parameters(), lr=LR)

load_checkpoint(destination_folder + f'/model-{model_name}.pt', best_model,
                optimizer)
evaluate(best_model, test_loader)
