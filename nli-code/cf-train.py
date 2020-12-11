import os

import time
import datetime
import random
random.seed(123)
import numpy as np
np.random.seed(123)
import torch
torch.manual_seed(123)
import torch.optim as O
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import logging
import pickle

import datasets
import models

from utils import *
from pdb import set_trace

BATCH_SIZE = 32

class Train():
    def __init__(self):
        print("program execution start: {}".format(datetime.datetime.now()))
        self.args = parse_args()
        self.device = get_device(self.args.gpu)
        self.logger = get_logger(self.args, "train")
        self.logger.info("Arguments: {}".format(self.args))
        
        dataset_options = {
                                            'batch_size': self.args.batch_size, 
                                            'device': self.device
                                        }
        print(self.args.dataset, dataset_options)
        self.dataset = datasets.__dict__[self.args.dataset](dataset_options)
        
        self.model_options = {
                                    'out_dim': self.dataset.out_dim(),
                                    'dp_ratio': self.args.dp_ratio,
                                    'd_hidden': self.args.d_hidden,
                                    'device': self.device,
                                    'dataset': self.args.dataset
                                }
        self.model = models.__dict__[self.args.model](self.model_options)
        
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
        self.opt = O.Adam(self.model.parameters(), lr = self.args.lr)
        self.best_val_acc = None
        self.scheduler = StepLR(self.opt, step_size=5, gamma=0.5)

        print("resource preparation done: {}".format(datetime.datetime.now()))

    def result_checkpoint(self, epoch, train_loss, val_loss, train_acc, val_acc, took):
        if self.best_val_acc is None or val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save({
                'accuracy': self.best_val_acc,
                'options': self.model_options,
                'model_dict': self.model.state_dict(),
            }, '{}/{}/{}/best-{}-{}-params.pt'.format(self.args.results_dir, self.args.model, self.args.dataset, self.args.model, self.args.dataset))
        self.logger.info('| Epoch {:3d} | train loss {:5.2f} | train acc {:5.2f} | val loss {:5.2f} | val acc {:5.2f} | time: {:5.2f}s |'
                .format(epoch, train_loss, train_acc, val_loss, val_acc, took))

    def clp_loss(self, output, labels, cf_output, lambda_coef):
        counterfactual_loss = (output - cf_output).abs().sum()
        sigmoid_out = torch.sigmoid(output)
        #print("Loss function ", sigmoid_out)
        loss = self.criterion(sigmoid_out, labels) - lambda_coef * counterfactual_loss
        #loss = criterion(output, labels) - lambda_coef * counterfactual_loss
        return loss
    
    def train(self):
        self.model.train(); self.dataset.train_iter.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0
        #for batch_idx, batch in enumerate(self.dataset.train_iter):
        #    print(batch)
        #    print(batch.premise)
        #    print(batch.hypothesis)
        #    print(batch.batch_size) 
        #    ch = input()
        train_dataloader = pickle.load(open("cf-both-train.pkl", "rb"))
        #print("Using pickled")
        for premise, hypothesis, cf_premise, cf_hypothesis, label in train_dataloader:
            self.opt.zero_grad()
            answer = self.model(premise, hypothesis, BATCH_SIZE)#batch)
            cf_answer = self.model(cf_premise, cf_hypothesis, BATCH_SIZE)#batch)
            #answer = torch.sigmoid(answer)
            loss = self.clp_loss(answer, label, cf_answer, 0.000)#self.criterion(answer, label)#batch.label)
             
            n_correct += (torch.max(answer, 1)[1].view(label.size()) == label).sum().item()
            #n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
            n_total += BATCH_SIZE#batch.batch_size
            n_loss += loss.item()
            
            loss.backward(); self.opt.step()
        train_loss = n_loss/n_total
        train_acc = 100. * n_correct/n_total
        return train_loss, train_acc

    def validate(self):
        self.model.eval(); self.dataset.dev_iter.init_epoch()
        n_correct, n_total, n_loss = 0, 0, 0
        with torch.no_grad():
            val_loader = pickle.load(open("cf-both-val.pkl", "rb"))
            for premise, hypothesis, cf_premise, cf_hypothesis, label in val_loader:
            #for batch_idx, batch in enumerate(self.dataset.dev_iter):
                answer = self.model(premise, hypothesis, BATCH_SIZE)#batch)
                answer = torch.sigmoid(answer)
                loss = self.criterion(answer, label)#batch.label)
                
                n_correct += (torch.max(answer, 1)[1].view(label.size()) == label).sum().item()
                #n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
                n_total += BATCH_SIZE#batch.batch_size
                n_loss += loss.item()

            val_loss = n_loss/n_total
            val_acc = 100. * n_correct/n_total
            return val_loss, val_acc

    def execute(self):
        print(" [*] Training starts!")
        print('-' * 99)
        best_so_far = -1
        for epoch in range(1, self.args.epochs+1):
            start = time.time()

            train_loss, train_acc = self.train()
            val_loss, val_acc = self.validate()
            self.scheduler.step()
            
            took = time.time()-start
            if val_acc>best_so_far:
                best_so_far = val_acc
                print("Saving checkpoint, new best ", best_so_far)
                self.result_checkpoint(epoch, train_loss, val_loss, train_acc, val_acc, took)

            print('| Epoch {:3d} | train loss {:5.2f} | train acc {:5.2f} | val loss {:5.2f} | val acc {:5.2f} | time: {:5.2f}s |'.format(
                epoch, train_loss, train_acc, val_loss, val_acc, took))
        self.finish()

    def finish(self):
        self.logger.info("[*] Training finished!\n\n")
        print('-' * 99)
        print(" [*] Training finished!")
        print(" [*] Please find the saved model and training log in results_dir")

task = Train()
task.execute()
