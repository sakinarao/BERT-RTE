"""
RTE task using BERT
Source Code: https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=GEgLpFVlo1Z-
https://huggingface.co/docs/transformers/model_doc/bert

"""

import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BERT tokenizer.
print('\n\n\n------------RTE Task Using BERT-------------\n\n')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

from transformers import BertForSequenceClassification, AdamW, BertConfig
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", #12-layer BERT model, with an uncased vocab
    num_labels = 2, #Output labels for binary classification   
    output_attentions = False, # if model returns attentions weights
    output_hidden_states = False, # if model returns all hidden-states
)

model.cuda()

from torch.utils.data import Dataset

def load_data(df):
    MAX_LEN = 512
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = []

    premise_list = df['premise'].to_list()
    hypothesis_list = df['hypothesis'].to_list()
    label_list = df['label'].to_list()

    for (premise, hypothesis, label) in zip(premise_list, hypothesis_list, label_list):
      premise_id = tokenizer.encode(premise, add_special_tokens = False,max_length=512, truncation=True)
      hypothesis_id = tokenizer.encode(hypothesis, add_special_tokens = False,max_length=512, truncation=True)
      pair_token_ids = [tokenizer.cls_token_id] + premise_id + [tokenizer.sep_token_id] + hypothesis_id + [tokenizer.sep_token_id]
      premise_len = len(premise_id)
      hypothesis_len = len(hypothesis_id)
      segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))
      attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))

      token_ids.append(torch.tensor(pair_token_ids))
      seg_ids.append(segment_ids)
      mask_ids.append(attention_mask_ids)
      y.append(label_dict[str(label)])


    from torch.nn.utils.rnn import pad_sequence
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    y = torch.tensor(y)
   
    dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
    print(len(dataset))
    return dataset

import pandas as pd
train_df = pd.read_csv("/home/csgrads/rao00134/SuperGlue-tasks-using-BERT/dataset/RTE/train.csv", delimiter=',', header= None, names=['premise', 'hypothesis', 'label', 'Index'])
valid_df = pd.read_csv("/home/csgrads/rao00134/SuperGlue-tasks-using-BERT/dataset/RTE/val.csv", delimiter=',', header= None, names=['premise', 'hypothesis', 'label', 'Index'])
label_dict = {'entailment': 0, 'not_entailment': 1}

print('Number of training sentences: {:,}\n'.format(train_df.shape[0]))
train_df.sample(5)

print('Number of validation sentences: {:,}\n'.format(valid_df.shape[0]))
valid_df.sample(5)

train_data = load_data(train_df)
valid_data = load_data(valid_df)

train_loader = DataLoader(train_data,batch_size=32,shuffle=True,)
val_loader = DataLoader(valid_data,batch_size=32,shuffle=True)

def accuracy(y_pred, y_test):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
  return acc

import time
from transformers import get_linear_schedule_with_warmup
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

# Number of EPOCHS
EPOCHS = 6
total_steps = len(train_loader) * EPOCHS

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

def train(model, train_loader, val_loader, optimizer,scheduler):  
  total_step = len(train_loader)

  for epoch in range(EPOCHS):
    #Time for training epochs
    start = time.time()
    model.train()

    
    total_train_loss = 0
    total_train_acc  = 0
    for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(train_loader):

      # Unpack the training batch from our dataloader
      pair_token_ids = pair_token_ids.to(device)
      mask_ids = mask_ids.to(device)
      seg_ids = seg_ids.to(device)
      labels = y.to(device)

      #clear any previously calculated gradients before performing a backward pass
      optimizer.zero_grad()

      #Get the loss and prediction
      loss, prediction = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()

      acc = accuracy(prediction, labels)
      
      # Accumulate the training loss and accuracy over all of the batches for calculating average loss at end
      total_train_loss += loss.item()
      total_train_acc  += acc.item()

      # Perform a backward pass to calculate the gradients.
      loss.backward()

      # Clip the norm of the gradients to 1.0.
      # This is to help prevent the "exploding gradients" problem.
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      # Update parameters and take a step using the computed gradient.
      optimizer.step()

      # Update the learning rate.
      scheduler.step()

    # Calculate the average accuracy and loss over all of the batches.
    train_acc  = total_train_acc/len(train_loader)
    train_loss = total_train_loss/len(train_loader)

    # EVALUATION MODE
    model.eval()

    total_val_acc  = 0
    total_val_loss = 0
    with torch.no_grad():
      for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(val_loader):

        #clear any previously calculated gradients before performing a backward pass
        optimizer.zero_grad()

        # Unpack this validation batch from our dataloader. 
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = y.to(device)
        
        #Get the loss and prediction
        loss, prediction = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()

        # Calculate the accuracy for this batch
        acc = accuracy(prediction, labels)

        # Accumulate the validation loss and Accuracy
        total_val_loss += loss.item()
        total_val_acc  += acc.item()

    # Calculate the average accuracy and loss over all of the batches.
    val_acc  = total_val_acc/len(val_loader)
    val_loss = total_val_loss/len(val_loader)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

train(model, train_loader, val_loader, optimizer,scheduler)

# Load Test data
test_data_df = pd.read_csv("/home/csgrads/rao00134/SuperGlue-tasks-using-BERT/dataset/RTE/test.csv", header=None, names=('premise','hypothesis','label','idx'))

premise_list = test_data_df['premise'].to_list()
hypothesis_list = test_data_df['hypothesis'].to_list()

# Predict the answer
def predict(premise, hypothesis):
  sequence = tokenizer.encode_plus(premise, hypothesis, return_tensors="pt")['input_ids'].to(device)
  logits = model(sequence)[0]
  probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
 
  proba_yes = round(probabilities[1], 3)
  proba_no = round(probabilities[0], 3)
  print(f"premise: {premise},hypothesis:{hypothesis} entailment: {proba_yes}, not_entailment: {proba_no}")


for i in range(10):
  premise = premise_list[i]
  hypothesis = hypothesis_list[i]
  predict(premise,hypothesis)

"""
#Predictions for Test Data
def predict(premise, hypothesis):
  premise_id = tokenizer.encode(premise, add_special_tokens = False,max_length=512)
  hypothesis_id = tokenizer.encode(hypothesis, add_special_tokens = False,max_length=512)
  sequence = ([tokenizer.cls_token_id] + premise_id + [tokenizer.sep_token_id] + hypothesis_id + [tokenizer.sep_token_id])
  sequence=torch.tensor([sequence]).to(device)
  logits = model(sequence)[0]
  probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
  proba_yes = round(probabilities[1], 3)
  proba_no = round(probabilities[0], 3)
  print(f"premise: {premise},hypothesis:{hypothesis} entailment: {proba_yes}, not_entailment: {proba_no}")

for i in range(5):
  premise = premise_list[i]
  hypothesis = hypothesis_list[i]
  predict(premise,hypothesis)

"""