import os 
import sys
from datasets import Dataset
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_custom_batches(dataset, batch_size=32):
    file_len = []
    for i in range(len(dataset)):
        file_len.append(len(dataset[i]['act']))
    file_len = np.array(file_len)
    batches, batch = [], []
    for i in np.argsort(file_len):
        if len(batch) > 0 and (len(batch) == batch_size or len(dataset[int(i)]['act']) != len(dataset[batch[0]]['act'])):
            batches.append(batch.copy())
            batch = []
        batch.append(int(i))
    if len(batch) > 0:
        batches.append(batch)
    return batches


def get_val_perf(batches_val, validation, model):
    l_, s_ = 0, 0
    sum_loss = 0
    for i in range(len(batches_val)):
        small_dataset = validation.select(batches_val[i])

        emb = np.array(small_dataset['embeddings'])
        emb = torch.from_numpy(emb).to(torch.float32).to(device)

        true = torch.from_numpy((np.array(small_dataset['act'])-1).flatten()).to(device)
        with torch.no_grad():
            pred = model(emb)
        loss = criterion(pred, true)
        sum_loss += loss.item() / len(true)
        l_ += (torch.argmax(pred, axis=1)==true).sum().item()
        s_ += len(pred)
    return l_ / s_, sum_loss / len(batches_val)
