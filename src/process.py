import os 
import sys
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
import torch


dic = {'bert': 'all-MiniLM-L6-v2'} # rajouter les autres embedder dans le futur 

### Daily dialog embedding

def embed_daily_dialog(name='bert'):
    model = SentenceTransformer(dic[name])
    dataset = load_dataset('daily_dialog','multi-label')  
    training_dataset = dataset['train'].map(lambda row: {'embeddings': model.encode(row['dialog'])}, batched=False)
    validation_dataset = dataset['validation'].map(lambda row: {'embeddings': model.encode(row['dialog'])}, batched=False)
    test_dataset = dataset['test'].map(lambda row: {'embeddings': model.encode(row['dialog'])}, batched=False)
    #training_dataset.to_csv('data/daily_dialog_' + name + '.hug') au pif mais t'as capté
    #validation_dataset.to_csv(data/daily_dialog_bert.hug)
    #test_dataset.to_csv(data/daily_dialog_bert.hug)

