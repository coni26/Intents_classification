import os 
import sys
from datasets import load_dataset, Dataset, load_from_disk
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd


def get_encoders():
    df = pd.read_csv('encoders.csv')
    return df


def embed_daily_dialog(encoder='all-MiniLM-L6-v2', test_only=False):
    model = SentenceTransformer(encoder)
    dataset = load_dataset('daily_dialog','multi-label')  
    if test_only:
        test_dataset = dataset['test'].map(lambda row: {'embeddings': model.encode(row['dialog'])}, batched=False)
        test_dataset.save_to_disk("data/daily_dialog_" + encoder + "_test.hf")
    else:
        training_dataset = dataset['train'].map(lambda row: {'embeddings': model.encode(row['dialog'])}, batched=False)
        validation_dataset = dataset['validation'].map(lambda row: {'embeddings': model.encode(row['dialog'])}, batched=False)
        test_dataset = dataset['test'].map(lambda row: {'embeddings': model.encode(row['dialog'])}, batched=False)
        training_dataset.save_to_disk("data/daily_dialog_" + encoder + "_train.hf")
        validation_dataset.save_to_disk("data/daily_dialog_" + encoder + "_validation.hf")
        test_dataset.save_to_disk("data/daily_dialog_" + encoder + "_test.hf")

    

def load_data_embedded(encoder='all-MiniLM-L6-v2', test_only=False):
    arr = os.listdir('data/')
    if test_only:
        if not("daily_dialog_" + encoder + "_train.hf" in arr):
            embed_daily_dialog(encoder, test_only=test_only)
        test = load_from_disk("data/daily_dialog_" + encoder + "_test.hf")
        return None, None, test
    else:
        if not("daily_dialog_" + encoder + "_train.hf" in arr):
            embed_daily_dialog(encoder, test_only=test_only)
        train = load_from_disk("data/daily_dialog_" + encoder + "_train.hf")  
        validation = load_from_disk("data/daily_dialog_" + encoder + "_validation.hf")  
        test = load_from_disk("data/daily_dialog_" + encoder + "_test.hf")   
        return train, validation, test
