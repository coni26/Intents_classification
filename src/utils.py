import os 
import sys
from datasets import Dataset
import numpy as np


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
