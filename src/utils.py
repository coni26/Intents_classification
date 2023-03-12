import os 
import sys
from datasets import Dataset
import numpy as np
import torch
from torch import nn
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
    criterion = nn.CrossEntropyLoss(reduction='sum')
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


def get_nb_parameters(model):
    res = 0
    for name, param in model.named_parameters():
        if len(param.shape) > 1:
            res += param.shape[0] * param.shape[1]
        else:
            res += param.shape[0]
    return res


def get_model_stats(model, test, nb_cat=4):
    predictions = []
    for i in range(len(test)):
        emb = np.array(test[i]['embeddings'])
        emb = torch.from_numpy(emb).to(torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(emb)
        predictions.append(torch.argmax(pred, axis=1).cpu().numpy())
        
    max_len = max(map(len, test['act']))
    res = np.nan * np.ones(shape=(len(test), max_len))
    true_mat = np.nan * np.ones(shape=(len(test), max_len))
    pred_mat = np.nan * np.ones(shape=(len(test), max_len))
    res_len = np.zeros((max_len-1, 2), dtype=int)
    res_cat = np.zeros((nb_cat, 2), dtype=int)
    cat_pos = np.zeros((max_len, 4), dtype=int)
    res_last = 0

    for i in range(len(test)):
        true = np.array(test[i]['act']) - 1
        res[i, 0:len(true)] = true == predictions[i]
        true_mat[i, 0:len(true)] = true 
        pred_mat[i, 0:len(true)] = predictions[i] 
        res_len[len(true)-2, 0] += (true == predictions[i]).sum()
        res_len[len(true)-2, 1] += len(true)
        for j in range(len(true)):
            res_cat[true[j], 0] += predictions[i][j] == true[j]
            res_cat[true[j], 1] += 1
            cat_pos[j, true[j]] += 1
        res_last += predictions[i][-1] == true[-1]
        

    accuracy = np.nansum(res) / np.sum(~np.isnan(res))
    acc_per_position = np.nansum(res, axis=0) / np.sum(~np.isnan(res), axis=0)
    acc_per_len = res_len[:, 0] / res_len[:, 1]
    acc_first = acc_per_position[0]
    acc_last = res_last / len(test)
    return accuracy, acc_per_position, acc_per_len, acc_first, acc_last


def train(model, nb_epochs=100):
    val_losses, train_losses = [], []
    
    if torch.cuda.is_available():
        model.cuda()
        
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=20,  gamma=0.1)

    for epoch in range(nb_epochs):
        sum_loss = 0
        random.shuffle(train_batches)
        for i, batch in enumerate(train_batches):
            batch_train = train.select(batch)
            emb = np.array(batch_train['embeddings'])
            emb = torch.from_numpy(emb).to(torch.float32).to(device)

            true = torch.from_numpy((np.array(batch_train['act'])-1).flatten()).to(device)

            optimizer.zero_grad()
            pred = model(emb)
            loss = criterion(pred, true)
            loss.backward()

            optimizer.step()
            sum_loss += loss.item() / len(true)
        scheduler.step()
        val_acc, val_loss = get_val_perf(val_batches, validation, model)
        val_losses.append(val_loss)
        train_losses.append(sum_loss / len(train_batches))
        print("Epoch: {:>3} | Loss: ".format(epoch) + f"{sum_loss / len(train_batches):.4e}" + " | Validation Loss: " + f"{val_loss:.4e}" + " | Validation Accuracy: " + f"{round(val_acc, 4)}")
    return model