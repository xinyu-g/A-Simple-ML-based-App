import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from . import constants as c
import torch



def split_data(receipt, length):
    data_raw = receipt.to_numpy() # convert to numpy array
    data = []

    print(len(data_raw))
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - length + 1): 
        data.append(data_raw[index: index + length])
    
    data = np.array(data)
    print(len(data))

    x_train = data[:,:-1,:]
    y_train = data[:,-1,:]

    return [x_train, y_train]


def plot(original, predict, hist, file):
    
    sns.set_style("darkgrid")    

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
    ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (LSTM)", color='tomato')
    ax.set_title('Stock price', size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Receipt Count", size = 14)
    ax.set_xticklabels('', size=10)


    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=hist, color='royalblue')
    ax.set_xlabel("Epoch", size = 14)
    ax.set_ylabel("Loss", size = 14)
    ax.set_title("Training Loss", size = 14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)

    fig.savefig(file)


def train(model, x_train, y_train, criterion, optimiser):
    hist = np.zeros(c.num_epochs)
    start_time = time.time()
    # lstm = []

    for t in range(c.num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))

    return y_train_pred, hist


def predict(x_train, y_train, model, scaler):
    start = torch.cat((x_train[-1][1:], y_train[-1].unsqueeze(1)), 0) 
    predictions = []
    for i in range(365):
        y_pred = model(start.unsqueeze(1))
        predictions.append(y_pred)
        start = torch.cat((start[1:], y_pred[-1].unsqueeze(1)), 0) 

    daily_predict = []
    for pred in predictions:
        trans = scaler.inverse_transform(pred.detach().numpy())
        daily_predict.append(trans)

    monthly_predict = {}
    for i in range(12):
        s = sum(c.days[:i])
        offset = c.days[i]
        cnt = sum(daily_predict[s:s+offset])
        monthly_predict[c.Months[i]] = cnt[0][0].item()


    return monthly_predict
