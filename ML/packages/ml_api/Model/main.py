import pandas as pd
import numpy as np
# import argparse
from .utils import split_data, plot, train, predict
from .models import LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
import torch 
import torch.nn as nn
import time
from . import constants as c



def make_prediction(*, inputs):

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--file', help='intput file path', type=str, default="./data_daily.csv")

    # args = parser.parse_args()


    # read in data

    data = pd.read_csv('Model/data_daily.csv', dtype= {
    '# Date': str,
    'Receipt_Count': int
    })

    scaler = MinMaxScaler(feature_range=(-1,1))
    receipt = data[['Receipt_Count']]
    receipt['Receipt_Count'] = scaler.fit_transform(receipt['Receipt_Count'].values.reshape(-1,1))

    # split data into batches of length 30 (using the first 29 to predict the 30th number)
    x_train, y_train = split_data(receipt, 30)

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    
    if isinstance(inputs, str):
        m = inputs
    else:
        m = inputs['inputs'][0]['return']


    if m == 'lstm':
        # LSTM model
        model = LSTM(input_dim=c.input_dim, hidden_dim=c.hidden_dim, output_dim=c.output_dim, num_layers=c.num_layers)
        criterion = torch.nn.MSELoss(reduction='mean')
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        # training the LSTM model
        print('Starting training the LSTM Model')
        y_pred_lstm, hist_lstm = train(model, x_train, y_train_lstm, criterion, optimiser)

        # plot training results
        predict1 = pd.DataFrame(scaler.inverse_transform(y_pred_lstm.detach().numpy()))
        original1 = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))
        plot(original1, predict1, hist_lstm, 'lstm_training.png')

        # prediction for next year lstm
        monthly_predicts_lstm = predict(x_train, y_train_lstm, model, scaler)

        print('Prediction Results for Year 2022 using LSTM:')

        for k, v in monthly_predicts_lstm.items():
            print(k, v)

        final_predictions = monthly_predicts_lstm
    else:
        # GRU model
        model2 = GRU(input_dim=c.input_dim, hidden_dim=c.hidden_dim, output_dim=c.output_dim, num_layers=c.num_layers)
        criterion2 = torch.nn.MSELoss(reduction='mean')
        optimiser2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        

        # training the GRU model
        print('Starting training the GRU Model')
        y_pred_gru, hist_gru = train(model2, x_train, y_train_gru, criterion2, optimiser2)

        # plot training results
        predict2 = pd.DataFrame(scaler.inverse_transform(y_pred_gru.detach().numpy()))
        original2 = pd.DataFrame(scaler.inverse_transform(y_train_gru.detach().numpy()))
        plot(original2, predict2, hist_gru, 'gru_training.png')
        

        # prediction for next year gru
        monthly_predicts_gru = predict(x_train, y_train_gru, model2, scaler)

        print('Prediction Results for Year 2022 using GRU:')

        for k, v in monthly_predicts_gru.items():
            print(k, v)
        final_predictions = monthly_predicts_gru

    
    


    

    # torch.save(model, c.LSTM_MODEL_PATH)
    # torch.save(model2, c.GRU_MODEL_PATH)

    

    results = {
            "predictions": [final_predictions],
        }

    return results
















