from sklearn.preprocessing import MinMaxScaler,StandardScaler
import time as tm
import datetime as dt
from yahoo_fin import stock_info as yf
import numpy as np
from models.Lstm.lstm import lstm_main
from models.bilstm.bilstm import bi_lstm_main
#from models.arima.arima import arima_main
from graphs.graphs import make_graph
import matplotlib.pyplot as plt
import pandas as pd

def get_time_interval(s,e):
    date_start = (dt.date.today() - dt.timedelta(days=s)).strftime('%Y-%m-%d')
    date_end = (dt.date.today() - dt.timedelta(days=e)).strftime('%Y-%m-%d')
    return date_start, date_end

def get_clean_data(stock,s,e):
    date_start,date_end = get_time_interval(s,e)
    init_df = yf.get_data(
    stock, 
    start_date=date_end, 
    end_date=date_start,
    interval='1d')

    init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
    init_df['date'] = init_df.index
    return init_df,date_start,date_end


def calculateStandardDiviation(hist, start):
    len_hist = len(hist)
    epochs = len(hist[0].history['loss'])
    xx = [ x + start for x in range(1,epochs+1)]
    yerr = []
    sd = []
    for x in range(0, epochs):
        boo = []
        for y in range(0, len_hist):
            boo.append(hist[y].history['loss'][x])
        yerr.append(sum(boo)/len(boo))
        sd.append(np.std(boo))
    xx = np.array(xx)
    yerr = np.array(yerr)
    sd = np.array(sd)
    return xx,yerr,sd

def main():
    stock = 'AAPL'
    scaler = MinMaxScaler()
    data,date_start,date_end = get_clean_data(stock,4,1000)
    data_visualisation = data.copy()
    data['close'] = scaler.fit_transform(np.expand_dims(data['close'].values, axis=1))
    raw_seq = data['close']

    #input is data - pred_days - runs
    
    scaled_lstm,lstm_history =  lstm_main(raw_seq,3,3,False)
    descaled_lstm = []
    for avg in scaled_lstm:
        descaled_lstm.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])

    x_lstm,yerr_lstm,sd_lstm = calculateStandardDiviation(lstm_history, 0)
    plt.errorbar(x_lstm,yerr_lstm,sd_lstm,linestyle="solid",marker='^',label="LSTM")
    
    
    scaled_lstm_attention,lstm_attention_history =  lstm_main(raw_seq,3,3,True)
    descaled_lstm_attention = []
    for avg in scaled_lstm_attention:
        descaled_lstm_attention.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])

    x_lstm_attention,yerr_lstm_attention,sd_lstm_attention = calculateStandardDiviation(lstm_attention_history, 0.25)
    plt.errorbar(x_lstm_attention,yerr_lstm_attention,sd_lstm_attention,linestyle="solid",marker='^', label="LSTM-A")
    
    scaled_bi_lstm,bi_lstm_history =  bi_lstm_main(raw_seq,3,3,False)
    descaled_bi_lstm = []
    for avg in scaled_bi_lstm:
        descaled_bi_lstm.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])

    x_bi_lstm,yerr_bi_lstm,sd_bi_lstm = calculateStandardDiviation(bi_lstm_history, 0.5)
    plt.errorbar(x_bi_lstm,yerr_bi_lstm,sd_bi_lstm,linestyle="solid",marker='^',label="BI-LSTM")

    scaled_bi_lstm_attention,bi_lstm_attention_history =  bi_lstm_main(raw_seq,3,3,True)
    descaled_bi_lstm_attention = []
    for avg in scaled_bi_lstm_attention:
        descaled_bi_lstm_attention.append(scaler.inverse_transform(np.array(avg).reshape(-1,1))[0][0])
    
    x_bi_lstm_attention,yerr_bi_lstm_attention,sd_bi_lstm_attention = calculateStandardDiviation(bi_lstm_attention_history, 0.75)
    plt.errorbar(x_bi_lstm_attention,yerr_bi_lstm_attention,sd_bi_lstm_attention,linestyle="solid",marker='^', label="BI-LSTM-A")

    plt.legend()
    plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.title('LSTM')
    # plt.plot(lstm_history.history['loss'], label='LSTM')
    # plt.plot(lstm_attention_history.history['loss'], label='LSTM-A')
    # plt.plot(bi_lstm_history.history['loss'], label='BI-LSTM')
    # plt.plot(bi_lstm_attention_history.history['loss'], label='BI-LSTM-A')
    # plt.legend()
    # plt.show()
    
    #print("[2020-06-06 : 2022-11-20]\n",arima_main(3))
  
    # temp = []
    # temp.append(descaled_lstm)
    # temp.append(descaled_lstm_attention)
    # temp.append(descaled_bi_lstm)
    # temp.append(descaled_bi_lstm_attention)
    # data_real,date_start,date_end = get_clean_data(stock,1,10)
    # values = data_real['close'].values
    # values = values[::-1]
    # temp.append([values[2],values[1],values[0]])

    # make_graph(data_visualisation,temp,stock)
    # writeFile = open("aapl_values.txt", "w")
    # writeFile.write(str(date_end) + " : " + str(date_start) + "\n")

    # writeFile.write("LSTM: \n")
    # writeFile.writelines(str(descaled_lstm) + "\n")

    # writeFile.write("LSTM-A: \n")
    # writeFile.writelines(str(descaled_lstm_attention) + "\n")

    # writeFile.write("BI-LSTM: \n")
    # writeFile.writelines(str(descaled_bi_lstm) + "\n")

    # writeFile.write("BI-LSTM-A: \n")
    # writeFile.writelines(str(descaled_bi_lstm_attention) + "\n")

    # writeFile.close()

    # readFile = open("src/graphs/main.txt", "r")
    # pred = readFile.readlines()
    # print(raw_seq)
    # print(pred)

    # readFile.close()

main()

