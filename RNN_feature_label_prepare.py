import datetime
import time
import json
import urllib.request
import pandas as pd
import numpy as np
from TimeMachine import creatTimeStamp

def MACD(data):

    short_ave = data['close'].rolling(12).mean()
    long_ave = data['close'].rolling(26).mean()
    diff = short_ave - long_ave
    dea = diff.rolling(9).mean()
    macd = diff - dea
    return macd


def get_coin_data(currencyPair, start_date, end_date, hm_minute):
    '''
    currencyPair input form 'USDT_BCT, 'USDT_STR', ect
    start_date format : '2017-01-01'
    end_date format: '2018-01-01'
    hm_minute: how_many_minute input format should be int, 1 represent 1min ; 60 represent 1hour
    '''
    start_stamp = str(creatTimeStamp(start_date))
    end_stamp = str(creatTimeStamp(end_date))
    period = str(hm_minute * 60)

    url = 'https://poloniex.com/public?command=returnChartData&currencyPair=' + currencyPair + '&start=' + start_stamp + '&end=' + end_stamp + '&period=' + period
    data = urllib.request.urlopen(url)
    data = data.read().decode()
    data = json.loads(data)
    data1 = {'date': data['date'], 'close': data['close']}
    data1 = pd.DataFrame(data1)
    dates = np.array(data1['date'], dtype='datetime64[s]')
    data1['date'] = dates
    data1.set_index('date')
    macd = MACD(data1)
    data1['MACD'] = macd
    data1.dropna(axis=0, inplace=True)
    return data1

def creat_features_labels(data, magnitude, lag_window, sample_size=128):
    '''
    the difference between this function and that in patter_collection_CypoCurrency
    is that, for RNN model training, the shape of each one feature should be sample_size, 2)
    instead of (1, 2)
    so here another parameter of sample_size is required.
    i would define defaule value as 128
    so bar will not start at 0 but to start at 128 and then each feature and label
    will be further created by bar += 1
    '''
    label = []
    features = []
    bar = sample_size
    x = len(data) - lag_window - sample_size
    
    while bar < x:
        feature = data[bar - sample_size:bar]
        features.append(feature)
        maxim = max(data['close'][bar+1: bar+lag_window])
        currentP = data['close'][bar]
        highmag = currentP * (1 + magnitude)
        lowmag = currentP * (1 - magnitude)
        minim = min(data['close'][bar+1: bar+lag_window])
        if maxim > highmag:
            label.append(1)
            bar += 1
        elif minim < lowmag:
            label.append(-1)
            bar += 1
        else:
            label.append(0)
            bar += 1

    if len(features) != len(label):
        features = features[:len(label)]


    return features, label
