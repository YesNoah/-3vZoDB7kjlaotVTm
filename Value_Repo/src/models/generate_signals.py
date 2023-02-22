
import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta
from .train_model import nnmodel
import numpy as np

def MACD_Strategy(df, risk):
    MACD_Buy=[]
    MACD_Sell=[]
    position=False

    for i in range(0, len(df)):
        if df['MACD_12_26_9'][i] > df['MACDs_12_26_9'][i] :
            MACD_Sell.append(np.nan)
            if position ==False:
                MACD_Buy.append(df['yhat'][i])
                position=True
            else:
                MACD_Buy.append(np.nan)
        elif df['MACD_12_26_9'][i] < df['MACDs_12_26_9'][i] :
            MACD_Buy.append(np.nan)
            if position == True:
                MACD_Sell.append(df['yhat'][i])
                position=False
            else:
                MACD_Sell.append(np.nan)
        elif position == True and df['yhat'][i] < MACD_Buy[-1] * (1 - risk):
            MACD_Sell.append(df["yhat"][i])
            MACD_Buy.append(np.nan)
            position = False
        elif position == True and df['yhat'][i] < df['yhat'][i - 1] * (1 - risk):
            MACD_Sell.append(df["yhat"][i])
            MACD_Buy.append(np.nan)
            position = False
        else:
            MACD_Buy.append(np.nan)
            MACD_Sell.append(np.nan)

    df['MACD_Buy_Signal_price'] = MACD_Buy
    df['MACD_Sell_Signal_price'] = MACD_Sell

def MACD_color(data):
    MACD_color = [False]
    for i in range(1, len(data)):
        if data['MACDh_12_26_9'][i] > data['MACDh_12_26_9'][i-1]:
            MACD_color.append(True)
        else:
            MACD_color.append(False)
    return MACD_color

def buysell(result):
    macd = ta.macd(result['yhat'])
    result_ = pd.concat([result, macd], axis=1).reindex(result.index)
    MACD_strategy = MACD_Strategy(result_, 0.025)
    result_['positive'] = MACD_color(result_)
    result_.loc[:, 'ds'] = pd.to_datetime(result_['ds'])
    result_.set_index('ds', inplace=True)
    
    plt.rcParams.update({'font.size': 10})
    fig, ax1 = plt.subplots(figsize=(14,8))
    fig.suptitle('MSFT', fontsize=10, backgroundcolor='blue', color='white')
    ax1 = plt.subplot2grid((14, 8), (0, 0), rowspan=8, colspan=14)
    ax2 = plt.subplot2grid((14, 12), (10, 0), rowspan=6, colspan=14)
    ax1.set_ylabel('Price')
    ax1.plot('yhat',data=result_, label='Close Price', linewidth=0.5, color='blue')
    ax1.scatter(result_.index, result_['MACD_Buy_Signal_price'], color='green', marker='^', alpha=1)
    ax1.scatter(result_.index, result_['MACD_Sell_Signal_price'], color='red', marker='v', alpha=1)
    ax1.legend()
    ax1.grid()
    ax1.set_xlabel('Date', fontsize=8)

    ax2.set_ylabel('MACD', fontsize=8)
    ax2.plot('MACD_12_26_9', data=result_, label='MACD', linewidth=0.5, color='blue')
    ax2.plot('MACDs_12_26_9', data=result_, label='signal', linewidth=0.5, color='red')
    ax2.bar(result_.index,'MACDh_12_26_9', data=result_, label='Volume',width=1,alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax2.grid()
    plt.savefig('reports\\figures\\MACD_BUY_SELL.png', bbox_inches='tight')
    