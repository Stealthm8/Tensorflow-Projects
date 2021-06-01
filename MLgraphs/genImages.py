import pathlib
from pykalman import KalmanFilter
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import math
from math import sqrt
import statsmodels.api as sm
import datetime
from datetime import date
import mplfinance as mpf
import yfinance as yf
yf.pdr_override()

def KalmanFilterAverage(x):
    kf = KalmanFilter(transition_matrices = [1],
    observation_matrices = [1],
    initial_state_mean = 0,
    initial_state_covariance = 1,
    observation_covariance=1,
    transition_covariance=.01)
    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    return state_means

def KalmanFilterRegression(x,y):
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
    initial_state_mean=[0,0],
    initial_state_covariance=np.ones((2, 2)),
    transition_matrices=np.eye(2),
    observation_matrices=obs_mat,
    observation_covariance=2,
    transition_covariance=trans_cov)
    state_means, state_covs = kf.filter(y.values)
    return state_means

def plot_indicator(startDate, endDate, ticker1, ticker2, up, down):
    newData = pdr.get_data_yahoo([ticker1, ticker2], start = startDate, end = endDate)
    index = 0

    place = 0
    mc = mpf.make_marketcolors(up='blue',down='black')
    s  = mpf.make_mpf_style(marketcolors=mc)

    firstPrice = newData["Adj Close"][ticker1]
    secondPrice = newData["Adj Close"][ticker2]
    df1 = pd.DataFrame({'secondPrice':secondPrice,'firstPrice':firstPrice})
    df1.index = pd.to_datetime(df1.index)
    state_means = KalmanFilterRegression(KalmanFilterAverage(firstPrice),KalmanFilterAverage(secondPrice))
    df1['hr'] = - state_means[:,0]
    df1['spread'] = df1.secondPrice + (df1.firstPrice * df1.hr)
    meanSpread = df1.spread.rolling(window=20).mean()
    stdSpread = df1.spread.rolling(window=20).std()
    upperSpread = meanSpread + (2 * stdSpread)
    lowerSpread = meanSpread - (2 * stdSpread)
    df2 = pd.DataFrame({'Open':df1['spread'].shift(periods=1),'High':df1['spread'],'Low':df1['spread'],'Close':df1['spread'],'Volume':df1['spread'],'UpperB':upperSpread,'LowerB':lowerSpread})
    df2 = df2[20:]

    for  i in range(len(df2)-11):
    	openPrice = df2.iloc[i+10]['Open']
    	closePrice = df2.iloc[i+10]['Close']
    	place = i+1+index
    	fileName = "output" + str(place) + ".png"

    	if(openPrice < closePrice):
    		newpath = pathlib.Path(up) / fileName
    	else:
    		newpath = pathlib.Path(down) / fileName

    	mpf.plot(df2[i:i+10], type='candle', style=s, figsize=(3,3), axisoff=True, savefig=newpath, addplot=mpf.make_addplot(df2[i:i+10][['UpperB', 'LowerB']],secondary_y=False,color='orange'))
    print(place)

endDate = date.today() - datetime.timedelta(days = 0 * 365)
startDate = date.today() - datetime.timedelta(days = 1 * 365)

#plot_indicator(startDate, endDate, 'NQ=F', 'ES=F', "C:/Users/elias/Desktop/images/NQ_ES/UP", "C:/Users/elias/Desktop/images/NQ_ES/DOWN")
#plot_indicator(startDate, endDate, 'YM=F', 'RTY=F', "C:/Users/elias/Desktop/images/YM_RTY/UP", "C:/Users/elias/Desktop/images/YM_RTY/DOWN")
plot_indicator(startDate, endDate, 'GC=F', 'SI=F', "C:/Users/elias/Desktop/images/GC_SI/UP", "C:/Users/elias/Desktop/images/GC_SI/DOWN")
#plot_indicator(startDate, endDate, 'ZT=F', 'ZB=F', "C:/Users/elias/Desktop/images/ZT_ZN/UP", "C:/Users/elias/Desktop/images/ZT_ZN/DOWN")
