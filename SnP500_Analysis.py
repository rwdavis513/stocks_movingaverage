# -*- coding: utf-8 -*-
"""
Created on Sun Aug 02 17:01:47 2015

@author: Bob
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import urllib2
import numpy as np
import sklearn.decomposition as decomp
os.chdir('C:/Users/Bob/Documents/DataScience/Stocks/recent')


def loadStockList():
    """ Load the list of stocks"""
    
    stocklist = pd.read_csv('constituents.csv')
    list(stocklist)    
    stockData = pd.read_csv('constituents-financials.csv')
    
    list(stockData)
    stockData.info
    stockData.dtypes
    stockData.shape
    pd.pivot_table(stockData,index=['Sector'],values=['Market Cap'],aggfunc=np.sum,margins=True)
    stockData[['Name','Dividend Yield']]
    stockData['HighLowDelta'] = stockData['52 week low'] - stockData['52 week high']

    return (stocklist,stockData)

def yahooFinanceAPI(ticker,fromDate,ToDate,period='d'):
    """ Download stock data from the Yahoo Finance API"""

    if type(fromDate) != datetime.date:
        print("Error fromDate is not a datetime.date variable type.")
        return 
    if type(toDate) != datetime.date:
        print("Error toDate is not a datetime.date variable type.")
        return     
    
    fromDate.year
    #http://ichart.yahoo.com/table.csv?s=BAS.DE&a=0&b=1&c=2000 &d=0&e=31&f=2010&g=w&ignore=.csv
    query = "http://ichart.yahoo.com/table.csv?s=" + ticker + "&a=" + str(fromDate.month) + "&b=" + str(fromDate.day) + "&c=" + str(fromDate.year) + \
                                                              "&d=" + str(toDate.month) + "&e=" + str(toDate.day) + "&f=" + str(toDate.year) + "&g=" + period + "&ignore=.csv"
    print(query)                                                         
    stockHistData = pd.read_csv(query)
    
    return stockHistData

        


fromDate = datetime.date(2014,9,1)
toDate = datetime.date.today()    

(stocklist,curStockData) = loadStockList()
allStockData = pd.DataFrame(index=pd.date_range(fromDate,toDate))

#plt.ioff()

for curStockTicker in stocklist['Symbol']:
    #curStockTicker = stocklist['Symbol'][11]
    try:
        histStockData = yahooFinanceAPI(curStockTicker,fromDate,toDate)
    except urllib2.HTTPError, err:
        print(err)
        print(curStockTicker + " had an error pulling the data.")
    else:
        #histStockData.tail()
        histStockData['Date'] = pd.to_datetime(histStockData['Date'])
        histStockData.set_index(['Date'],inplace=True)
        histStockData['HighLowDelta'] = histStockData['High'] - histStockData['Low']
        
        fig = plt.figure()
        ax = histStockData.plot(y=['Open'],title=curStockTicker)
        ax.set_ylabel('Open')
        plt.savefig(curStockTicker + '.png',format='png')    
        plt.close(fig)
        
        #histStockData.plot(y=['Close'])
        #histStockData.plot(secondary_y=['Volume'])
        
        histStockData.to_csv(curStockTicker + '.csv')
        
        allStockData = allStockData.join(histStockData['Open'])   
        allStockData.rename(columns={'Open':curStockTicker},inplace=True)
    
"""
allStockData.to_csv('allStocks.csv')
#allStockData.shape

allStockDataFilt = allStockData.dropna(how='all').fillna(0)
pca = decomp.PCA(n_components=5)
pca.fit(allStockDataFilt)
allStockDataFilt_t = pca.transform(allStockDataFilt)
allStockDataFilt_t.shape
allStockDataFilt_t = pd.DataFrame(allStockDataFilt_t)
allStockDataFilt_t.plot()
pca.score()
pca.components_

mean = np.mean(allStockDataFilt,axis=0)
normalized = allStockDataFilt - mean
evals, evecs = np.linalg.eig(np.cov(normalized.T))
order = evals.argsort()[::-1]
"""