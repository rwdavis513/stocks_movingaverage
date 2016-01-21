__author__ = 'rwdavis513'


from moving_average import StockData
from matplotlib import pyplot as plt
import pandas as pd

# Assumptions
    # The Long Term Trend is less than the short term volatility

# Buy Signals
    # Moving Average CrossOvers
    # Settings: MA window size

# Sell Signals
    # Where to start stoploss (risk)
    #   How to calculate baseline variation (mrange, fit residuals, etc)
    # When to increase stoploss and by how much
    #   Shift based on max, or 95% quantile

class Optimizer(object):
    # Write algo to optimize all the settings (Stop Loss point, Moving Average Cross overs)
    # How to identify the types of markets?

    def __init__(self):
        self.sd = StockData()
        self.systemScores = pd.DataFrame(columns=['Symbol','near_ma','far_ma','Expectancy','R-StdDev','SQN'])

    def calc_factors_trade(self, symbol='MMM', near_ma=20, far_ma=80):
        self.settings = {'near_ma':near_ma, 'far_ma':far_ma}
        self.sd.calc_ma_stats(symbol,near_ma,far_ma)
        self.sd.calc_ewma(symbol)
        self.sd.calc_ewma_residuals(symbol)
        output2 = self.sd.calc_moving_range(symbol)
        self.sd.trade(symbol,near_ma,far_ma)

    def plot_results(self, symbol):
        near_ma = self.settings['near_ma']
        far_ma =  self.settings['far_ma']
        #ax = self.sd.data[[symbol, symbol + '_minus_ewma_res',symbol + '_recent_max', symbol + '_stop_loss']].plot()
        ax = self.sd.data[[symbol, symbol + '_ma_' + str(near_ma), symbol + '_ma_' + str(far_ma), symbol + '_stop_loss']].plot()
        for i in range(self.sd.order_history.shape[0]):
            ax.axvline(self.sd.order_history['Purchase Date'][i], color='red', linewidth=2)
            ax.axvline(self.sd.order_history['Sell Date'][i], color='green',linewidth=2)
        #fig = plt.figure()
        if self.sd.order_history.shape[0] >= 0:   # Need to have at least one row
            plt.hist(self.sd.order_history['Rmultiple'])

    def score(self):
        symbol = self.sd.order_history_symbol
        expectancy = self.sd.order_history['Rmultiple'].mean()
        RstdDev = self.sd.order_history['Rmultiple'].std()
        SQN = expectancy / RstdDev
        scoresNumbers = list(map(lambda x: round(x,3), [expectancy,RstdDev,SQN]))

        near_ma = self.settings['near_ma']
        far_ma = self.settings['far_ma']
        scores = [symbol,near_ma, far_ma, scoresNumbers[0], scoresNumbers[1], scoresNumbers[2]]
        columns = ['Symbol', 'near_ma', 'far_ma', 'Expectancy', 'R-StdDev', 'SQN']
        a = zip(columns,scores)
        #scoresDict = {}
        #for i in range(len(columns)):
        #    scoresDict[columns[i]] = scores[i]
        scoresDict = {k: v for k, v in a}

        scoresDF = pd.DataFrame(data=scoresDict, index=[0])
        self.systemScores = pd.concat([self.systemScores, scoresDF], ignore_index=True)
        #print(scoresDict)

    def run(self, symbol, near_range=[10,20,30], far_range=[30,60,90], plot_results=False):
        for near_ma in near_range:
            for far_ma in far_range:
                print("Running for Near Moving Average=" + str(near_ma) + " and Far Moving Average=" + str(far_ma))
                self.calc_factors_trade(symbol,near_ma,far_ma)
                if plot_results: self.plot_results(symbol)
                self.score()

if __name__ == "__main__":

    op = Optimizer()
    op.run('ACT')
    print(op.systemScores)
    op.systemScores.to_csv('MovingAverageComparison_ACT.csv')

# Next Steps:
    # ReFactor Moving Average.py so the trading logistics are separate from the algo
    #    Update the algo so it runs on a list of stocks rather than just an individual one.

    # Add Position Sizing Algo to the system