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

        #print(scoresDict)

    def run(self, symbol_list, near_range=[10,20,30], far_range=[30,60,90], plot_results=False):
        for near_ma in near_range:
            for far_ma in far_range:
                print("Running for Near Moving Average=" + str(near_ma) + " and Far Moving Average=" + str(far_ma))
                score = self.sd.backtest(symbol_list,near_ma,far_ma)
                #if plot_results: self.plot_results(symbol)
                self.systemScores = pd.concat([self.systemScores, score], ignore_index=True)


if __name__ == "__main__":

    op = Optimizer()
    op.run(['ACT'])
    print(op.systemScores)
    op.systemScores.to_csv('MovingAverageComparison.csv')

# Next Steps:
    # ReFactor Moving Average.py so the trading logistics are separate from the algo
    #    Update the algo so it runs on a list of stocks rather than just an individual one.

    # Add Position Sizing Algo to the system