__author__ = 'rwdavis513'


from portfolio import Portfolio
import unittest
import pandas as pd
import numpy as np


class RmultipleDistribution(object):

    def __init__(self, expectancy=0.6, Rstddev=1.0):
        # How could I extend this to include non-normal distribution?
        self.expectancy = expectancy
        self.Rstddev = Rstddev

    def trade(self):

        return round(np.random.normal(self.expectancy, self.Rstddev),3)



class PositionSizingAlgo(object):
    # Run through X number of trades given a specific position sizing plan (%equity risk per trade)
    # Returns a dataframe of trades including the total equity
        # output: DataFrame: Trade#,Rmultiple,Cash Equity

    def __init__(self, initial_equity=100000, pct_equity_risk=0.01, num_trades=10, expectancy=0.5,RstdDev=2.0):
        self.cash_equity = initial_equity
        self.Rdist = RmultipleDistribution(expectancy, RstdDev)  # Need to update to be non-normal distribution
        self.pct_equity_risk = pct_equity_risk
        self.num_trades = num_trades
        self.order_history = []


    def simulate(self):
        for i in range(self.num_trades):
            risk = self.pct_equity_risk*self.cash_equity
            Rmultiple = self.Rdist.trade()
            profit_loss = Rmultiple*risk
            self.cash_equity += round(profit_loss,2)
            self.order_history.append((i, Rmultiple, profit_loss, self.cash_equity))

        colnames = ['Trade Number','Rmultiple','Profit or Loss','Cash Equity']
        self.order_history_df = pd.DataFrame(data=self.order_history, columns=colnames)
        self.order_history_df.to_csv('Simulate_Position Sizing.csv')
        return self.order_history_df

class TestPositionSizingAlgo(unittest.TestCase):

    def setUp(self):
        initial_equity = 200000
        self.num_trades = 1000
        self.pct_equity_risk = 0.01
        self.PSA = PositionSizingAlgo(initial_equity, self.pct_equity_risk,self.num_trades)


    def test_position_sizing(self):
        order_history = self.PSA.simulate()
        print(list(order_history.columns))
        self.assertEqual(['Trade Number','Rmultiple','Profit or Loss','Cash Equity'], list(order_history.columns))  # Verify the correct columns
        self.assertEqual(order_history.shape[0],self.num_trades) # Verify the number of trades
        #order_history['ProfitRisk'] =

if __name__ == "__main__":

    #unittest.main()

    initial_equity = 100000
    num_trades = 100
    pct_equity_risk = 0.01
    for i in range(10):
        PSA = PositionSizingAlgo(initial_equity, pct_equity_risk, num_trades,0.5,1.0)
        order_history = PSA.simulate()
        order_history['Cash Equity'].plot()

# Next Steps:
#   Add an optimization class which will simulate different pct_equity_risk (position sizing strategies)
#   Modify the Rdistribution class to include non-normal distribution (bimodal with X % losses and Y % gains#   Integrate with portfolio class to calculate the number of shares to purchase, etc (Is this really needed?)