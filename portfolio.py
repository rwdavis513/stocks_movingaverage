__author__ = 'rwdavis513'

import pandas as pd
import numpy as np
import unittest

class Portfolio(object):
# Class to keep track of the current equity and holdings for each account.
#     Total Equity
# Position Sizing Strategy:
#     Per Order:
#          Percent of equity risked (0-100%)
#          Calculate number of shares to purchase for each stock based on stop loss
#               C = P / R
#          Update the equity based on the allocation


    def __init__(self, initial_equity, equity_currency='dollars',pct_equity_risk=0.01):
        if not initial_equity >= 0 and initial_equity <= 10000000000000:
            print("Error: Enter a valid amount")
        self.cash_equity = initial_equity
        self.equity_currency = equity_currency
        self.pct_equity_risk = pct_equity_risk
        self.current_holdings = {}

    def total_cost(self, num_shares, price, fees=0):
        # Allow for easy addition of fees later
        return num_shares * price

    def calc_volume_to_buy(self, price, stop_loss, pct_equity_risk):
        if np.isnan(pct_equity_risk):
            pct_equity_risk = self.pct_equity_risk
        risk_per_share = price - stop_loss
        num_shares = round((self.cash_equity * pct_equity_risk) / risk_per_share,0)
        if self.total_cost(num_shares, price) >= self.cash_equity:    # If the cost of purchasing the shares is too high then re-calculate num_shares needed
            num_shares = round(self.cash_equity / price, 0)
        return num_shares

    def buy_stock(self, symbol, price, stop_loss, pct_equity_risk=np.nan):
        num_shares = self.calc_volume_to_buy(price, stop_loss, pct_equity_risk)

        self.cash_equity = self.cash_equity - self.total_cost(num_shares, price)
        self.current_holdings[symbol] = num_shares

    def sell_stock(self, symbol, sell_price):
        try:
            num_shares = self.current_holdings[symbol]
        except KeyError:
            print("KeyError: %s was not found in the list of current holdings.".format(symbol))
            return
        self.cash_equity += self.total_cost(num_shares, sell_price)
        del self.current_holdings[symbol]

class TestPortfolio(unittest.TestCase):

    def setUp(self):
        print("In setUp")
        self.po = Portfolio(180000, 0.01)  # $180,000 dollar beginning amount with 1% equity risk per trade
        self.stock_info = {'symbol': 'MMM', 'price': 50, 'stop_loss': 45}
        self.sell_price = 100

    def test_buy_stock(self):
        print("Test buying stock...")
        self.po.buy_stock(*self.stock_info.values())
        self.assertEqual(162000, self.po.cash_equity)
        num_shares = self.po.current_holdings[self.stock_info['symbol']]
        self.assertEqual(360, num_shares)
        self.assertTrue(self.stock_info['symbol'] in self.po.current_holdings.keys())

    def test_sell_stock(self):
        print("Test Selling Stock...")
        self.po.buy_stock(*self.stock_info.values())
        self.po.sell_stock(self.stock_info['symbol'], self.sell_price)
        self.assertEqual(198000,self.po.cash_equity)
        self.assertFalse(self.stock_info['symbol'] in self.po.current_holdings.keys())

if __name__ == "__main__":

    unittest.main()

