__author__ = 'rwdavis513'

import pandas as pd

class Portfolio(object):
# Class to keep track of the current equity and holdings for each account.
#     Total Equity
#

    def __init__(self, initial_equity, equity_currency='dollars'):
        if not initial_equity >= 0 and initial_equity <= 10000000000000:
            print("Error: Enter a valid amount")
        self.cash_equity = initial_equity
        self.equity_currency = equity_currency




# Addition for a Position Sizing Strategy:
#     Per Order:
#          Percent of equity risked (0-100%)
#          Calculate number of shares to purchase for each stock based on stop loss
#               C = P / R
#          Update the equity based on the allocation