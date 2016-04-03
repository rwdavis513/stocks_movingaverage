

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from portfolio import Portfolio

class StockData(object):
# I need to re-factor this code.

    def __init__(self,stockdata_csv='rawdata/csv/allStocks.csv'):
        self.data = pd.read_csv(stockdata_csv, index_col=0)
        self.data.set_index(pd.to_datetime(self.data.index), inplace=True)
        self.portfolio = Portfolio(100000, pct_equity_risk=0.01)

    def graph_data(self,symbol='MMM'):
        fig = plt.figure()
        ax = self.data.plot(y=[symbol])
        ax.set_ylabel('Open')
        plt.show()

    def calc_ewma_manual(self, symbol='MMM', l=0.1):
        timeSeries = self.data[symbol]

        for i in range(timeSeries.count()):
            if not np.isnan(timeSeries[i]):
                if i == 0:
                    ewma = [timeSeries[i]]
                else:
                    ewma.append(timeSeries[i]*l + (1-l)*ewma[i-1])
            else:
                if i == 0:
                    ewma = [np.min(timeSeries[:500])]
                else:
                    ewma.append(ewma[i-1])
            print(timeSeries[i], ewma[i])
        return ewma

    def calc_ewma(self,symbol='MMM',alpha=0.1):
        com = (1-alpha)/alpha
        self.data[symbol + '_ewma'] = pd.ewma(self.data[symbol],com=com)

    def calc_ewma_residuals(self,symbol='MMM'):
        if not symbol + "_ewma" in self.data.columns:
            raise Exception("Error: Call .calc_ewma first.")

        self.data[symbol + "_ewma_res"] = self.data[symbol] - self.data[symbol + "_ewma"]
        window_size = 30
        self.data[symbol + "_minus_ewma_res"] = self.data[symbol] - map(lambda x: 3*abs(x), pd.rolling_mean(self.data[symbol + '_ewma_res'],window_size, min_periods=0))
        self.data[symbol + "_minus_ewma_res"] = pd.rolling_mean(self.data[symbol + "_minus_ewma_res"],window_size,min_periods=0)

    #def _movingaverage(self, interval, window_size):
    #    window = np.ones(int(window_size)) / float(window_size)
    #    return np.convolve(interval, window, 'same')

    #def calc_movingaverage(self,symbol='MMM',window_size=20):
    #    self.data[symbol + "_ma"] = self._movingaverage(self.data[symbol],window_size)

    def calc_ma(self, symbol='MMM', window_size=20):
        self.data[symbol + "_ma_" + str(window_size)] = pd.rolling_mean(self.data[symbol], window_size, min_periods=0)

    def calc_moving_range(self,symbol='MMM',window_size=40):
        timeSeries = self.data[symbol]
        output = np.ones(window_size)*np.min(timeSeries[:500])  # set first 20 values equal to the min of the first 500 to avoid issues with nan

        for i in range(window_size-1,timeSeries.shape[0]-1):
            if not np.isnan(np.min(timeSeries[i-window_size:i])):
                output = np.append(output,np.max(timeSeries[i-window_size:i])-np.min(timeSeries[i-window_size:i]))
            else:
                output = np.append(output,output[i-1])
        output2 = pd.Series(output, index=timeSeries.index)

        self.data[symbol + "_mrange"] = output2
        window_size = 30
        self.data[symbol + "_minus_mrange"] = pd.rolling_mean(self.data[symbol] - self.data[symbol + '_mrange'], window_size,min_periods=0)
        return output2

    def calc_ma_stats(self,symbol='MMM',near_ma=20,far_ma=60):
        near_col_name = symbol + "_ma_" + str(near_ma)
        far_col_name = symbol + "_ma_" + str(far_ma)
        if not near_col_name in self.data.columns:
            self.calc_ma(symbol, near_ma)
        if not far_col_name in self.data.columns:
            self.calc_ma(symbol, far_ma)

        delta_col_name = symbol + "_ma_" + str(near_ma) + "-ma_" + str(far_ma)
        delta = self.data[near_col_name] - self.data[far_col_name]
        self.data[delta_col_name] = delta.round(1)

        for col_name in [near_col_name,far_col_name]:
            self.data[col_name + '_grad'] = pd.rolling_mean(np.gradient(self.data[col_name],100),10)

    def plot_crossovers(self,symbol='MMM',near_ma=20,far_ma=60):
        if not symbol in self.data.columns:
            raise Exception(symbol + " not found within the dataset")
        near_col_name = symbol + "_ma_" + str(near_ma)
        far_col_name = symbol + "_ma_" + str(far_ma)
        delta_col_name = symbol + "_ma_" + str(near_ma) + "-ma_" + str(far_ma)
        crossOverPts = self.data[delta_col_name][self.data[delta_col_name] == 0]

        ax = self.data[[symbol,near_col_name,far_col_name]].plot()
        for crossOverPoint in crossOverPts.index:
            ax.axvline(crossOverPoint,color='grey')

        # Plot the Gradient
        #self.data[col_name + '_grad'].plot(secondary_y=True)

        #Buy when the gradient of the near term moving average is positive and when it is crossing the long term average
        entry_points = self.data[symbol][(self.data[col_name+'_grad'] > 0) & (self.data[delta_col_name] == 0)]
        for entry_point in entry_points.index:
            ax.axvline(entry_point, color='red',linewidth=3)

        #self.data[symbol + '_entry'] = entry_points*np.ones(entry_points.shape[0])

        plt.show()
        return crossOverPts

    def trade(self, symbol='MMM',near_ma=20,far_ma=60):
        if not symbol + '_ma_' + str(near_ma) in self.data.columns:
            raise Exception("Please run .calc_ma_stats first")
        #timeSeries = self.data[symbol].dropna()
        col_name = symbol + '_ma_' + str(near_ma)
        delta_col_name = symbol + "_ma_" + str(near_ma) + "-ma_" + str(far_ma)
        order_history = []
        self.recent_max_list = []
        self.stop_loss_list = []
        holding_stock = False
        recent_max = 0

        for i in range(self.data[symbol].shape[0]):
            entry_criteria = self.data[col_name +'_grad'][i] > 0 and self.data[delta_col_name][i] == 0

            if recent_max < self.data[symbol][i] and not np.isnan(self.data[symbol][i]):
                recent_max = self.data[symbol][i]

            if not holding_stock:
                if entry_criteria and not np.isnan(self.data[symbol][i]):
                    holding_stock = True
                    purchase_price = self.data[symbol][i]
                    purchase_date = self.data.index[i]
                    recent_max = purchase_price
                    delta_to_max = np.std(self.data[symbol + '_ewma_res'][i-50:i])*3
                    initial_stop_loss = purchase_price - delta_to_max    # initial stop loss is minus_ewma_res
                    self.stop_loss_list.append(initial_stop_loss)
                    self.portfolio.buy_stock(symbol,purchase_price, initial_stop_loss)
                    entry_criteria = False
                    print(self.portfolio.current_holdings)
                    #print("PURCHASE: "+ purchase_date.strftime('%m/%d/%y') + "   " + str(purchase_price))
                else:
                    self.stop_loss_list.append(np.nan)
            else:
                if np.std(self.data[symbol + '_ewma_res'][i-50:i])*3 < delta_to_max:
                    delta_to_max = np.std(self.data[symbol + '_ewma_res'][i-50:i])*3   # allow the delta from the max to get smaller not larger
                stop_loss = recent_max - delta_to_max
                exit_criteria = self.data[symbol][i] < stop_loss
                self.stop_loss_list.append(stop_loss)

                #exit_criteria = self.data[symbol][i] < self.data[symbol + '_minus_ewma_res'][i-1]

                #print(self.data.index[i].strftime('%m/%d/%y') + "   " + str(self.data[symbol][i]) + "  max: " + str(recent_max) + " Range: " + str(self.data[symbol + '_mrange'][i]))
                if exit_criteria and not np.isnan(self.data[symbol][i]):
                    holding_stock = False   #SELL!
                    sell_price = self.data[symbol][i]
                    sell_date = self.data.index[i]
                    num_shares = self.portfolio.current_holdings[symbol]
                    self.portfolio.sell_stock(symbol, sell_price)
                    order_history.append((purchase_date, purchase_price, sell_date, sell_price, initial_stop_loss, num_shares, self.portfolio.cash_equity))
                    #print("       SELL: " + sell_date.strftime('%m/%d/%y') + "   Price:" + str(sell_price) + "   MAX:" + str(recent_max) + "  Range:" + str(self.data[symbol + '_mrange'][i]) + "  Delta:" + str(recent_max - self.data[symbol + '_mrange'][i]))
                    purchase_price = np.nan
                    sell_price = np.nan
                    print(self.portfolio.current_holdings)
                    exit_criteria = False
                    #ax.axvline(self.data.index[i], color='blue')

            self.recent_max_list.append(recent_max)
        self.data[symbol + '_stop_loss'] = pd.Series(self.stop_loss_list, index=self.data.index)
        self.data[symbol + '_recent_max'] = pd.Series(self.recent_max_list,index=self.data.index)
        #self.data[symbol + '_recent_max_minus_mrange'] = self.data[symbol + '_recent_max'] - self.data[symbol + '_mrange']

        order_history_df = pd.DataFrame(order_history, columns=['Purchase Date','Purchase Price','Sell Date','Sell Price','Stop Loss', 'Number of Shares','Total Cash Equity'])
        order_history_df['Symbol'] = symbol
        order_history_df['Profit_Loss_share'] = order_history_df['Sell Price'] - order_history_df['Purchase Price']
        order_history_df['Risk'] = order_history_df['Purchase Price'] - order_history_df['Stop Loss']
        order_history_df['Rmultiple'] = order_history_df['Profit_Loss_share'] / order_history_df['Risk']
        if hasattr(self,'order_history'):
            self.order_history = pd.concat([self.order_history, order_history_df],ignore_index=True)
        else:
            self.order_history = order_history_df

    def calc_factors_trade(self, symbol='MMM'):
        self.calc_ma_stats(symbol, self.settings['near_ma'], self.settings['far_ma'])
        self.calc_ewma(symbol)
        self.calc_ewma_residuals(symbol)
        #output2 = self.calc_moving_range(symbol)
        self.trade(symbol, self.settings['near_ma'], self.settings['far_ma'])  # Updates order history

    def backtest(self, symbol_list=['MMM','ACT'], near_ma=10, far_ma=90, save=False):
        self.symbol_list = symbol_list
        self.settings = {'near_ma': near_ma, 'far_ma': far_ma}
        for symbol in symbol_list:
            print("BackTesting " + symbol)
            self.calc_factors_trade(symbol)
        if save:
            try:
                self.order_history.to_csv('Order_History.csv')  # Add numbering
            except IOError:
                print("IOError: Order History Not saved.")
        return self.calc_score()

    def show_symbol_columns(self,symbol='MMM'):
        for col in self.data.columns:
            if symbol in col:
                print(col)

    def plot_results(self, symbol_list):

        near_ma = self.settings['near_ma']
        far_ma = self.settings['far_ma']
        # ax = self.data[[symbol, symbol + '_minus_ewma_res',symbol + '_recent_max', symbol + '_stop_loss']].plot()
        for symbol in symbol_list:
            print("Plotting for " + symbol)
            ax = self.data[[symbol, symbol + '_ma_' + str(near_ma),
                            symbol + '_ma_' + str(far_ma), symbol + '_stop_loss']].plot()
            symbol_order_history = self.order_history[self.order_history['Symbol']==symbol]
            for row in symbol_order_history.iterrows():
                try:
                    ax.axvline(row[1]['Purchase Date'], color='red', linewidth=2)
                    ax.axvline(row[1]['Sell Date'], color='green', linewidth=2)
                except KeyError:
                    print("KeyError: No entries found for item.")
                    print(symbol_order_history)

            # fig = plt.figure()

    def calc_score(self,plot_histo=True):
        # symbol = self.sd.order_history_symbol
        if plot_histo and self.order_history.shape[0] >= 0:  # Need to have at least one row
            plt.hist(self.order_history['Rmultiple'])
        expectancy = self.order_history['Rmultiple'].mean()
        RstdDev = self.order_history['Rmultiple'].std()
        SQN = expectancy / RstdDev
        scoresNumbers = list(map(lambda x: round(x, 3), [expectancy, RstdDev, SQN]))

        near_ma = self.settings['near_ma']
        far_ma = self.settings['far_ma']
        scores = ['-'.join(self.symbol_list), near_ma, far_ma, scoresNumbers[0], scoresNumbers[1], scoresNumbers[2]]
        columns = ['Symbols', 'near_ma', 'far_ma', 'Expectancy', 'R-StdDev', 'SQN']
        a = zip(columns, scores)
        # scoresDict = {}
        # for i in range(len(columns)):
        #    scoresDict[columns[i]] = scores[i]
        scoresDict = {k: v for k, v in a}
        self.score = pd.DataFrame(data=scoresDict, index=[0])
        return self.score

if __name__ == "__main__":

    sd = StockData()
    symbol_list = sd.data.sample(5,axis=1).columns   # Take a random sample of x stocks to run the system on
    sd.backtest(symbol_list,10,60,save=True)
    sd.plot_results(symbol_list[[1,4]])
    plt.hist(sd.order_history['Rmultiple'])
    print(sd.score)

    #myStockData.graph_data()
    #sd.calc_ewma()
    #myStockData.data[['MMM','MMM_ewma']].plot()
    #sd.calc_ewma_residuals()
    #myStockData.data['MMM_ewma_res'].plot()

    #output2 = sd.calc_moving_range()

#    col_names = ['MMM']
#    for i in range(10,100,10):
#        sd.calc_ma('MMM',i)
#        col_names.append('MMM_ma_'+str(i))
#    sd.data[col_names].plot()
    #sd.plot_crossovers()
    #sd.calc_ma_stats()

    #sd.trade()

    #ax = sd.data[['MMM', 'MMM_minus_ewma_res','MMM_recent_max','MMM_stop_loss']].plot()
    #ax = sd.data[['MMM', 'MMM_ma_20', 'MMM_ma_60']].plot()
    #for i in range(sd.order_history.shape[0]):
    #    ax.axvline(sd.order_history['Purchase Date'][i], color='red', linewidth=2)
    #    ax.axvline(sd.order_history['Sell Date'][i], color='green',linewidth=2)
    #plt.hist(sd.order_history['Rmultiple'])

    # Thoughts:
         # Write algo to optimize all the settings (Stop Loss point, Moving Average Cross overs)
         # How to identify the types of markets?
         # When to re-factor?