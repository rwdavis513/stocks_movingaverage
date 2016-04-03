__author__ = 'rwdavis513'

import pandas as pd
import unittest
import matplotlib.pyplot as plt
import json

# Correlations
   # Run the correlation across a variety of timeframes, to see how they change. Look for stocks that used to be highly correlated and are no longer that way.


#######################
    # 2/3/16 : Next Steps: Filter to a delta in the Rsquared > 0.5  (Highlight stocks which went from not correlating to correlating).
    #          Filter to top 3 stocks with 2 comparisons per stock(6 comparisons across 3x3 matrix). (It's not really too important how the filtering is done, just get something to work for now.)
    #          Create a DBO
    #          Graphs to confirm:
               #     Overlay Plot of time trend with before / after trend line (x3)
               #     Correlations before (3x) and after (x3)

class CorrelationAnalysis(object):
    # Data Analysis Ideas:
    #    Stock Comparisons:
    #        Changes TimeFrameA vs TimeFrameB: Correlation changes, delta changes, etc
    #        Shift in one Stock, but not in another
    #        Highly correlated (or not correlated) stocks across different industries/sectors. Surprising correlations or not correlations.
    #    Individual Stock News:
    #       Trends TimeframeA vs TimeFrameB: Trending up, down or sideways
    #       Changes in volatility
    #    Spreads - Stock A and B are always correlated, check the spread now
    #
    # Create Stock purchase setup and give recommendations
    #    Analyze Other people's reports and build a framework to capture/reproduce their analysis
    #        Value investing, technical analysis

    def __init__(self):
        self.load_data()
        self.time_period1 = 90
        self.time_period2 = 90
        self.analysis_summary = {}
        self.analyze_data()
        print(self.analysis_summary)
        with open('analysis_summary','w') as f:
            f.write(json.dumps(self.analysis_summary))


    def load_data(self):
        self.data = pd.read_csv('rawdata/csv/allStocks.csv', index_col=0)

    def analyze_data(self):
        # Compare stats between timeframes

        end_interval1 = self.data.shape[0]
        start_interval1 = end_interval1 - self.time_period1
        self.dc1, self.dchigh1 = self.analyze_sub(start_interval1,end_interval1)

        end_interval2 = start_interval1 - 1
        start_interval2 = end_interval2 - self.time_period2
        self.dc2, self.dchigh2 = self.analyze_sub(start_interval2,end_interval2)

        self.common_cols, self.not_common_cols1, self.not_common_cols2 = self.compare_corr(self.dchigh1, self.dchigh2)
        self.calc_delta()
        self.intervals = (start_interval1,end_interval1,start_interval2,end_interval2)
        self.create_analysis_summary()

    def compare_corr(self,dchigh1,dchigh2):
        common_cols = list(set(dchigh1.keys()) and set(dchigh2.keys()))
        not_common_cols1 = [col for col in dchigh1.keys() if
                                    col not in common_cols]
        not_common_cols2 = [col for col in dchigh2.keys() if
                                    col not in common_cols]
        return (common_cols, not_common_cols1, not_common_cols2)

    def calc_delta(self):
        cols = self.not_common_cols1      # Create a list of columns which used to be significant but are not any longer.
        dc_not_common = {}
        for col in cols:
            dc_not_common[col] = self.dchigh1[col]
        self.dc_not_common_df = pd.DataFrame(data=dc_not_common).stack()
        self.dc_not_common_df.sort(ascending=False)

    def create_analysis_summary(self):
        corr_list = self.dc_not_common_df[:5]   # Take the top 5 correlations
        #print(corr_list)
        for key1,key2 in corr_list.index:
            self.analysis_summary[key1] = {key2:['Rsquared',corr_list[(key1,key2)],self.intervals[0],self.intervals[1]]}   # Create the data object model
                                                #Relationship, Value, Start Interval, End Interval

    def analyze_sub(self, start_interval, end_interval):
        #Search through different time intervals to find changes in correlations
        #print("Analyzing data from "+str(start_interval) + " to " + str(end_interval))
        data_subset = self.data[start_interval:end_interval]
        #print(data_subset.head())
        return self.analyze_df(data_subset)

    def analyze_df(self, data_subset):
        # Return a subset of interesting correlations.  How do I want to return this data? Flatten the df?
        data_corr = data_subset.corr()
        return (data_corr, self.filter_corr(data_corr))

    def filter_corr(self, data_corr):
        corr_dict = {}
        for col in data_corr.columns:
            high_corr_series = data_corr[col][data_corr[col] >= 0.95]
            high_corr_series = high_corr_series[high_corr_series <> 1]
            if len(high_corr_series) > 0:
                corr_dict[col] = high_corr_series  # Drop correlations to itself
        #num_corrgt95 = [corr_dict[x] for x in corr_dict.keys()]
        return corr_dict

    def plot_corr_comparison(self):
        cols = self.not_common_cols1  # List of symbols which correlated in the first timeframe but didn't correlate in the second
        fig, ax = plt.subplots(1,2)

        for key1, key2 in self.dc_not_common_df[:1].index:
            self.data[key1].plot(ax=ax[0])
            self.data[key2].plot(ax=ax[0], secondary_y=True)
            ax[1].scatter(self.data[key1], self.data[key2])
            #fig = ax.get_figure()
            #plt.close(fig)
        plt.savefig(key1 + " vs " + key2 + " comparison.png")

class TestCorrelationAnalysis(unittest.TestCase):

    def setUp(self):

        self.ca = CorrelationAnalysis()

    def test_analyze(self):
        self.ca.analyze()


class ShiftTrendAnalysis(object):

    def __init__(self):
        self.load_data()

    def load_data(self):
        self.data = pd.read_csv('rawdata/csv/allStocks.csv', index_col=0)

    def analyze_data(self):
        pass

if __name__ == "__main__":

    ca = CorrelationAnalysis()
    c1, nc1, nc2 = ca.compare_corr(ca.dchigh1, ca.dchigh2)
    ca.plot_corr_comparison()

# Summary Statistics
   # Across different timeframes: (years -> months -> weeks -> daily)
   # Trends : Market -> Sector -> Industry -> Stocks
   # Shifts : Market -> Sector -> Industry -> Stocks

# Pattern Analysis and identification
   # Look for stock patterns across a variety of time frames.

