
from interface import DataRetrievalInterface
import pandas as pd
import numpy as np
import unittest
import datetime
import json
import Quandl
import requests
import math
import os, sys
import settings


def write_json(filename, json_temp):
    with open(filename,'w') as file:
        json.dump(json_temp, file)

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

class DataRetrieval(DataRetrievalInterface):

    def __init__(self, filename="../rawdata/snp.csv"):
        # load the data from file and store as a pandas data frame
        self.df = pd.read_csv(filename)

        if( "snp.csv" in filename ):
            # todo remove this stupid hack
            # Simulate the daily return on all the stocks
            self.true_data = pd.read_csv( filename="../rawdata/all_stock_data.csv" )

            # pick the day that you care about
            

            # add a new column representing the fake daily return
            percent_daily_return = np.random.randn(len(self.df))+np.random.randn(1)
            self.df['daily return'] = self.df['LocalPrice']*percent_daily_return/100



            # fix some of the names
            self.df.rename(columns={'GICSSector':'sector'}, inplace=True)
            self.df.rename(columns={'Ticker':'ticker'}, inplace=True)

class YahooFinanceAPI(object):

    def __init__(self):
        self.api_guide = self.read_json('yahoo_api_guide.json')
        self.load_key_metrics()
        self.outputdir = '../rawdata/csv/'

    def pull(self, ticker, fromDate, toDate, period='d'):
        """ Download stock data from the Yahoo Finance API"""

        if type(fromDate) != datetime.date:
            print("Error fromDate is not a datetime.date variable type.")
            return
        if type(toDate) != datetime.date:
            print("Error toDate is not a datetime.date variable type.")
            return

        # http://ichart.yahoo.com/table.csv?s=BAS.DE&a=0&b=1&c=2000 &d=0&e=31&f=2010&g=w&ignore=.csv
        query = "http://ichart.yahoo.com/table.csv?s=" + ticker + "&a=" + str(fromDate.month) + "&b=" + str(fromDate.day) + "&c=" + str(fromDate.year) + \
                "&d=" + str(toDate.month) + "&e=" + str(toDate.day) + "&f=" + str(toDate.year) + "&g=" + period + "&ignore=.csv"
        print(query)
        stockHistData = pd.read_csv(query)
        stockHistData['symbol'] = ticker
        return stockHistData

    def read_json(self, filename='yahoo_api_guide.json'):
        with open(filename, 'r') as f:
            fj = json.loads(f.read())
            #print(f.read())
        return fj

    def resp_to_json(self, resp):
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception("API call failed with status code: {}".formart(resp.status_code))

    def pull_key_metrics(self, symbol_list):
        step_size = 20
        l = len(symbol_list)
        print(l)
        rounded_length = int(round(l / step_size, 0)*step_size)
        print(rounded_length)
        #rounded_length = 100
        for i in range(0, rounded_length, step_size):
            tempfile = 'yahoo_api_' + str(i) + '-' + str(i+step_size)
            data = self.pull_data(symbol_list[i:i+step_size], self.key_metrics, tempfile)
            if hasattr(self, 'data'):
                self.data = pd.concat([self.data, data])
            else:
                self.data = data
        if rounded_length != l:
            for i in range(rounded_length, l):
                tempfile = 'yahoo_api_' + str(i) + '-' + str(i+step_size)
                data = self.pull_data(symbol_list[i:i+step_size], self.key_metrics, tempfile)
                self.data = pd.concat([self.data, data])

        self.data.to_csv(self.outputdir + 'yahoo_api_alldata.csv')

    def pull_data(self, symbol_list, param_list=['Symbol', 'Ask', 'Earnings/Share'], filename='yahoo_api_data.csv'):
        if not os.path.exists(self.outputdir):
            os.mkdir(self.outputdir)
        outfilename = self.outputdir + filename
        query_str = self.build_query_str(symbol_list, param_list)
        print(query_str)
        resp = requests.get(query_str)
        if resp.status_code == 200:
            with open(outfilename, 'w') as outfile:
                outfile.write(','.join(param_list) + '\n')
                outfile.write(resp.content)
        else:
            raise Exception("Yahoo API returned incomplete status code: " + str(resp.status_code))
        data = pd.read_csv(outfilename)
        return data

    def build_query_str(self, symbol_list, param_list=['Symbol', 'Ask', 'Earnings/Share', 'EPS Estimate Current Year']):
        if type(symbol_list) != list:
            raise Exception('Expected a list instead of ' + str(type(symbol_list)))

        symbol_str = self.api_guide['Symbol'] + '=' + '+'.join(symbol_list)
        arg_str = self.api_guide['arg_params']
        for param_key in param_list:
            try:
                arg_str += self.api_guide[param_key]
            except KeyError:
                print('API Parameter ' + param_key + ' not found in the api guide.')
                print('Here is the list of available options:')
                print(self.api_guide)
        query_str = self.api_guide['base_url'] + symbol_str + '&' + arg_str
        return query_str

    def load_key_metrics(self):
        wiki_tickers = pd.read_csv('WIKI_tickers.csv')
        self.symbols = [symbol[symbol.find('/')+1:] for symbol in wiki_tickers['quandl code']]
        self.key_metrics = ['Symbol',
                            'Name',
                     'Market Capitalization',
                     'Earnings/Share',
                     'Dividend Yield',
                     'EBITDA',
                     'EPS Estimate Current Year',
                     'EPS Estimate Next Quarter',
                     'EPS Estimate Next Year',
                     '200-day Moving Average',
                     '50-day Moving Average',
                     '52-week High',
                     '52-week Low',
                     '52-week Range',
                     'Price/Book',
                     'Book Value',
                     'Volume',
                     'Average Daily Volume',
                     'Short Ratio',
                     'P/E Ratio',
                     'PEG Ratio',
                     'Ask',
                     'Open',
                     'Previous Close',
                     'Change in Percent',
                     'Day Range',
                     'More Info',
                     'Notes']


    #Other Sources
    # https://greenido.wordpress.com/2009/12/22/yahoo-finance-hidden-api/
    #      http://finance.yahoo.com/d/quotes.csv?s=GE+PTR+MSFT&f=snd1l1yr


class TestYahooFinanceAPI(unittest.TestCase):

    def setUp(self):
        self.yfAPI = YahooFinanceAPI()

#    def test_yahoofinanceapi(self):
#         ticker = 'AAPL'
#         from_date = datetime.date(2014, 9, 1)
#         to_date = datetime.date.today()
#         stockdata = self.yfAPI.pull(ticker, from_date,to_date)
#         print(stockdata.columns[:10])
#         print(stockdata.head())
    def test_load_init_file(self):
        self.yahoo_api = YahooFinanceAPI()
        params = []
        for key in self.yahoo_api.api_guide.keys():
            if key not in ['base_url', 'arg_stock_symbol', 'arg_params', 'example']:
                params.append(key)
        #print(params)
        #self.yahoo_api.pull_data(['GOOG','MMM'], params)
        print(self.yahoo_api.symbols)
        self.yahoo_api.pull_key_metrics(self.yahoo_api.symbols)

class QuandlAPI(object):

    def __init__(self, version='v3', API_KEY='', settings_dir='./'):
        #self.settings_dir = os.path.abspath(object.__module__)
        self.settings_dir = settings_dir
        self.QUANDL_API_URL = 'https://www.quandl.com/api/' + version + '/'
        self.API_KEY = API_KEY
        self.load_local_data()


    def load_local_data(self):
        self.datasets = pd.DataFrame(load_json(self.settings_dir + 'quandl_datasets.json'))
        self.wiki_tickers = pd.read_csv(self.settings_dir + 'WIKI_tickers.csv')

        #self.allstockdata = pd.read_csv('../rawdata/all_stock_data.csv')

    def resp_to_json(self,resp):
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception("API call failed with status code: {}".formart(resp.status_code))

    def num_stocks_moving(self): # NOT COMPLETE
        # Queries the unicorn database to get stock price changes
        codes = ['NYSE_ADV', 'NYSE_DEC', 'NYSE_UNC', 'NYSE_ADV_VOL',
                 'NYSE_DEC_VOL', 'NYSE_UNCH_VOL', 'NYSE_52W_HI', 'NYSE_52W_LO']
        num_stocks = Quandl.get("URC/" + codes[0])

    def quandl_get(self, **kwargs):
        if self.API_KEY != '':
            data = Quandl.get(authtoken=self.API_KEY, **kwargs)
        else:
            data = Quandl.get(**kwargs)
        return data

    def update_indices(self, outfile='../rawdata/snp_raw.csv'):
        data = Quandl.get("SPDJ/SPX")
        data.to_csv(outfile)
        return data

    def pull_stock_list(self, stock_list, trim_start, trim_end):
        #database_codes = self.wiki_tickers['quandl code']
        if 'WIKI' not in stock_list.iloc[0]:
            database_codes = list(map(lambda x: 'WIKI/'+ x, stock_list))
        else:
            database_codes = stock_list

        ticker_list = [x + '.4' for x in list(database_codes)]  # Pull only the fourth column
        data = self.pull_data(ticker_list, trim_start=trim_start, trim_end=trim_end)
        return data

    def pull_commodities(self, trim_start, trim_end):
        database = 'LME'
        commodities_datasets = {'Aluminum': 'PR_AL', }
        Quandl.get("LME/PR_AL", authtoken="TwFoyqDFoRsxYvx-3ddp")

    def pull_data(self, dataset_codes, trim_start='2016-01-01',trim_end=''):
        if trim_end == '':
            trim_end = datetime.date.today().strftime('%Y-%m-%d')
        try:
            if self.API_KEY != '':
                    data = Quandl.get(dataset_codes, authtoken=self.API_KEY, trim_start=trim_start,
                                  trim_end=trim_end)
            else:
                data = Quandl.get(dataset_codes, trim_start=trim_start, trim_end=trim_end)
        except:
            data = pd.DataFrame()

        return data

    def prior_close_date(self, testdatetime):
        testdate = datetime.datetime.strptime(str(testdatetime.year) + '-' + str(testdatetime.month) + '-' + str(testdatetime.day), '%Y-%m-%d')
        prior_close = testdate + datetime.timedelta(hours=14)    # Need to convert this to UTC? Markets close at 2pm MST?

        if testdatetime > prior_close:  # If the current time is after the close then pull the data for today
            day_to_pull = testdate
        else:
            day_to_pull = testdate - datetime.timedelta(days=1)

        return day_to_pull

    def daily_update(self, outputdir='../rawdata/', outfile='daily_stock_data.csv', hist_stock_data='all_stock_data.csv'):
        today = self.prior_close_date(datetime.datetime.now())
        if today.weekday() < 5:  # Monday = 0, Sunday = 6
            today_str = today.strftime('%Y-%m-%d')   # Need to update so it runs for the previous close.
            data = self.update_all_stocks_data(outputdir + outfile, trim_start=today, trim_end=today)
            #data = pd.read_csv('../rawdata/daily_stock_data.csv', index_col=0)
            histdata = pd.read_csv(hist_stock_data, index_col=0)
            print(histdata.shape)
            alldata = pd.concat([histdata, data])
            print(histdata.shape, data.shape, alldata.shape)
            os.rename(hist_stock_data, hist_stock_data[:hist_stock_data.find('.')] + '_old.csv')
            alldata.to_csv(hist_stock_data)
            return data
        else:
            print("Notice: Daily Stock data not pulled because it is not a weekday.")

    def update_all_stocks_data(self, filename='', trim_start='2016-01-01', trim_end=''):
        # SOURCE WIKI
        #database_codes = self.wiki_tickers.sample(num_stocks_to_pull)['quandl code']
        data = pd.DataFrame()
        batch_size = 500
        n_batches = math.ceil(len(self.wiki_tickers.index)/batch_size)+1
        for i in range(0, int(n_batches)):
            start_row = i*batch_size
            end_row = min(start_row + batch_size, len(self.wiki_tickers.index))
            print("Pulling data for stocks:")
            print(start_row, end_row)
            print(list(self.wiki_tickers['quandl code'][start_row:end_row]))

            if i == 0:
                data = self.pull_stock_list(self.wiki_tickers['quandl code'][start_row:end_row], trim_start=trim_start, trim_end=trim_end)
            else:
                data = data.join(self.pull_stock_list(self.wiki_tickers['quandl code'][start_row:end_row], trim_start=trim_start, trim_end=trim_end))
            if filename != '':
                data.to_csv(os.path.dirname(filename) + '/temp' + str(i) + '.csv')

        if filename != '':
            data.to_csv(filename)
        self.allstockdata = data
        return data

    def update_WIKI_tickers(self):
        stocklist = pd.read_csv("https://s3.amazonaws.com/quandl-static-content/Ticker+CSV%27s/WIKI_tickers.csv")
        if stocklist.shape[0] > 10:
            stocklist.to_csv('WIKI_tickers.csv')
            self.WIKI_tickers = stocklist

    def update_datasets(self):
        databases_list = []
        for page in range(1, 5):
            print(page)
            resp_json = self.resp_to_json(requests.get(self.QUANDL_API_URL + 'databases' + '?=page=' + str(page)))
            databases_list.extend(resp_json['databases'])

        write_json('quandl_databases.json', databases_list)
        self.datasets = pd.DataFrame(databases_list)
        #self.datasets.to_csv('quandl_datasets.csv',index=False)   # doesn't work due to no ascii characters in description

    def update_stock_list(self, outputdir='../rawdata/', filename='stocklist.csv'):
        # Pulls Stock info from NASDAQ.com for NYSE, NASDAQ, and AMEX
        #http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download
        stock_exchange = 'nyse'
        outfilename = outputdir + filename

        base_url = 'http://www.nasdaq.com/screening/companies-by-name.aspx'
        params = '?letter=0&exchange=' + stock_exchange + '&render=download'
        qry_str = base_url + params

        resp = requests.get(qry_str)
        if resp.status_code == 200:
            with open(outfilename, 'w') as outfile:
                outfile.write(resp.content)
        else:
            raise Exception("NASDAQ.com API returned incomplete status code: " + str(resp.status_code))

    # Other Free Datasets to add
        # Commodities Data
            # https://www.quandl.com/collections/markets/commodities
            # https://www.quandl.com/blog/api-for-commodity-data
        # World Index Funds
        #    https://www.quandl.com/data/NASDAQOMX/documentation/metadata
        # American Association of Individual Investors
        #    https://www.quandl.com/data/AAII/documentation/metadata
        # InterContintenal Exchange
        #    https://www.quandl.com/data/ICE/documentation/documentation
        # Chicago Mercantile Exchange Futures
        #    https://www.quandl.com/data/CME/documentation/documentation
        # Unicorn Research Company    Quandl Code: URC
        #    https://www.quandl.com/data/URC/documentation/metadata
                # Lists # of stocks: advancing, declining, unchanged by price and volume
                # List # of stocks: with 52week high and low
                # Exchanges: NYSE, AMEX, and NASDAQ

class TestQuandlAPIDataRetreival(unittest.TestCase):

    def setUp(self):
        QUANDL_API_KEY = settings.QUANDL_API_KEY
        print(QUANDL_API_KEY)
        self.qAPI = QuandlAPI(API_KEY=QUANDL_API_KEY)

    #def test_prior_close_date(self):
    #    timelist = [datetime.datetime.today() + datetime.timedelta(hours=i) for i in range(24)]
    #    for test_time in timelist:
    #        print(test_time, self.qAPI.prior_close_date(test_time))

    # def test_update_stock_list(self):
    #     self.qAPI = QuandlAPI()
    #     outdir = '../rawdata/'
    #     outfile = 'stocklisttest.csv'
    #     self.qAPI.update_stock_list(outdir, outfile)
    #     self.assertTrue(os.path.exists(outdir + outfile))

    # def test_getdatasets(self):
    #     self.qAPI.update_datasets()
    #     print(self.qAPI.datasets.columns[:10])
    #     print(self.qAPI.datasets.head())
    #     print(self.qAPI.datasets.shape)
    #     self.qAPI.update_indices()

    def test_update_stockdata(self):
         self.qAPI.update_all_stocks_data('../rawdata/csv/all_stock_data_160328.csv', trim_start='2016-03-24', trim_end='2016-03-24')

    #def test_daily_stockupdate(self):
    #    self.qAPI.daily_update()


if __name__ == "__main__":
    #df = pd.read_csv('../rawdata/snp.csv')
    #data = DataRetrieval()
    unittest.main()
