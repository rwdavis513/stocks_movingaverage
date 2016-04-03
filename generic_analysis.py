import pandas as pd
#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

#stock_list = pd.read_csv('/home/bob/Documents/AutoVid/datademistifier/rawdata/snp.csv')
stock_list = pd.read_csv('/home/bob/Documents/AutoVid/datademistifier/rawdata/csv/stocklist.csv')
data = pd.read_csv('/home/bob/Documents/AutoVid/datademistifier/rawdata/csv/all_stock_data.csv', index_col=0)

new_columns = [col[col.find('.')+1:col.find(' ')] for col in data.columns]
data.columns = new_columns

def old():
    stock_list[['Ticker','Company','GICSSubIndustry']]
    industries = stock_list.groupby('GICSSubIndustry').groups.keys()

    industry_trends = {}

    for industry in industries:
        ticker_list = list(stock_list[stock_list['GICSSubIndustry'] == industry]['Ticker'])
        ticker_list = [col for col in ticker_list if col in d.columns]
        industry_trends[industry] = d[ticker_list].mean(axis=1)

    d_industry = pd.DataFrame(industry_trends)

    for industry in industries:
        print(industry)
        fig = plt.figure()
        ax = d_industry[industry].plot()
        ax.set_ylabel(industry)
        plt.savefig('images/' + industry + '.png')
        plt.close()

def create_grps(data, stock_list, grpname='GICSSector'):
    if grpname not in stock_list.columns:
        raise Exception('Missing ' + grpname + ' from stock_list.')

    stock_list[['Symbol', 'Name', grpname]]
    grps = stock_list.groupby(grpname).groups.keys()
    grp_trends = {}

    for group in grps:
        ticker_list = list(stock_list[stock_list[grpname] == group]['Symbol'])
        ticker_list = [col for col in ticker_list if col in data.columns]
        grp_trends[group] = data[ticker_list].mean(axis=1)

    grps_df = pd.DataFrame(grp_trends)

    return grps_df


def plot_grps(grps_df):

    for grp in grps_df.keys():
        print(grp)
        grp_title = grp.replace('/', '')
        fig = plt.figure()
        ax = grps_df[grp].plot()
        ax.set_ylabel(grp)
        plt.savefig('images/' + grp_title + '.png')
        plt.close()


def plot_corr(data, title='Correlation', folder='images/'):
    title = title.replace('/', '')
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(x, y, marker='o', ls='None')
    ax[0].set_xlabel(data.columns[0])
    ax[0].set_ylabel(data.columns[1])
    ax[0].set_title(data.columns[1] + ' vs ' + data.columns[0])
    ax[1] = plt.subplot(122)
    data.plot(ax=ax[1])
    #ax[1].plot(data.ix[:, 0])
    #ax[1].plot(data.ix[:, 1], secondary_y=True)
    labels = ax[1].get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    plt.savefig(folder + title + '.png')
    plt.close()


def analyze_corr(data, stock_list, grpname):
    grps_df = create_grps(data, stock_list, grpname)
    grp_cov = grps_df.cov()
    grp_cov_stacked = grp_cov.stack()
    grp_cov_stacked.sort_values(ascending=False, inplace=True)
    mylist = [(name[0], name[1], grp_cov_stacked[name]) for name in grp_cov_stacked.index if not name[0] == name[1]]
    # mylistnew = []
    # for row in mylist:
    #     if row not in mylistnew:
    #         mylistnew.extend(row)
    grps_cov_stacked_new = pd.DataFrame(mylist, columns=['Grp1', 'Grp2', 'Value'])
    grps_cov_stacked_new['OID'] = grps_cov_stacked_new['Grp1'] + '_' + grps_cov_stacked_new['Grp2'] + '_' + str(grps_cov_stacked_new['Value'])
    return grps_df, grps_cov_stacked_new


def analyze_fundamentals():
    sf = pd.read_csv('/home/bob/Documents/AutoVid/datademistifier/rawdata/csv/yahoo_api_alldata.csv')

    graham = sf[(sf['P/E Ratio'] < 15) & (sf['Price/Book'] < 1.5)][['Symbol', 'Name', 'P/E Ratio', 'Price/Book']]
    return rf


def relative_strength(timeseries):
    timeseries = timeseries.dropna()
    timeseries_delta = timeseries.diff(-1)
    gains = timeseries_delta[timeseries_delta>=0].fillna(0)
    losses = timeseries_delta[timeseries_delta<0].fillna(0)

    ewma_gains = pd.ewma(gains, com=13)
    ewma_gains = ewma_gains.iloc[13:]     # Drop the first 13 points to get a valid stat
    ewma_losses = pd.ewma(losses, com=13)*-1
    ewma_losses = ewma_losses.iloc[13:]     # Drop the first 13 points to get a valid stat
    rs = ewma_gains/ewma_losses
    rsi = 100*(1-1/(1+rs))
    return rsi


def plot_rsi(timeseries, title='RSI Overlay.png'):
    fig, ax = plt.subplots(2, 1, sharex=True)
    timeseries.plot(ax=ax[0])
    ax[0].set_ylabel(timeseries.columns[0])   # should only have one column
    ax[0].set_title(timeseries.columns[0] + ' time trend with RSI')
    rsi = relative_strength(timeseries)
    rsi.plot(ax=ax[1])
    ax[1].set_ylabel('Relative Strength Index (RSI)')
    ax[1].set_xlabel('Date')
    labels = ax[1].get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=10)
    plt.savefig(title)
    plt.close()


def calc_rsi_dataframe(data):
    #plot_rsi(data[[data.columns[3]]], 'images/' + data.columns[3] + '.png')
    for colname in data.columns:
        data[colname + ' rsi'] = relative_strength(data[[colname]])
    data_rsi = data[[col for col in data.columns if 'rsi' in col]]    # New to change so it doesn't add to the original dataframe
    return data_rsi


def filter_rsi(data_rsi):
    recent_day = data_rsi.iloc[-1,]
    top_values = recent_day.sort_values(ascending=False).head(n=10)
    return top_values


def plot_top_rsi(top_values, data):
    for colname in top_values.index:
        colname = colname.replace(' rsi', '')
        plot_rsi(data[[colname]], 'images/' + colname + ' rsi.png')


def filter_outliers(mn):
    stats = mn.describe([0.01,0.99])
    mn_filter = mn[(mn<stats['1%']) & (mn>t['99%'])]
    return mn_filter

#industry_df, corr_list = analyze_corr(data, stock_list, 'industry')
#print(corr_list.head())
#for i in range(0, 10, 2):
#    plot_corr(industry_df[[corr_list.ix[i][0], corr_list.ix[i][1]]], title=corr_list.ix[i][0] + ' vs ' + corr_list.ix[i][1])

#plot_grps(industry_df)
sector_df, sector_corr_list = analyze_corr(data, stock_list, 'Sector')
#print(sector_df.shape)
#print(sector_corr_list.shape)
#plot_grps(sector_df)
for i in range(0, 50, 2):
    plot_corr(sector_df[[sector_corr_list.ix[i][0], sector_corr_list.ix[i][1]]], title=sector_corr_list.ix[i][0] + ' vs ' + sector_corr_list.ix[i][1])
#for i in range(sector_corr_list.shape[0]-1, 50, -2):
#    plot_corr(sector_df[[sector_corr_list.ix[i][0], sector_corr_list.ix[i][1]]], title=sector_corr_list.ix[i][0] + ' vs ' + sector_corr_list.ix[i][1])
