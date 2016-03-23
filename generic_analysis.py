import pandas as pd
#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

stock_list = pd.read_csv('/home/bob/Documents/AutoVid/datademistifier/rawdata/snp.csv')
data = pd.read_csv('/home/bob/Documents/AutoVid/datademistifier/rawdata/all_stock_data.csv', index_col=0)

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

    stock_list[['Ticker', 'Company', grpname]]
    grps = stock_list.groupby(grpname).groups.keys()
    grp_trends = {}

    for group in grps:
        ticker_list = list(stock_list[stock_list[grpname] == group]['Ticker'])
        ticker_list = [col for col in ticker_list if col in data.columns]
        grp_trends[group] = data[ticker_list].mean(axis=1)

    grps_df = pd.DataFrame(grp_trends)

    return grps_df


def plot_grps(grps_df):

    for grp in grps_df.keys():
        print(grp)
        fig = plt.figure()
        ax = grps_df[grp].plot()
        ax.set_ylabel(grp)
        plt.savefig('images/' + grp + '.png')
        plt.close()


def plot_corr(data, title='Correation', folder='images/'):
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

industry_df, corr_list = analyze_corr(data, stock_list, 'GICSSubIndustry')
print(corr_list.head())
for i in range(0, 10, 2):
    plot_corr(industry_df[[corr_list.ix[i][0], corr_list.ix[i][1]]], title=corr_list.ix[i][0] + ' vs ' + corr_list.ix[i][1])

#plot_grps(industry_df)
#sector_df = create_grps(data, stock_list, 'GICSSector')
#plot_grps(sector_df)

