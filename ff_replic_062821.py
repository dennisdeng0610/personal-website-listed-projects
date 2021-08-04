# coding=utf-8
"""
@author : dennisdeng
@file   : ff_replic_062821.py
@ide    : PyCharm
@time   : 06-28-2021 14:30:45
"""

# import packages
import os
import pandas as pd
import wrds
import pandas_datareader
from pandas.tseries.offsets import MonthEnd
import datetime
import numpy as np
from scipy.stats import skew

# pandas settings
pd.options.mode.chained_assignment = None
pd.set_option('precision', 3)
pd.set_option('display.max_columns', None)


def data_acquirer(local=None):
    """download crsp and compustat data from wrds, fama-french sample from ff-website

    :param local: equals True to load files locally
    :return: crsp, compustat, and ff dataset (6 datasets in total)
    """
    # returns the current working directory
    os.getcwd()

    # create .pgpass file with the following statement:
    # conn = wrds.Connection(wrds_username='username')
    # conn.create_pgpass_file()
    # conn.close()
    # the file is located at C:/Users/[username]/AppData/Roaming/postgresql
    # or create pgpass.conf with text:
    # "wrds-pgdata.wharton.upenn.edu:9737:wrds:[username]:[password]"

    # acquire data
    # create a "date" variable for each set in pd.datetime for linking purpose
    # push the "date" to the last day of each month
    if not local:
        # build the connection to wrds
        db = wrds.Connection(wrds_username='dennisdeng')

        # acquire compustat data from wrds
        _comp = db.raw_sql("""
                            select a.gvkey, a.datadate, a.at, a.pstkl, a.txditc, a.fyear, a.ceq, a.lt, 
                            a.mib, a.itcb, a.txdb, a.pstkrv, a.seq, a.pstk, b.sic, b.year1, b.naics
                            from comp.funda as a
                            left join comp.names as b
                            on a.gvkey = b.gvkey
                            where indfmt='INDL'
                            and datafmt='STD'
                            and popsrc='D'
                            and consol='C'
                            """)
        _comp['date'] = pd.to_datetime(_comp['datadate'], format='%Y-%m-%d', errors='ignore') + MonthEnd(0)

        # acquire pension data from wrds
        _pens = db.raw_sql("""
                                select gvkey, datadate, prba
                                from comp.aco_pnfnda
                                where indfmt='INDL'
                                and datafmt='STD'
                                and popsrc='D'
                                and consol='C'
                                """)
        _pens['date'] = pd.to_datetime(_pens['datadate'], format='%Y-%m-%d', errors='ignore') + MonthEnd(0)

        # acquire linktable from wrds
        _link = db.raw_sql("""
                          select gvkey, lpermno as permno, lpermco as permco, linktype, linkprim, liid,
                          linkdt, linkenddt
                          from crspq.ccmxpf_linktable
                          where substr(linktype,1,1)='L'
                          and (linkprim ='C' or linkprim='P')
                          """)
        _link['date_start'] = pd.to_datetime(_link['linkdt'], format='%Y-%m-%d', errors='ignore')
        _link['date_end'] = pd.to_datetime(_link['linkenddt'], format='%Y-%m-%d', errors='ignore')

        # acquire crsp data from wrds
        # restrict to US exchanges(1,2,3) and common shares(10,11)
        _crsp = db.raw_sql("""
                              select a.permno, a.permco, a.date, b.exchcd, b.siccd, b.naics,
                              a.ret, a.retx, a.shrout, a.prc
                              from crspq.msf as a
                              left join crspq.msenames as b
                              on a.permno=b.permno
                              and b.namedt<=a.date
                              and a.date<=b.nameendt
                              where b.shrcd in (10,11)
                              and b.exchcd in (1,2,3)
                              """)
        _crsp['date'] = pd.to_datetime(_crsp['date'], format='%Y-%m-%d', errors='ignore') + MonthEnd(0)

        # acquire crsp delisting data from wrds
        _dlret = db.raw_sql("""
                                select a.permno, a.permco, a.dlret, a.dlretx, a.dlstdt, 
                                b.exchcd as dlexchcd, b.siccd as dlsiccd, b.naics as dlnaics
                                from crspq.msedelist as a
                                left join crspq.msenames as b
                                on a.permno=b.permno
                                and b.namedt<=a.dlstdt
                                and a.dlstdt<=b.nameendt
                                where b.shrcd in (10,11)
                                and b.exchcd in (1,2,3)
                                """)
        _dlret['date'] = pd.to_datetime(_dlret['dlstdt'], format='%Y-%m-%d', errors='ignore') + MonthEnd(0)
        db.close()

        # acquire ff-1993-3factors data
        pd.set_option('precision', 2)
        data2 = pandas_datareader.famafrench.FamaFrenchReader('F-F_Research_Data_Factors', start='1900',
                                                              end=str(datetime.datetime.now().year + 1))
        _ff = data2.read()[0] / 100  # Monthly data
        _ff['Mkt'] = _ff['Mkt-RF'] + _ff['RF']

        # acquire 10 book-to-market portfolios
        data2 = pandas_datareader.famafrench.FamaFrenchReader('Portfolios_Formed_on_BE-ME', start='1900',
                                                              end=str(datetime.datetime.now().year + 1))
        data2 = data2.read()[0][
                    ['Lo 10', 'Dec 2', 'Dec 3', 'Dec 4', 'Dec 5', 'Dec 6', 'Dec 7', 'Dec 8', 'Dec 9', 'Hi 10']] / 100
        data2.columns = 'BM01', 'BM02', 'BM03', 'BM04', 'BM05', 'BM06', 'BM07', 'BM08', 'BM09', 'BM10'
        _ff = pd.merge(_ff, data2, how='left', on=['Date'])

        # acquire 10 size portfolios
        data2 = pandas_datareader.famafrench.FamaFrenchReader('Portfolios_Formed_on_ME', start='1900',
                                                              end=str(datetime.datetime.now().year + 1))
        data2 = data2.read()[0][
                    ['Lo 10', 'Dec 2', 'Dec 3', 'Dec 4', 'Dec 5', 'Dec 6', 'Dec 7', 'Dec 8', 'Dec 9', 'Hi 10']] / 100
        data2.columns = 'ME01', 'ME02', 'ME03', 'ME04', 'ME05', 'ME06', 'ME07', 'ME08', 'ME09', 'ME10'
        _ff = pd.merge(_ff, data2, how='left', on=['Date'])

        # acquire 5*5 book-to-market * size portfolios
        data2 = pandas_datareader.famafrench.FamaFrenchReader('25_Portfolios_5x5', start='1900',
                                                              end=str(datetime.datetime.now().year + 1))
        data2 = data2.read()[0].rename(
            columns={"SMALL LoBM": "ME1 BM1", "SMALL HiBM": "ME1 BM5", "BIG LoBM": "ME5 BM1",
                     "BIG HiBM": "ME5 BM5"}) / 100
        _ff = pd.merge(_ff, data2, how='left', on=['Date'])
        _ff = _ff.reset_index().rename(columns={"Date": "date"})
        _ff['date'] = pd.DataFrame(_ff[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)
        _ff['year'] = _ff['date'].dt.year
        _ff['month'] = _ff['date'].dt.month
        # restrict ff data from 1973 to 2020
        _ff = _ff.loc[(_ff['year'] <= 2020) & (_ff['year'] >= 1973)]
        _ff.reset_index(drop=True, inplace=True)

        _comp.to_pickle('comp.pkl')
        _pens.to_pickle('pens.pkl')
        _link.to_pickle('link.pkl')
        _crsp.to_pickle('crsp.pkl')
        _dlret.to_pickle('dlret.pkl')
        _ff.to_pickle('ff.pkl')
    else:
        _comp = pd.read_pickle('comp.pkl')
        _pens = pd.read_pickle('pens.pkl')
        _link = pd.read_pickle('link.pkl')
        _crsp = pd.read_pickle('crsp.pkl')
        _dlret = pd.read_pickle('dlret.pkl')
        _ff = pd.read_pickle('ff.pkl')
    return _comp, _pens, _link, _crsp, _dlret, _ff


comp, pens, link, crsp, dlret, ff = data_acquirer(local=True)


def size_portfolio_replic(df1, df2, df3, step3=None):
    """construct size portfolio and compare with ff from Jan 1973 to Dec 2020

    :param df1: crsp dataset from wrds
    :param df2: delisted dataset from wrds
    :param df3: fama french dataset for comparison purpose
    :param step3: if true return crsp with decile for HML & SMB calculation
    :return: replication statistics and correlations with fama-french-size-portfolio
    """
    # merge crsp & delisted stock returns
    _crsp = pd.merge(df1, df2, how='outer', on=['date', 'permno'])

    # use the two returns together
    _crsp = _crsp[~(_crsp['ret'].isna() & _crsp['dlret'].isna())]
    _crsp.loc[_crsp['ret'].isna(), 'ret'] = 0
    _crsp.loc[_crsp['dlret'].isna(), 'dlret'] = 0
    _crsp['ret'] = (_crsp['ret'] + 1) * (_crsp['dlret'] + 1) - 1

    _crsp['prc'] = abs(_crsp['prc'])  # use positive price
    _crsp['me'] = _crsp['prc'] * _crsp['shrout'] / 1000  # shares in thousands, market value in millions
    _crsp['year'] = _crsp['date'].dt.year
    _crsp['month'] = _crsp['date'].dt.month

    # sort the df first by stock code then by date
    _crsp = _crsp.sort_values(by=['permno', 'date'], ascending=True)

    # get the market value for the previous month, for the value-weighted rets calculations later
    _crsp['lag_me'] = _crsp['me'].shift(1)
    _crsp.loc[_crsp['permno'].shift(1) != _crsp['permno'], 'lag_me'] = np.nan

    # get the fiscal year: if month<=6 then fyear = year - 1
    _crsp['fyear'] = _crsp['year']
    _crsp.loc[_crsp['month'] <= 6, 'fyear'] = _crsp['fyear'] - 1

    # at the end of each June, use me as indicator to construct new portfolio
    _construct = _crsp[_crsp['month'] == 6].copy()
    _construct['fyear'] = _construct['fyear'] + 1  # me at June is used for the following fyear
    _construct = _construct[['fyear', 'me', 'permno']]

    # merge the me indicator with the original dataset
    # now there are 2 me related values:
    # lag_me for vw ret calculation, and me_ind for decile classification
    _crsp = pd.merge(_crsp, _construct, how='left', on=['fyear', 'permno'])
    _crsp = _crsp[_crsp['me_y'].notna()]
    _crsp = _crsp.drop(columns='me_x')
    _crsp.rename(columns={'me_y': 'me_ind'}, inplace=True)

    # obtain the breakpoints, use nyse stocks and me at end of June (start of July)
    _nyse = _crsp.loc[(_crsp['exchcd'] == 1) & (_crsp['month'] == 7)]

    # use quantile function to get breakpoints for each time period
    _indicator = _nyse.groupby(['fyear'])['me_ind'].quantile(0.1).to_frame()
    _indicator.reset_index(drop=False, inplace=True)
    _indicator.rename(columns={'me_ind': 'd'}, inplace=True)
    for i in range(2, 10):
        _dec_insert = _nyse.groupby(['fyear'])['me_ind'].quantile(0.1 * i)
        _dec_insert.reset_index(drop=True, inplace=True)
        _indicator.insert(_indicator.shape[1], 'd' * i, _dec_insert)

    # merge the breakpoints to the original dataset
    _crsp = pd.merge(_crsp, _indicator, how='left', on=['fyear'])

    # obtain the decile for each observation
    _crsp.loc[(_crsp['me_ind'] <= _crsp['d']), 'decile'] = 1  # dec1
    _crsp.loc[(_crsp['me_ind'] > _crsp['d' * 9]), 'decile'] = 10  # dec10
    for i in range(1, 9):
        _crsp.loc[(_crsp['me_ind'] > _crsp['d' * i])
                  & (_crsp['me_ind'] <= _crsp['d' * (i + 1)]), 'decile'] = i + 1  # dec2-9

    # if step3 is true, return crsp for HML & SMB calculation
    if step3:
        return _crsp

    # obtain the value-weighted rets for each month
    _crsp['ret*lag_me'] = _crsp['ret'] * _crsp['lag_me']
    _crsp_vw = (_crsp.groupby(['year', 'month', 'decile'])['ret*lag_me'].sum() /
                _crsp.groupby(['year', 'month', 'decile'])['lag_me'].sum()).to_frame()
    _crsp_vw.reset_index(drop=False, inplace=True)
    _crsp_vw.rename(columns={'decile': 'port', 0: 'Size_Ret'}, inplace=True)

    # restrict time from Jan1973 to Dec2020
    _crsp_vw = _crsp_vw.loc[(_crsp_vw['year'] <= 2020) & (_crsp_vw['year'] >= 1973)]
    _crsp_vw.reset_index(drop=True, inplace=True)

    _ff = df3.copy()
    _rf = _ff[['year', 'month', 'RF']]  # get risk-free from ff
    _ff['wml_size'] = _ff['ME01'] - _ff['ME10']  # get long-short portfolio by dec1 minus dec10 in ff
    _crsp_vw = pd.merge(_crsp_vw, _rf, on=['year', 'month'], how='inner')
    _crsp_vw['exret'] = _crsp_vw['Size_Ret'] - _crsp_vw['RF']  # get excess returns

    # get long-short portfolio by dec1 minus dec10 in replication
    _ls = pd.merge(_crsp_vw[_crsp_vw['port'] == 1], _crsp_vw[_crsp_vw['port'] == 10], on=['year', 'month'], how='inner')
    _ls['wml'] = _ls['Size_Ret_x'] - _ls['Size_Ret_y']  # dec1 - dec10
    _ls = _ls[['year', 'month', 'wml']]

    # annualized, in percentage
    _ls_mean = np.mean(_ls['wml']) * 12 * 100
    _ls_std = np.std(_ls['wml']) * np.sqrt(12) * 100

    # get output values,
    # rows are exrets, standard deviations, Sharpe Ratios, skewnesses, and correlations with ff
    # columns are dec1 to dec10, and long-short
    _output = pd.DataFrame(index=np.arange(5), columns=np.arange(11))
    _output.iloc[[0], :-1] = _crsp_vw.groupby('port')['exret'].mean() * 12 * 100
    _output.iloc[[0], [10]] = _ls_mean
    _output.iloc[[1], :-1] = _crsp_vw.groupby('port')['exret'].std() * np.sqrt(12) * 100
    _output.iloc[[1], [10]] = _ls_std
    _output.iloc[[2], :-1] = np.array(_output.iloc[[0], :-1]) / np.array(_output.iloc[[1], :-1])
    _output.iloc[[2], [10]] = _ls_mean / _ls_std
    _output.iloc[[3], :-1] = _crsp_vw.groupby('port')['exret'].skew()
    _output.iloc[[3], [10]] = skew(_ls['wml'])

    # get the correlations for each decile between replication and ff
    for i in range(11):
        if i <= 8:
            _replic = _crsp_vw[_crsp_vw['port'] == (i + 1)]
            _replic.reset_index(drop=True, inplace=True)
            _ff_group = _ff[['year', 'month', ('ME0' + str(i + 1)), 'RF']]
            _ff_group['exret_ff'] = _ff_group['ME0' + str(i + 1)] - _ff_group['RF']
        elif i == 9:
            _replic = _crsp_vw[_crsp_vw['port'] == (i + 1)]
            _replic.reset_index(drop=True, inplace=True)
            _ff_group = _ff[['year', 'month', 'ME10', 'RF']]
            _ff_group['exret_ff'] = _ff_group['ME10'] - _ff_group['RF']
        else:
            _replic = _ls
            _replic['exret'] = _replic['wml']
            _ff_group = _ff[['year', 'month', 'wml_size', 'RF']]
            _ff_group['exret_ff'] = _ff_group['wml_size']
        _compare = pd.merge(_replic, _ff_group, on=['year', 'month'], how='left')
        _output.iloc[[4], [i]] = _compare.corr().loc['exret', 'exret_ff']

    # rename the output stats
    _output.rename(
        columns={0: 'D1', 1: 'D2', 2: 'D3', 3: 'D4', 4: 'D5',
                 5: 'D6', 6: 'D7', 7: 'D8', 8: 'D9', 9: 'D10', 10: 'LS'},
        index={0: 'exret', 1: 'sd', 2: 'SR', 3: 'skew', 4: 'corr'},
        inplace=True)
    return _output


size_portfolio_stats = size_portfolio_replic(crsp, dlret, ff)


def btm_portfolio_replic(df1, df2, df3, df4, df5, df6, step3=None):
    """construct book-to-market portfolio and compare with ff from Jan 1973 to Dec 2020

    :param df1: compustat dataset from wrds
    :param df2: pension dataset from wrds
    :param df3: linktable from wrds dataset
    :param df4: crsp dataset from wrds
    :param df5: delisted dataset from wrds
    :param df6: fama french dataset for comparison purpose
    :param step3: if true return crsp with decile for HML & SMB calculation
    :return: replication statistics and correlations with fama-french-btm-portfolio
    """

    # obtain book values following accounting rules
    _comp = pd.merge(df1, df2, how='outer', on=['date', 'gvkey'])
    _comp['she'] = _comp['seq']
    _comp.loc[_comp['she'].isna(), 'she'] = _comp.loc[_comp['she'].isna(), 'ceq'] + _comp.loc[
        _comp['she'].isna(), 'pstk']
    _comp.loc[_comp['she'].isna(), 'she'] = _comp.loc[_comp['she'].isna(), 'at'] - _comp.loc[
        _comp['she'].isna(), 'lt'] - _comp.loc[_comp['she'].isna(), 'mib']
    _comp.loc[_comp['she'].isna(), 'she'] = _comp.loc[_comp['she'].isna(), 'at'] - _comp.loc[
        _comp['she'].isna(), 'lt']
    _comp['ps'] = _comp['pstkrv']
    _comp.loc[_comp['ps'].isna(), 'ps'] = _comp.loc[_comp['ps'].isna(), 'pstkl']
    _comp.loc[_comp['ps'].isna(), 'ps'] = _comp.loc[_comp['ps'].isna(), 'pstk']
    _comp['dt'] = _comp['txditc']
    _comp.loc[_comp['dt'].isna(), 'dt'] = _comp.loc[_comp['dt'].isna(), 'itcb'] + _comp.loc[
        _comp['dt'].isna(), 'txdb']
    _comp.loc[_comp['dt'].isna(), 'dt'] = _comp.loc[_comp['dt'].isna(), 'itcb']
    _comp.loc[_comp['dt'].isna(), 'dt'] = _comp.loc[_comp['dt'].isna(), 'txdb']
    _comp.loc[_comp['ps'].isna(), 'ps'] = 0
    _comp.loc[_comp['dt'].isna(), 'dt'] = 0
    _comp.loc[_comp['prba'].isna(), 'prba'] = 0
    _comp['be'] = _comp['she'] - _comp['ps'] + _comp['dt'] - _comp['prba']

    _link = df3
    # set the na end date in linktable to be current date
    _link.loc[_link['date_end'].isna(), 'date_end'] = pd.to_datetime('2021-07-03')
    # merge compustat with linktable, and make sure the linkage time is valid
    _comp_link = pd.merge(_comp, _link, how='outer', on=['gvkey'])
    _comp_link = _comp_link.loc[(_comp_link['date'] <= _comp_link['date_end'])
                                & (_comp_link['date'] >= _comp_link['date_start'])]
    _comp_link = _comp_link[['permco', 'be', 'fyear']]

    # obtain me values
    # merge crsp & delisted stock returns
    _crsp = pd.merge(df4, df5, how='outer', on=['date', 'permno', 'permco'])
    _crsp = _crsp[~(_crsp['ret'].isna() & _crsp['dlret'].isna())]
    _crsp.loc[_crsp['ret'].isna(), 'ret'] = 0
    _crsp.loc[_crsp['dlret'].isna(), 'dlret'] = 0
    _crsp['ret'] = (_crsp['ret'] + 1) * (_crsp['dlret'] + 1) - 1
    _crsp['prc'] = abs(_crsp['prc'])  # use positive price
    _crsp['me'] = _crsp['prc'] * _crsp['shrout'] / 1000  # shares in thousands, market value in millions
    _crsp['year'] = _crsp['date'].dt.year
    _crsp['month'] = _crsp['date'].dt.month

    _me = _crsp.groupby(['year', 'month', 'permco'])['me'].sum().reset_index()  # sum different subsidiaries' me values
    _me = _me.loc[_me['month'] == 12]  # use December me values at t-1 year to match with t-1 year be values
    _me.rename(columns={'year': 'fyear'}, inplace=True)
    _me = _me.drop(columns='month')

    # obtain b/m ratios
    _bm = pd.merge(_comp_link, _me, how='inner', on=['fyear', 'permco'])
    _bm['btm'] = _bm['be'] / _bm['me']
    _bm['fyear'] = _bm['fyear'] + 1
    _crsp = _crsp[['permco', 'exchcd', 'ret', 'me', 'date', 'year', 'month', 'permno']]
    _crsp.rename(columns={'me': 'mktcap'}, inplace=True)  # mktcap is used to calculate value-weighted rets
    # get the fiscal year: if month<=6 then fyear = year - 1
    _crsp['fyear'] = _crsp['year']
    _crsp.loc[_crsp['month'] <= 6, 'fyear'] = _crsp['fyear'] - 1
    # merge b/m ratio with the crsp data, notice that within each fiscal year, bm ratio does not change
    _crsp = pd.merge(_crsp, _bm, how='left', on=['fyear', 'permco'])

    # obtain the breakpoints, use nyse stocks and me at end of June (start of July)
    _nyse = _crsp.loc[(_crsp['exchcd'] == 1) & (_crsp['month'] == 7)]

    # use quantile function to get breakpoints for each time period
    _indicator = _nyse.groupby(['fyear'])['btm'].quantile(0.1).to_frame()
    _indicator.reset_index(drop=False, inplace=True)
    _indicator.rename(columns={'btm': 'd'}, inplace=True)
    for i in range(2, 10):
        _dec_insert = _nyse.groupby(['fyear'])['btm'].quantile(0.1 * i)
        _dec_insert.reset_index(drop=True, inplace=True)
        _indicator.insert(_indicator.shape[1], 'd' * i, _dec_insert)
    # merge the breakpoints to the original dataset
    _crsp = pd.merge(_crsp, _indicator, how='left', on=['fyear'])
    # obtain the decile for each observation
    _crsp.loc[(_crsp['btm'] <= _crsp['d']), 'decile'] = 1  # dec1
    _crsp.loc[(_crsp['btm'] > _crsp['d' * 9]), 'decile'] = 10  # dec10
    for i in range(1, 9):
        _crsp.loc[(_crsp['btm'] > _crsp['d' * i])
                  & (_crsp['btm'] <= _crsp['d' * (i + 1)]), 'decile'] = i + 1  # dec2-9

    # get the lag_me values for value-weighted rets calculation
    _crsp = _crsp.sort_values(by=['permno', 'year', 'month'], ascending=True)
    _crsp['lag_me'] = _crsp['mktcap'].shift(1)
    _crsp.loc[_crsp['permno'].shift(1) != _crsp['permno'], 'lag_me'] = np.nan
    _crsp['ret*lag_me'] = _crsp['ret'] * _crsp['lag_me']
    _crsp = _crsp[_crsp['ret*lag_me'].notna()]
    _crsp.reset_index(drop=True, inplace=True)

    # if step3 is true, return crsp for HML & SMB calculation
    if step3:
        return _crsp

    # obtain the value-weighted rets for each month
    _crsp_vw = (_crsp.groupby(['year', 'month', 'decile'])['ret*lag_me'].sum() /
                _crsp.groupby(['year', 'month', 'decile'])['lag_me'].sum()).to_frame()
    _crsp_vw.reset_index(drop=False, inplace=True)
    _crsp_vw.rename(columns={'decile': 'port', 0: 'B/M_Ret'}, inplace=True)
    # restrict time from Jan1973 to Dec2020
    _crsp_vw = _crsp_vw.loc[(_crsp_vw['year'] <= 2020) & (_crsp_vw['year'] >= 1973)]
    _crsp_vw.reset_index(drop=True, inplace=True)

    _ff = df6.copy()
    _rf = _ff[['year', 'month', 'RF']]  # get risk-free from ff
    _ff['wml_size'] = _ff['BM10'] - _ff['BM01']  # get long-short portfolio by dec10 minus dec1 in ff
    _crsp_vw = pd.merge(_crsp_vw, _rf, on=['year', 'month'], how='inner')
    _crsp_vw['exret'] = _crsp_vw['B/M_Ret'] - _crsp_vw['RF']  # get excess returns

    # get long-short portfolio by dec10 minus dec1 in replication
    _ls = pd.merge(_crsp_vw[_crsp_vw['port'] == 1], _crsp_vw[_crsp_vw['port'] == 10], on=['year', 'month'], how='inner')
    _ls['wml'] = _ls['B/M_Ret_y'] - _ls['B/M_Ret_x']  # dec10 - dec1
    _ls = _ls[['year', 'month', 'wml']]

    # annualized, in percentage
    _ls_mean = np.mean(_ls['wml']) * 12 * 100
    _ls_std = np.std(_ls['wml']) * np.sqrt(12) * 100

    # get output values,
    # rows are exrets, standard deviations, Sharpe Ratios, skewnesses, and correlations with ff
    # columns are dec1 to dec10, and long-short
    _output = pd.DataFrame(index=np.arange(5), columns=np.arange(11))
    _output.iloc[[0], :-1] = _crsp_vw.groupby('port')['exret'].mean() * 12 * 100
    _output.iloc[[0], [10]] = _ls_mean
    _output.iloc[[1], :-1] = _crsp_vw.groupby('port')['exret'].std() * np.sqrt(12) * 100
    _output.iloc[[1], [10]] = _ls_std
    _output.iloc[[2], :-1] = np.array(_output.iloc[[0], :-1]) / np.array(_output.iloc[[1], :-1])
    _output.iloc[[2], [10]] = _ls_mean / _ls_std
    _output.iloc[[3], :-1] = _crsp_vw.groupby('port')['exret'].skew()
    _output.iloc[[3], [10]] = skew(_ls['wml'])

    # get the correlations for each decile between replication and ff
    for i in range(11):
        if i <= 8:
            _replic = _crsp_vw[_crsp_vw['port'] == (i + 1)]
            _replic.reset_index(drop=True, inplace=True)
            _ff_group = _ff[['year', 'month', ('BM0' + str(i + 1)), 'RF']]
            _ff_group['exret_ff'] = _ff_group['BM0' + str(i + 1)] - _ff_group['RF']
        elif i == 9:
            _replic = _crsp_vw[_crsp_vw['port'] == (i + 1)]
            _replic.reset_index(drop=True, inplace=True)
            _ff_group = _ff[['year', 'month', 'BM10', 'RF']]
            _ff_group['exret_ff'] = _ff_group['BM10'] - _ff_group['RF']
        else:
            _replic = _ls
            _replic['exret'] = _replic['wml']
            _ff_group = _ff[['year', 'month', 'wml_size', 'RF']]
            _ff_group['exret_ff'] = _ff_group['wml_size']
        _compare = pd.merge(_replic, _ff_group, on=['year', 'month'], how='left')
        _output.iloc[[4], [i]] = _compare.corr().loc['exret', 'exret_ff']

    # rename the output stats
    _output.rename(
        columns={0: 'D1', 1: 'D2', 2: 'D3', 3: 'D4', 4: 'D5',
                 5: 'D6', 6: 'D7', 7: 'D8', 8: 'D9', 9: 'D10', 10: 'LS'},
        index={0: 'exret', 1: 'sd', 2: 'SR', 3: 'skew', 4: 'corr'},
        inplace=True)
    return _output


btm_portfolio_stats = btm_portfolio_replic(comp, pens, link, crsp, dlret, ff)


def hml_smb_portfolio_replic(df1, df2, df3, df4, df5, df6):
    """construct book-to-market portfolio and compare with ff from Jan 1973 to Dec 2020

    :param df1: compustat dataset from wrds
    :param df2: pension dataset from wrds
    :param df3: linktable from wrds dataset
    :param df4: crsp dataset from wrds
    :param df5: delisted dataset from wrds
    :param df6: fama french dataset for comparison purpose
    :return: replication statistics and correlations with fama-french-hml-smb-portfolio
    """

    # obtain size & btm deciles for each observation
    _data_size = size_portfolio_replic(df4, df5, df6, True)
    _data_btm = btm_portfolio_replic(df1, df2, df3, df4, df5, df6, True)
    _crsp = pd.merge(_data_size, _data_btm, on=['year', 'month', 'permno'], how='inner')

    # obtain 6 portfolios formed on size and book-to-market (2 x 3)
    _crsp['size_group'] = 2  # top 50% size==big==2
    _crsp.loc[_crsp['decile_x'] <= 5, 'size_group'] = 1  # bottom 50% size==small==1
    _crsp['btm_group'] = 2  # middle 40% btm==medium==2
    _crsp.loc[_crsp['decile_y'] <= 3, 'btm_group'] = 1  # bottom 30% btm==low==1
    _crsp.loc[_crsp['decile_y'] >= 8, 'btm_group'] = 3  # top 30% btm==high==3

    _crsp_vw = (_crsp.groupby(['year', 'month', 'size_group', 'btm_group'])['ret*lag_me'].sum() /
                _crsp.groupby(['year', 'month', 'size_group', 'btm_group'])['lag_me_y'].sum()).to_frame()
    _crsp_vw.reset_index(drop=False, inplace=True)
    _crsp_vw.rename(columns={0: 'Ret'}, inplace=True)

    # get hml by high group minus low group
    _hml = pd.pivot_table(_crsp_vw, values='Ret', index=['year', 'month'], columns=['btm_group'])
    _hml.rename(columns={1: 'low', 2: 'medium', 3: 'high'}, inplace=True)
    _hml['HML_Ret'] = _hml['high'] - _hml['low']
    _hml.reset_index(drop=False, inplace=True)

    # get smb by small group minus big group
    _smb = pd.pivot_table(_crsp_vw, values='Ret', index=['year', 'month'], columns=['size_group'])
    _smb.rename(columns={1: 'small', 2: 'big'}, inplace=True)
    _smb['SMB_Ret'] = _smb['small'] - _smb['big']
    _smb.reset_index(drop=False, inplace=True)

    # merge hml and smb rets, and restrict time range
    _hml_smb = pd.merge(_hml, _smb, on=['year', 'month'])
    _hml_smb = _hml_smb[['year', 'month', 'HML_Ret', 'SMB_Ret']].dropna()
    _hml_smb = _hml_smb.loc[(_hml_smb['year'] <= 2020) & (_hml_smb['year'] >= 1973)]

    # get output values
    _output = pd.DataFrame(index=np.arange(5), columns=np.arange(2))
    _output.iloc[[0], [0]] = np.mean(_hml_smb['HML_Ret']) * 12 * 100
    _output.iloc[[0], [1]] = np.mean(_hml_smb['SMB_Ret']) * 12 * 100
    _output.iloc[[1], [0]] = np.std(_hml_smb['HML_Ret']) * np.sqrt(12) * 100
    _output.iloc[[1], [1]] = np.std(_hml_smb['SMB_Ret']) * np.sqrt(12) * 100
    _output.iloc[[2], [0]] = np.mean(_hml_smb['HML_Ret']) * np.sqrt(12) / np.std(_hml_smb['HML_Ret'])
    _output.iloc[[2], [1]] = np.mean(_hml_smb['SMB_Ret']) * np.sqrt(12) / np.std(_hml_smb['SMB_Ret'])
    _output.iloc[[3], [0]] = skew(_hml_smb['HML_Ret'])
    _output.iloc[[3], [1]] = skew(_hml_smb['SMB_Ret'])
    _compare = pd.merge(_hml_smb, df6, on=['year', 'month'], how='left')
    _output.iloc[[4], [0]] = _compare.corr().loc['HML_Ret', 'HML']
    _output.iloc[[4], [1]] = _compare.corr().loc['SMB_Ret', 'SMB']

    # rename the output stats
    _output.rename(
        columns={0: 'HML', 1: 'SMB'},
        index={0: 'ret', 1: 'sd', 2: 'SR', 3: 'skew', 4: 'corr'},
        inplace=True)
    return _output


hml_smb_portfolio_stats = hml_smb_portfolio_replic(comp, pens, link, crsp, dlret, ff)
