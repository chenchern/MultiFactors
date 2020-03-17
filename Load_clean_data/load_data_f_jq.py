# !usr/bin/env/python3
# -*- encoding:utf-8 -*-
# @date     : 2020/03/02
# @author   : CHEN WEI
# @filename : load_data_f_jq.py
# @software : pycharm

from jqdatasdk import *
from basic_func import DateUtils
import os


param = dict()
param['user_name'] = '13102002507'
param['security_code'] = 'kaKA@779432436'
param['end_date'] = '2019-12-31'
param['start_date'] = '2009-12-01'

# %% 登录
auth(param['user_name'], param['security_code'])

# %% 获取申万一级行业的行业代码和股票代码
industries = get_industries(name='sw_l1', date=param['end_date'])
ind_code = industries.index.tolist()
stock_code = [get_industry_stocks(code, date=param['end_date']) for code in ind_code]

# %% 获取行业代码和股票代码的对应
ind_stock_dict = dict(zip(ind_code, stock_code))
ind_stock_df = pd.DataFrame.from_dict(ind_stock_dict, orient='index').\
    transpose().stack().reset_index().iloc[:, -2:].dropna()
ind_stock_df.rename(columns={'level_1': 'ind_code', 0: 'stock_code'}, inplace=True)

ind_code_name_dict = dict(zip(ind_code, industries.name))
ind_stock_df['ind_name'] = [ind_code_name_dict[x] for x in ind_stock_df['ind_code']]


# let's make some cool method. Flat list of list into a list.
stock_code = [item for sublist in stock_code for item in sublist]  # 申万一级行业包含的全部股票

# %% 获取每月最后一个交易日的全股票估值数据
date = pd.date_range(param['start_date'], param['end_date'], freq='BM').strftime('%Y-%m-%d').tolist()
q = query(valuation).filter(valuation.code.in_(stock_code))

valuation_data = pd.DataFrame()
for _date in date:
    fd = get_fundamentals(q, date=_date)
    valuation_data = pd.concat([valuation_data, fd], ignore_index=True)

valuation_data = pd.merge(valuation_data, ind_stock_df, how='left', left_on='code', right_on='stock_code')
valuation_data.to_csv(os.path.join(os.getcwd(), 'Load_clean_data/valuation.csv'), encoding='utf_8_sig')

"""
对于stock_code中的一些股票，估值数据和量价数据都没有，但是聚宽的函数api底层代码的问题，get_fundamentals遇到没有的股票代码不会报错，
但是get_price遇到没有的股票代码会报错，所以我们提取valuation_data的stock_code数据出来
"""
valuation_data = pd.read_csv('Load_clean_data/valuation.csv', index_col=0)

# %% 获取每月最后一个交易日的全股票收盘价数据
stock_code = valuation_data.stock_code.unique().tolist()
price_data = get_bars(stock_code, count=150, unit='1M', fields=['date', 'close'],
                      end_dt=param['end_date'], include_now=True)
price_data = price_data.reset_index(drop=False).drop(['level_1'], axis=1).rename(columns={'level_0': 'stock_code'})

"""
聚宽这个get_price的api真的很奇怪，按月提取的数据有些数据居然不是最后一个交易日的。目前没找到根据单独一天的日期来提取行情的接口，
所以暂时采用比较粗糙的做法：把取出来的那个月的数据日期映射为当月最后一个交易日日期。
"""
price_data['date'] = DateUtils.to_ceiling_busi_day(*price_data['date'])

# %% 保存price_data
price_data.to_csv(os.path.join(os.getcwd(), 'Load_clean_data/price.csv'))
"""
需要计算全部股票的月度收益率，但是从聚宽中调取出来的数据如果股票停牌，当天是直接缺失数据的，而不是补充为NA,所以不能直接
shift并作差，需要先补齐所有交易日，然后shift作差，对应停牌当天及下一天的收益率都为NA
"""
pnl = pd.DataFrame()
for code in stock_code:
    _price_data = price_data[price_data['stock_code'] == code].sort_values(by='date').reset_index(drop=True)
    all_trade_date = pd.date_range(_price_data['date'][0], param['end_date'], freq='BM').strftime('%Y-%m-%d').tolist()
    _new_data = pd.merge(pd.DataFrame({'date': all_trade_date}), _price_data, on='date', how='left')
    _new_data['stock_code'] = _new_data['stock_code'].unique()[0]
    _new_data['pnl'] = _new_data['close'] / _new_data['close'].shift(1) - 1
    pnl = pd.concat([pnl, _new_data], ignore_index=True)

pnl.to_csv(os.path.join(os.getcwd(), 'Load_clean_data/price_pnl.csv'))

# %% 获取沪深300每月最后一个交易日的收盘价数据
hs300 = get_bars('000300.XSHG', count=150, unit='1M', fields=['date', 'close'],
                 end_dt=param['end_date'], include_now=True)