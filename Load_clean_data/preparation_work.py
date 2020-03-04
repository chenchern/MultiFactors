# !usr/bin/env/python3
# -*- encoding:utf-8 -*-
# @date     : 2020/03/03
# @author   : CHEN WEI
# @filename : preparation_work.py
# @software : pycharm

import os
import pandas as pd
import numpy as np
from basic_func import BasicUtils

param = dict()
param['feature'] = ['ind_name', 'stock_code', 'day', 'pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio', 'market_cap']
param['date'] = '2019-12-31'

_dir = os.getcwd()
file = os.path.join(_dir, 'valuation.csv')
raw_data = pd.read_csv(file, index_col=0)[param['feature']]

# %% 计算各行业在2019-12-31这天的pb均值
ind_group = raw_data[raw_data['day'] == param['date']].groupby('ind_name').mean()
ind_group.sort_values(by='pe_ratio', ascending=False)

# %% 将某一天某个行业的全部个股EP值从大到小分为5组
data = raw_data[['ind_name', 'stock_code', 'day', 'pe_ratio', 'market_cap']]
data['ep'] = list(1 / data['pe_ratio'])
year = pd.date_range('2010-01-01', '2020-01-01', freq='BY').strftime("%Y-%m-%d").tolist()

pe_by_year = dict()
for _year in year:
    division = []
    for i, df in data[data['day'] == _year].groupby('ind_name'):
        if len(division) == 0:
            division = BasicUtils.df_division(df, col_name='market_cap', n_group=5)
        else:
            division = [pd.concat([a, b], ignore_index=True)
                        for a, b in zip(division, BasicUtils.df_division(df, col_name='market_cap', n_group=5))]
    pe_by_year[_year] = [np.mean(element['ep']) for element in division]

pe_by_year_df = pd.DataFrame(pe_by_year)
