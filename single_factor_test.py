# -*- encoding:utf-8 -*-
# @date     : 2020/03/03
# @author   : CHEN WEI
# @filename : Single_Factor_Test.py
# @software : pycharm

import pandas as pd
import numpy as np
import statsmodels.api as sm
from basic_func import BasicUtils

"""
We provide three method to verify whether a single factor is useful.
"""


class RegressionTest:
    """
    Regression method to test single factor. Key idea is to calculate t-value sequences.
    >>> valuation = pd.read_csv('./Load_clean_data/valuation.csv', index_col=0).rename(columns={'day': 'date'})
    >>> price = pd.read_csv('./Load_clean_data/price_pnl.csv', index_col=0)
    >>> raw_data = pd.merge(valuation, price, on=['stock_code', 'date'], how='left')
    >>> raw_data['ep'] = 1 / raw_data['pe_ratio']
    """
    def __init__(self, data, factor_name, start_date=None, end_date=None, freq='BM'):
        """

        :param data: pd.DataFrame.
                data of a single factor with columns: e.g ['ind_name', 'stock_code', 'date', 'market_cap', 'ep', 'pnl']
        :param factor_name: string.
        :param start_date: string. in format of '2020-01-01'.
        :param end_date: string. in format of '2020-01-01'.
        :param freq: string.
        """
        self.__pub_column = ['ind_name', 'stock_code', 'date', 'market_cap', 'pnl']
        assert np.array([col in data.columns.values for col in self.__pub_column]).all(), \
            'missing column or incorrect name in input data!'

        # data的初始化先把市值列和月收益列的空值删掉，即从股票池剔除交易日停牌的股票
        self.data = data.dropna(subset=['market_cap', 'pnl'])
        self.factor_name = factor_name
        self.start_date = start_date
        self.end_date = end_date

        try:
            self.date_range = pd.date_range(self.start_date, self.end_date, freq=freq)
        except ValueError:
            pass

    def _regression(self, _date):
        """
        Regression process of a single factor on a specific day.

        :param _date: string.
        :return:
        :examples:
        >>> self = RegressionTest(raw_data, 'ep')
        >>> _date = '2019-12-31'

        """
        data = self.data[self.data['date'] == _date].reset_index(drop=True)
        data[self.factor_name] = BasicUtils._standardize(data[self.factor_name])  # %% 标准化
        data[self.factor_name] = data[self.factor_name].fillna(0)  # %% 对缺失值的处理取标准化后序列的均值：因子值与全市场情况相同

        # 因子按市值加权
        weight = np.sqrt(data['market_cap'])  # %% weight of WLS.
        data['pnl'] = weight * data['pnl']
        data[self.factor_name] = weight * data[self.factor_name]

        ind = data['ind_name'].unique().tolist()
        data = BasicUtils.df_one_hot_encode(data, 'ind_name').drop('ind_name', axis=1)  # %% 哑变量处理

        ind.append(self.factor_name)
        x = sm.add_constant(data[ind])
        y = data['pnl']

        reg = sm.OLS(y, x)
        res = reg.fit()




