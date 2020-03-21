# -*- encoding:utf-8 -*-
# @date     : 2020/03/03
# @author   : CHEN WEI
# @filename : Single_Factor_Test.py
# @software : pycharm

import pandas as pd
import numpy as np
import statsmodels.api as sm
from basic_func import BasicUtils, DateUtils

"""
We provide three method to verify whether a single factor is useful.
"""


class RegressionTestUnderflow(AttributeError):
    pass


class RegressionTest:
    """
    Regression method to test single factor. Key idea is to calculate t-value sequences.
    >>> valuation = pd.read_csv('./Load_clean_data/valuation.csv', index_col=0).rename(columns={'day': 'date'})
    >>> price = pd.read_csv('./Load_clean_data/price_pnl.csv', index_col=0)
    >>> price['date'] = DateUtils.to_floor_busi_day(*price['date'])  # %% 因为回归法测试的时候要用下一期的股票收益率
    >>> raw_data = pd.merge(valuation, price, on=['stock_code', 'date'], how='left')
    >>> raw_data['ep'] = 1 / raw_data['pe_ratio']
    """
    def __init__(self, data, factor_name, start_date=None, end_date=None, freq='BM'):
        """
        注意数据的pnl这一列存的是不是对应日期那一期对上一期的收益率，而是下一期对当期的收益率

        :param data: pd.DataFrame.
                data of a single factor with columns: e.g ['ind_name', 'stock_code', 'date', 'market_cap', 'ep', 'pnl']
        :param factor_name: string.
        :param start_date: string. in format of '2020-01-01'.
        :param end_date: string. in format of '2020-01-01'.
        :param freq: string.
        """
        self._pub_column = ['ind_name', 'stock_code', 'date', 'market_cap', 'pnl']
        assert np.array([col in data.columns.values for col in self._pub_column]).all(), \
            'missing column or incorrect name in input data!'

        # data的初始化先把市值列和月收益列的空值删掉，即从股票池剔除交易日停牌的股票
        self.data = data.dropna(subset=['market_cap', 'pnl'])
        self.factor_name = factor_name
        self.start_date = start_date
        self.end_date = end_date

        try:
            self.date_range = pd.date_range(self.start_date, self.end_date, freq=freq).strftime('%Y-%m-%d')
        except ValueError:
            pass

    def _regression(self, _date):
        """
        Regression a single factor on a specific day.

        :param _date: string.
        :return: tuple. t-value and coef of this single factor.
        :examples:
        >>> self = RegressionTest(raw_data, 'ep')
        >>> _date = '2018-01-31'
        >>> self._regression(_date)
        """
        data = self.data[self.data['date'] == _date].reset_index(drop=True)
        if data.shape[0] == 0:
            print("in RegressionTest._regression: Empty dataframe on {}!".format(_date))
            return

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

        return res.tvalues[self.factor_name], res.params[self.factor_name]

    def regressions(self):
        """
        Regression of a single factor on sequential days.

        :return:
        :examples:
        >>> start_date = '2019-02-01'
        >>> end_date = '2019-12-31'
        >>> self = RegressionTest(raw_data, 'ep', start_date, end_date)
        >>> test = self.regressions()
        """
        if 'date_range' not in list(self.__dict__.keys()):
            raise RegressionTestUnderflow('in RegressionTest.regressions!')

        parameters = [self._regression(_date) for _date in self.date_range]  # TODO: try multiprocess
        parameters = list(filter(None.__ne__, parameters))
        t_value_seq = [param[0] for param in parameters]
        coef_seq = [param[1] for param in parameters]

        return t_value_seq, coef_seq

    def _ic(self, _date):
        """
        Calculate IC of a single factor on a specific day.

        :param: _date: string.
        :return: pearson_ic: float, IC value.
        :examples:
        >>> _date = '2019-05-31'
        >>> self = RegressionTest(raw_data, 'ep')
        """
        data = self.data[self.data['date'] == _date].reset_index(drop=True)
        if data.shape[0] == 0:
            print("in RegressionTest._ic: Empty dataframe on {}!".format(_date))
            return

        # %% 对因子列和市值列均做标准化处理
        data[self.factor_name] = BasicUtils._standardize(data[self.factor_name])
        data[self.factor_name] = data[self.factor_name].fillna(0)
        data['market_cap'] = BasicUtils._standardize(data['market_cap'], multiples=None)

        ind = data['ind_name'].unique().tolist()
        data = BasicUtils.df_one_hot_encode(data, 'ind_name').drop('ind_name', axis=1)

        ind.append('market_cap')
        x = sm.add_constant(data[ind])
        y = data[self.factor_name]

        reg = sm.OLS(y, x)
        res = reg.fit()

        residule = res.resid
        pearson_ic = np.corrcoef(data['pnl'], residule)[0, 1]

        return pearson_ic

    def ics(self):
        """

        :return:
        :examples:
        >>> start_date = '2019-05-01'
        >>> end_date = '2019-12-31'
        >>> self = RegressionTest(raw_data, 'ep', start_date, end_date)
        """
        if 'date_range' not in list(self.__dict__.keys()):
            raise RegressionTestUnderflow('in RegressionTest.regressions!')

        ic_seq = [self._ic(_date) for _date in self.date_range]
        ic_seq = list(filter(None.__ne__, ic_seq))

        return ic_seq









