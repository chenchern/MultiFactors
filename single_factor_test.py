# -*- encoding:utf-8 -*-
# @date     : 2020/03/03
# @author   : CHEN WEI
# @filename : Single_Factor_Test.py
# @software : pycharm

import pandas as pd
import numpy as np
# import statsmodels.api as sm
import math
from basic_func import BasicUtils, DateUtils

"""
We provide three method to verify whether a single factor is useful.
"""


class RegressionTestUnderflow(AttributeError):
    pass


class FactorInitial:
    """
    Initialize factor data that can be used for effectiveness test.
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


class RegressionTest(FactorInitial):
    """
    Regression method to test single factor. Key idea is to calculate t-value sequences.

    """
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
        >>> start_date = '2019-09-01'
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


class ICTest(FactorInitial):
    """
    IC method to test single factor.

    """
    def _ic(self, _date):
        """
        Calculate IC of a single factor on a specific day.

        :param: _date: string.
        :return: pearson_ic: float, IC value.
        :examples:
        >>> _date = '2019-05-31'
        >>> self = ICTest(raw_data, 'ep')
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

        :return: list. IC values.
        :examples:
        >>> start_date = '2019-05-01'
        >>> end_date = '2019-12-31'
        >>> self = ICTest(raw_data, 'ep', start_date, end_date)
        """
        if 'date_range' not in list(self.__dict__.keys()):
            raise RegressionTestUnderflow('in RegressionTest.regressions!')

        ic_seq = [self._ic(_date) for _date in self.date_range]
        ic_seq = list(filter(None.__ne__, ic_seq))  # %% 把ic_seq里的None去掉

        return ic_seq


class HierBackTest(FactorInitial):
    """
    Hierarchical backtest method to test single factors.
    """
    def stocks_div_single_ind(self, _date, industry, hier_num=5):
        """
        分层回测模型中将某个交易日单个行业内的全部股票根据因子值分为若干层，函数不做扩展，将所有股票处理成等权。

        :param _date:
        :param industry:
        :param hier_num:
        :return: hier: list of dict.
        :examples:
        >>> _date = '2018-01-31'
        >>> industry = '医药生物I'
        >>> industry = '交通运输I'
        >>> self = HierBackTest(raw_data, 'ep')
        >>> self.stocks_div_single_ind(_date, industry, 10)
        """
        cond_ind = self.data['ind_name'] == industry
        cond_date = self.data['date'] == _date
        data = self.data[cond_ind & cond_date]
        data = data.sort_values(self.factor_name, ascending=False).reset_index(drop=True)

        stock_num = data.shape[0]
        stock_each_hier = stock_num / hier_num  # %% 不直接按权值来处理，直接考虑n只股票均分
        assert stock_each_hier > 1, "there are too many hierarchy! Please enter a smaller number!"

        hier = []
        for k in range(1, hier_num + 1):
            ceil = math.ceil((k-1) * stock_each_hier)
            floor = math.floor(k * stock_each_hier)
            index = np.arange(ceil+1, floor+1) - 1
            # index = list(map(lambda x: int(x), index))
            _hier = dict(zip(data.iloc[index]['code'], [1] * len(index)))

            # %% 细节在于分割点处的那只股票如何按比例划分入左右相邻的层次里
            if k == 1 and (stock_num * k) % hier_num != 0:
                _hier[data.iloc[index[-1] + 1]['code']] = math.modf(k * stock_each_hier)[0]  # %% 保存最后一只票的权值
            elif k == hier_num and (stock_num * (k-1)) % hier_num != 0:
                _hier[data.iloc[index[0] - 1]['code']] = 1 - math.modf((k - 1) * stock_each_hier)[0]  # %% 保存第一只票的权值
            else:
                if (stock_num * (k-1)) % hier_num != 0:
                    _hier[data.iloc[index[0] - 1]['code']] = 1 - math.modf((k - 1) * stock_each_hier)[0]
                if (stock_num * k) % hier_num != 0:
                    _hier[data.iloc[index[-1] + 1]['code']] = math.modf(k * stock_each_hier)[0]

            hier.append(_hier)

        return hier

    def stocks_div_all_inds(self, _date, ind_weight=None, hier_num=5):
        """
        给定某个日期全行业之间的权重占比，这个权重保持和当天基准指数的行业配比相同，计算出各层的所有股票数，基准为1股
        行业之间的权重占比可以按照各行业的市值比给出

        :param _date:
        :param ind_weight: dict. weight of all industries.
        :param hier_num:
        :return:
        >>> self = HierBackTest(raw_data, 'ep')
        >>> all_ind_hier = self.stocks_div_all_inds(_date, hier_num=17)
        >>> [len(d.values()) for d in all_ind_hier]
        """
        ind = self.data['ind_name'].unique().tolist()
        if ind_weight is None:
            ind_weight = dict(zip(ind, [1/len(ind)] * len(ind)))

        # all_ind_hier = [dict()] * hier_num
        all_ind_hier = [dict() for q in range(hier_num)]  # %% 上面这句话太坑了，浅拷贝，更改一个dict，其他的dict都会被更改

        for _ind in ind:
            print(_ind)  # TODO: when hier_num is 17, IndexError.
            single_ind_hier = self.stocks_div_single_ind(_date, _ind, hier_num)

            # %% 对单行业内的每个层的股票都乘上该行业的权值
            for i, _hier in enumerate(single_ind_hier):
                _hier = {k: v * ind_weight[_ind] for k, v in _hier.items()}
                # print(i, _hier)
                all_ind_hier[i].update(_hier)

        return all_ind_hier

    # def