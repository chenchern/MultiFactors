# -*- encoding:utf-8 -*-
# @date     : 2020/03/03
# @author   : CHEN WEI
# @filename : Single_Factor_Test.py
# @software : pycharm

import pandas as pd

"""
We provide three method to verify whether a single factor is useful.
"""


class RegressionTest:
    """
    Regression method to test single factor. Key idea is to calculate t-value sequences.
    >>> valuation = pd.read_csv('./Load_clean_data/valuation.csv', index_col=0).rename(columns={'day': 'date'})
    >>> price = pd.read_csv('./Load_clean_data/price_pnl.csv', index_col=0)
    >>> raw_data = pd.merge(valuation, price, on=['stock_code', 'date'], how='left')
    """
    def __init__(self, data, start_date=None, end_date=None):
        """

        :param data: 
        :param :
        """
        self.data = data
        self.start_date = start_date
        self.end_date = end_date

    def _regression(self):
        return
