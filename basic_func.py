# !usr/bin/env/python3
# -*- encoding:utf-8 -*-
# @date     : 2020/03/03
# @author   : CHEN WEI
# @filename : BasicFunc.py
# @software : pycharm

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse


class BasicUtils:
    """
    In this part, we write some basic functions that are helpful.
    >>> raw_data = pd.read_csv('Load_clean_data/valuation.csv')
    """
    @staticmethod
    def df_division(data, col_name, n_group=5, ascending=False):
        """
        Divide a dataframe into several part according to a specific column.
        For example, we want to divide valuation data of a single industry on a specific day into 5 part
        according to market capitalization, this function is helpful.

        :param data:
        :param col_name:
        :param n_group:
        :param ascending:
        :return: division:
        :examples:
        >>> data = raw_data[raw_data['day'] == '2019-12-31']
        >>> col_name = 'pe_ratio'
        """
        assert col_name in data.columns, '{} is not in columns of data!'.format(col_name)
        assert data[col_name].dtype == 'float' or data[col_name].dtype == 'int', \
            'type of {} is not comparable!'.format(col_name)

        data.reset_index(drop=True, inplace=True)
        rows = data.shape[0]
        rows_each_group = rows // n_group
        data.sort_values(by=col_name, ascending=ascending, inplace=True)
        data.reset_index(drop=True, inplace=True)

        division = []
        for i in range(n_group):
            if not i == n_group-1:
                division.append(data.iloc[i * rows_each_group: (i+1) * rows_each_group, :])
            else:
                division.append(data.iloc[i * rows_each_group:, :])

        return division

    @staticmethod
    def _standardize(sequence, multiples=5):  # TODO: seems a little slow. Faster it later. 2020-03-05
        """
        standardize a sequence: firstly, remove extreme values of this sequence; secondly, subtract mean value
        and divide standard value.

        :param sequence:
        :param multiples: float or None.
                    if float, remove extreme values. if None, do not remove extreme values.
        :return:
        :examples:
        >>> sequence = [1, 2, np.nan, 4, 5]
        >>> sequence = pd.Series([1, 2, np.nan, 4, 5])
        >>> BasicUtils._standardize(sequence)
        """
        if multiples is not None:
            median = np.median(list(filter(lambda x: not pd.isnull(x), sequence)))
            middle_sequence = [abs(x - median) for x in sequence]
            new_median = np.median(list(filter(lambda x: not pd.isnull(x), middle_sequence)))

            # %% remove extreme values.
            for index, D in enumerate(sequence):
                if D > median + multiples * new_median:
                    sequence[index] = median + multiples * new_median
                if D < median - multiples * new_median:
                    sequence[index] = median - multiples * new_median

        # %% standardize. 标准化以后的NA值是保留了的
        std = np.nanstd(sequence)
        mean = np.nanmean(sequence)
        sd_sequence = [(x-mean)/std for x in sequence]

        return sd_sequence

    @staticmethod
    def df_one_hot_encode(data, col_name):
        """

        :param data:
        :param col_name:
        :return:
        :examples:
        >>> data = raw_data[raw_data['day'] == '2019-12-31'].head(5)
        >>> col_name = 'ind_name'
        >>> BasicUtils.df_one_hot_encode(data, col_name)
        """
        assert col_name in data.columns, '{} is not in columns of data!'.format(col_name)

        category = data[col_name].unique().tolist()
        for ind in category:
            data[ind] = [int(x == ind) for x in data[col_name]]

        return data


class DateUtils:
    """
    For some basic functions to deal with date.
    """
    @staticmethod
    def _to_ceiling_busi_day(date):
        """
        Map a day to the last business day of its month.

        :param date: str or any other format of datetime.
        :return ceiling_busi_day: str. Last business day of this month.
        :examples:
        >>> DateUtils._to_ceiling_busi_day('2020-01-01')
        >>> DateUtils._to_ceiling_busi_day('2020-01-31')
        >>> DateUtils._to_ceiling_busi_day('2019-11-30')
        >>> DateUtils._to_ceiling_busi_day('2019-12-21')
        """
        try:
            date = parse(date)
        except TypeError:
            date = date

        date = date.replace(day=1)
        ceiling_busi_day = pd.date_range(date, periods=1, freq='BM').strftime('%Y-%m-%d')[0]
        return ceiling_busi_day

    @staticmethod
    def to_ceiling_busi_day(*args):
        """
        Map multiple days to the last business days of their month.

        :param args:
        :return: list. list of date in format of string.
        :examples:
        >>> DateUtils.to_ceiling_busi_day(*('2020-01-01', '2020-02-23'))
        >>> DateUtils.to_ceiling_busi_day('2020-01-01', '2020-02-23')
        """
        ceiling_busi_day = [DateUtils._to_ceiling_busi_day(date) for date in args]

        return ceiling_busi_day

    @staticmethod
    def _to_floor_busi_day(date):
        """
        Map a day to the previous last business day of its month.

        :param date:
        :return:
        :examples:
        >>> date = '2019-01-01'
        >>> DateUtils._to_floor_busi_day('2019-01-01')
        >>> DateUtils._to_floor_busi_day('2020-02-29')
        >>> DateUtils._to_floor_busi_day('2019-12-31')
        """
        try:
            date = parse(date)
        except TypeError:
            date = date

        date = date + relativedelta(months=-1)
        date = date.replace(day=1)
        floor_busi_day = pd.date_range(date, periods=1, freq='BM').strftime("%Y-%m-%d")[0]

        return floor_busi_day

    @staticmethod
    def to_floor_busi_day(*args):
        """

        :param args:
        :return:
        :examples:
        >>> DateUtils.to_floor_busi_day(*('2020-01-01', '2020-02-23'))
        >>> DateUtils.to_floor_busi_day('2020-01-01', '2020-02-23')
        """
        floor_busi_day = [DateUtils._to_floor_busi_day(date) for date in args]

        return floor_busi_day

    @staticmethod
    def _to_next_ceiling_busi_day(date):
        """
        Map a day to last business day of next month.

        :param date:
        :return:
        >>> date = '2020-01-31'
        >>> DateUtils._to_next_ceiling_busi_day('2019-12-31')
        """
        try:
            date = parse(date)
        except TypeError:
            date = date

        date = date + relativedelta(months=+1)
        date = DateUtils._to_ceiling_busi_day(date)

        return date


