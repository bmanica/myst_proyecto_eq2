
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: FUNDAMENTAL ANALISIS                                                                       -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: lyhaesp, bmanica, JuanPablo2019, ACristinaVV.                                               -- #
# -- license: GNU General Public License v3.0                                                            -- #
# -- repository: https://github.com/bmanica/myst_proyecto_eq2                                            -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go


def data_manipulation(forex_1, forex_2, forex_3, indicator):
    """
    This function creates a data frames for further use. It only accepts 3 year worth of data.
    It requires information of a designated currency and a chosen indicator.
    Parameters
    ----------
    forex_1 : CSV File
           The first year of currency data

    forex_2 : CSV File
           The second year of currency data

    forex_3 : CSV File
           The third year of currency data

    indicator : CSV File
           All three years worth of indicator data

    """
    'Manipulates data so its easier to work with it'
    # Concatenate data
    usdmxn = pd.concat([forex_1, forex_2, forex_3], ignore_index=True)  # Merge all usdmxn data

    # Unemployment rate for three years
    indicator['Datetime'] = pd.to_datetime(indicator[['Date ', 'Time (UTC -05:00) ']].agg(' '.join, axis=1))

    # Fix datetime structures for both data frames
    usdmxn['timestamp'] = pd.to_datetime(usdmxn['timestamp'])
    usdmxn['date'] = [i.date() for i in usdmxn['timestamp']]

    return usdmxn, indicator


def Scenario_Clasification(indicator):
    """
    This function clasifies the indicator data according to pre-set scenerios
    Parameters
    ----------
    indicator : Dataframe
           All three years worth of indicator data

    """

    df = indicator.copy()

    def escenarios(df):
        """
        This function states the posible sceneries
        Parameters
        ----------
        df : Dataframe
               All three years worth of indicator data

        """

        if df['Actual '] >= df['Consensus '] >= df['Prior ']:
            return 'A'

        if df['Actual '] >= df['Consensus '] < df['Prior ']:
            return 'B'

        if df['Actual '] < df['Consensus '] >= df['Prior ']:
            return 'C'

        if df['Actual '] < df['Consensus '] < df['Prior ']:
            return 'D'

    df['Scenario'] = df.apply(escenarios, axis=1)
    df = df[['Country/Region ', 'Event ', 'Importance ', 'Period ', 'Actual ', 'Consensus ', 'Prior ', 'Datetime',
             'Scenario']]
    df = df.set_index('Datetime')

    return df


def Event_Data(usdmxn, unemployment):
    """
    This function return a dataframe for each indicator event. This includes 30 minutes prior and 30 after the indicator has
    been annouced.
    Parameters
    ----------
    usdmxn : Dataframe
        All three years worth of forex data

    """

    bad_data_2 = usdmxn.copy()

    usdmxn['open'] = 1 / bad_data_2['open']
    usdmxn['high'] = 1 / bad_data_2['low']
    usdmxn['low'] = 1 / bad_data_2['high']
    usdmxn['close'] = 1 / bad_data_2['close']
    usdmxn['volume'] = bad_data_2['volume'] * 500000 / usdmxn['close']

    mxnusd = usdmxn

    eventos_df = []

    for i in unemployment['Datetime']:
        sub_set_2 = mxnusd[mxnusd['date'] == i.date()]
        delta_down_2 = i - datetime.timedelta(minutes=30)
        delta_up_2 = i + datetime.timedelta(minutes=30)
        sub_set_2 = sub_set_2[(sub_set_2['timestamp'] >= delta_down_2) & (sub_set_2['timestamp'] <= delta_up_2)]
        eventos_df.append(sub_set_2.reset_index())

    for i in range(len(eventos_df)):
        if unemployment['Datetime'][i] not in eventos_df[i]['timestamp']:
            row = sum([eventos_df[i]['timestamp'] < unemployment['Datetime'][i]])
            row = row.cumsum()[len(row) - 1] - 1
            dif = unemployment['Datetime'][i] - eventos_df[i]['timestamp'][row]
            eventos_df[i]['timestamp'][row] = eventos_df[i]['timestamp'][row] + dif

    return eventos_df


def Metrics(eventos_df, unemployment_df):
    """
    This function returns a consolidated dataframe for metrics like direction, bullish pip, bear pip and volatility,
    Parameters
    ----------
    eventos_df : Dataframe
        Dataframe that contains trading informartion related to the chosen currency for each indicator event.

    unemployment : Dataframe
        Dataframe that contains indicator informartion.
    """
    # 1. Direction
    pos_t0 = []
    for i in range(len(eventos_df)):
        # Getting timestamp index at t=0
        x = eventos_df[i].index[
            eventos_df[i]['timestamp'] == eventos_df[i]['timestamp'][0] + datetime.timedelta(minutes=30)].tolist()
        pos_t0.append(x)

    dir_signo = [eventos_df[i]['close'][len(eventos_df[i]) - 1] - eventos_df[i]['open'][pos_t0[i][0]] for i in
                 range(len(pos_t0))]
    direccion = [1 if dir_signo[i] >= 0 else -1 for i in range(len(dir_signo))]

    # 2. Bullish Pips
    high_series = []

    for i in range(len(eventos_df)):
        high_list = eventos_df[i]['high'].tolist()
        high_value = max(high_list[pos_t0[i][0]:len(high_list)])
        high_series.append(high_value)

    # pips list
    pips_high = [high_series[i] - eventos_df[i]['open'][pos_t0[i][0]] for i in range(len(eventos_df))]

    # 3. Bear Pips
    low_series = []

    for i in range(len(eventos_df)):
        low_list = eventos_df[i]['low'].tolist()
        low_value = min(low_list[pos_t0[i][0]:len(low_list)])
        low_series.append(low_value)

    # pips list
    pips_low = [eventos_df[i]['open'][pos_t0[i][0]] - low_series[i] for i in range(len(eventos_df))]

    # 4. Voltility
    vol = [max(eventos_df[i]['high']) - min(eventos_df[i]['low']) for i in range(len(eventos_df))]

    # Creation of the dataframe
    pip_factor = 10000
    df_escenarios = unemployment_df.copy()
    df_escenarios = df_escenarios[['Scenario']]
    df_escenarios['Direction'] = direccion
    df_escenarios['Bullish_Pip'] = [pips_high[i] * pip_factor for i in range(len(pips_high))]
    df_escenarios['Bear_Pip'] = [pips_low[i] * pip_factor for i in range(len(pips_low))]
    df_escenarios['Volatility'] = [vol[i] * pip_factor for i in range(len(vol))]

    return df_escenarios

