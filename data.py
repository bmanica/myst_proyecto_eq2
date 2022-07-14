
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Fundamental Analysis Trading System                                                        -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: https://github.com/bmanica/myst_proyecto_eq2                                            -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# ====================================== Required packages ================================================ #

### Libraries to use
import pandas as pd
import datetime


# ================================= Data object definition ================================================ #

### Data reading from local carpet
usdmxn_18 = pd.read_csv('files/MP_M1_2018.csv')
usdmxn_19 = pd.read_csv('files/MP_M1_2019.csv')
usdmxn_20 = pd.read_csv('files/MP_M1_2020.csv')

# Merge all usdmxn data
usdmxn = pd.concat([usdmxn_18, usdmxn_19, usdmxn_20], ignore_index=True)

# Unemployment rate for three years
unemployment = pd.read_excel('files/Unemployment_Rate2.xlsx')
unemployment['Datetime'] = pd.to_datetime(unemployment[['Date ', 'Time (UTC -05:00) ']].agg(' '.join,
                                                                                            axis=1))
unemployment = unemployment.sort_values(by='Datetime')

# Fix datetime structures for usdmxn dataframe
usdmxn['timestamp'] = pd.to_datetime(usdmxn['timestamp'])
usdmxn['date'] = [i.date() for i in usdmxn['timestamp']]

### Subset in order to get just the USDMXN data for the dates in unemployment indicator
usdmxn_data = pd.DataFrame()

for i in unemployment['Datetime']:
    sub_set = usdmxn[usdmxn['date'] == i.date()]
    delta_down = i - datetime.timedelta(minutes=30)
    delta_up = i + datetime.timedelta(minutes=30)

    sub_set = sub_set[(sub_set['timestamp'] >= delta_down) & (sub_set['timestamp'] <= delta_up)]
    usdmxn_data = pd.concat([usdmxn_data, sub_set], ignore_index=True)

bad_data = usdmxn_data.copy()

usdmxn_data['open'] = 1 / bad_data['open']
usdmxn_data['high'] = 1 / bad_data['low']
usdmxn_data['low'] = 1 / bad_data['high']
usdmxn_data['close'] = 1 / bad_data['close']
usdmxn_data['volume'] = round(bad_data['volume'] * 500000 / usdmxn_data['close'])

fx_data = usdmxn_data

### Subset the required data for all trading living period
usdmxn_opt = usdmxn[usdmxn['timestamp'] >= pd.to_datetime(unemployment.index.values[0])]

bad_data_opt = usdmxn_opt.copy()

usdmxn_opt['open'] = 1 / bad_data_opt['open']
usdmxn_opt['high'] = 1 / bad_data_opt['low']
usdmxn_opt['low'] = 1 / bad_data_opt['high']
usdmxn_opt['close'] = 1 / bad_data_opt['close']
usdmxn_opt['volume'] = round(bad_data_opt['volume'] * 500000 / usdmxn_opt['close'])

# Split into training and test sets
training_usdmxn = usdmxn_opt[(pd.to_datetime(usdmxn_opt['date'])>=pd.to_datetime('2018-01-01')) &
                             (pd.to_datetime(usdmxn_opt['date'])<=pd.to_datetime('2019-01-01'))]

test_usdmxn = usdmxn_opt[(pd.to_datetime(usdmxn_opt['date'])>=pd.to_datetime('2019-01-02')) &
                         (pd.to_datetime(usdmxn_opt['date'])<=pd.to_datetime('2020-01-02'))]




