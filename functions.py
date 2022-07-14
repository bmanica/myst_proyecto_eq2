
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: FUNDAMENTAL ANALISIS                                                                       -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: lyhaesp, bmanica, JuanPablo2019, ACristinaVV.                                               -- #
# -- license: GNU General Public License v3.0                                                            -- #
# -- repository: https://github.com/bmanica/myst_proyecto_eq2                                            -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# ================================ Required packages and scripts ========================================== #

### Libraries to use
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from scipy.stats import shapiro, normaltest, anderson, jarque_bera,kruskal,levene, bartlett
from statsmodels.tsa.stattools import adfuller,acf, pacf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import chart_studio.plotly 
import plotly.offline as pyo
from plotly.offline import iplot
from pyswarm import pso


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

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#                           Statistical functions
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

def dw(x, funcs=[acf,pacf]):
    """
    Docstring:
    
    The porpouse of this function is to compute the durbin watson test for the time series residuals,
    in order to know if there is autocorrelation present.
    Also the PACF and the ACF plot is compute.
    
    Parameters
    -------------
    x: time series
    funcs: an array with the acf, pacf function.
    
    Returns
    -------------
    returns three figures, which contain the result of the durbin watson test and the PACF and ACF plot.
    
    References
    -------------
    https://www.statology.org/durbin-watson-test-python/
    """

    X =np.arange(len(x))
    Y = np.asarray(x)
    X = sm.add_constant(X)

   
    ols_res = OLS(Y,X).fit()
    
    statistic = durbin_watson(ols_res.resid)
    
#     2: No autocorrelation. Generally, we assume 1.5 to 2.5 as no correlation.
# 0- <2: positive autocorrelation. The more close it to 0, the more signs of positive autocorrelation.
# >2 -4: negative autocorrelation. The more close it to 4, the more signs of negative autocorrelation
#     #save results
    coment = 0
    if statistic == 2:
        c = 'No autocorrelation'
    elif statistic >=0 and statistic <2:
        c = 'Positive autocorrelation'
    else:
        c = 'negative autocorrelation'
    results = np.array(['Durbin Watson',statistic,c])
    
    
    #table plotly
    trace = go.Table(
        header=dict(values=['<b>Statistical test</b>', 
                            '<b>Statistic</b>', '<b>Comment</b>'],
                    line = dict(width=0),
                    fill = dict(color='rgba(42,63,95,0.8)'),
                    align = 'center',
                    font = dict(
                        color = '#ffffff',
                        size = 12
                    )),
        cells=dict(values=np.array(results).T,
                   line = dict(width=0),
                   fill = dict(color=[['#EBF0F8', '#ffffff', '#EBF0F8', ]]),
                   align = 'center', 
                   height = 40),
        columnwidth=[0.3, 0.25, 0.3])
    layout = dict(
        height=300,
        margin=dict(
            l=5,
            r=5,
            t=30,
            b=0
        )
    )
    data = [trace]
    andar_table = dict(data=data, layout=layout)
    
    #pacf plot

    y_pacf = pacf(x, nlags=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x= np.arange(len(y_pacf)),
        y= y_pacf,
        name= 'PACF',
        ))
    
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title="Partial Autocorrelation",
        xaxis_title="Lag",
        yaxis_title="Partial Autocorrelation Fuction"
        )
    
    
    
    #acf plot
    y_acf = acf(x)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x= np.arange(len(y_acf)),
        y= y_acf,
        name= 'ACF',
        ))
    fig.show()
    
    fig2.update_xaxes(rangeslider_visible=True)
    fig2.update_layout(
        title="Autocorrelation",
        xaxis_title="Lag",
        yaxis_title="Autocorrelation Fuction"
        )
    fig2.show()
    




    return iplot(andar_table)



def var_test(x,alfa:float):
    """
    Docstring
    
    The porpouse of this function is to test if the time series have the heteroscedasticity property.
    
    Parameters
    -------------------
    x: time series
    alfa: 1- significance level
    
    Returns
    --------------------
    A chart with the results of Levene and Bartlett statistical tests.
    """
    
    # data split 50/50
    split = int(len(x)/2)
    
    sample_1 = x[0:split]
    sample_2 = x[split:len(x)]
    
    # levene test
    statistic = levene(sample_1,sample_2)
    p_value1 = statistic[1]
    
    #barlett
    statistic2 = bartlett(sample_1,sample_2)
    p_value2 = statistic2[1]
        
    #save the results in the dataframe
    results = pd.DataFrame()
    
    results['Statistical test'] = ['Levene','Bartlett']
    results['significance_level'] = np.ones(len([p_value1, p_value2]))*(1-alfa)
    results['Pvalue'] = [p_value1, p_value2]
    results['H0: Null Hypothesis'] = ['Homocedasticity','Homocedasticity']
    results['Inference Result'] = [ "Heterocedasticity" if i<=0.05 else "Homocedasticity" for i in [p_value1, p_value2]]    
    
    #return results
    
     #table plotly
    trace = go.Table(
        header=dict(values=['<b>Statistical test</b>', '<b>Confidence Level</b>', '<b>P-Value</b>',
                            '<b>Null Hypothesis</b>', '<b>Comment</b>'],
                    line = dict(width=0),
                    fill = dict(color='rgba(42,63,95,0.8)'),
                    align = 'center',
                    font = dict(
                        color = '#ffffff',
                        size = 12
                    )),
        cells=dict(values=np.array(results).T,
                   line = dict(width=0),
                   fill = dict(color=[['#EBF0F8', '#ffffff', '#EBF0F8', '#ffffff', '#EBF0F8']]),
                   align = 'center', 
                   height = 40),
        columnwidth=[0.3, 0.25, 0.3, 0.25, 0.5])
    layout = dict(
        height=300,
        margin=dict(
            l=5,
            r=5,
            t=30,
            b=0
        )
    )
    data = [trace]
    andar_table = dict(data=data, layout=layout)




    return iplot(andar_table)


def normality_test(x,alfa:float, funcs:list = [shapiro, normaltest, jarque_bera, anderson]):
    """
    Docstring
    
    The porpouse of this function is to compute normalility test for a time series with the following statistical tests:
    shapiro, d'angostino, jarque bera and anderson darling.
    
    Parameters
    ------------
    x: time series
    alfa: 1-significance level
    funcs: a predifined list with normal test functions.
    
    
    Return 
    ------------
    
    A chart with the normalility test results.
    
    References
    --------------
    https://plotly.com/python/v3/normality-test/
    """
    
    # save results in a dataframe
    results = pd.DataFrame()
    
    # test's names
    results['Statistical_test'] = ['Shapiro Wilks',"D' Angostino",'Jarque Bera','Anderson Darling']
    
    # level of confidence
    results['significance_level'] = np.ones(len(funcs))*(alfa)
    
    # P-values
    results['P-values'] = [funcs[0](x)[1],funcs[1](x)[1],
                          funcs[2](x)[1],funcs[3](x)[1][2]]
    
    #null hypothesis
    results['H0: Null Hypothesis'] = ['R.V distribute normal','R.V. distribute normal','R.V. distribute  normal','R.V. distribute normal']
    
    # result of the test
    results['Inference Result'] = ["R.V. distribute normal" if i>=(1-j) else "R.V.  don't distribute normal" 
                            for i,j in zip(results['P-values'],results['significance_level'])]
    
    results['significance_level'] = np.ones(len(funcs))*(1-alfa)
    
    # Anderson Darling Criteria
    
    
    anderson = funcs[3](x)
    
    if anderson.statistic < results.iloc[3,2]:
        results.iloc[3,4] = 'R.V. distribute normal'
    else:
        
         results.iloc[3,4] = "R.V. don't distribute normal"
            
    #table plotly
    trace = go.Table(
        header=dict(values=['<b>Statistical test</b>', '<b>Confidence Level</b>', '<b>P-Value</b>',
                            '<b>Null Hypothesis</b>', '<b>Comment</b>'],
                    line = dict(width=0),
                    fill = dict(color='rgba(42,63,95,0.8)'),
                    align = 'center',
                    font = dict(
                        color = '#ffffff',
                        size = 12
                    )),
        cells=dict(values=np.array(results).T,
                   line = dict(width=0),
                   fill = dict(color=[['#EBF0F8', '#ffffff', '#EBF0F8', '#ffffff', '#EBF0F8']]),
                   align = 'center', 
                   height = 40),
        columnwidth=[0.3, 0.25, 0.3, 0.25, 0.5])
    layout = dict(
        height=300,
        margin=dict(
            l=5,
            r=5,
            t=30,
            b=0
        )
    )
    data = [trace]
    andar_table = dict(data=data, layout=layout)

    return iplot(andar_table)



def seasonality(x,alfa:float,m:int):
        """
        Docstring
        
        The porpouse of this function is to check if the time series have a seasonal component,
        wirh the kruskal wallis test.
        
        Parameters
        ----------------
        x: time series
        alfa: 1-significance level
        m: periods of the timeseries if monthly 12, quarter, etc....
        
        
        Returns
        ----------------
        A chart with the results.
        
         Reference
        ----------------
        https://knk00.medium.com/how-to-determine-seasonality-without-plots-f18cee913b95
    
        """
        
        
       
        dx = np.arange(len(x.index)) % m
        statistic, pvalue = kruskal(x,dx)
#         if p_value <= alfa:
#             seasonal = True
            
        # save results in a dataframe
        results = pd.DataFrame()

        # test's names
        results['Statistical_test'] = ['Kruskal Wallis']

        # level of confidence
        results['significance_level'] = (1-alfa)

        # P-values
        results['P-value'] = pvalue

        #null hypothesis
        results['H0: Null Hypothesis'] = ['All samples have the same median.']
        # result of the test
        results['Inference Result'] = ['There is not seasonality' if i>alfa else 'There is seasonality' for i in results['P-value']]


        #table plotly
        trace = go.Table(
            header=dict(values=['<b>Statistical test</b>', '<b>Confidence Level</b>', '<b>P-Value</b>',
                                '<b>Null Hypothesis</b>', '<b>Comment</b>'],
                        line = dict(width=0),
                        fill = dict(color='rgba(42,63,95,0.8)'),
                        align = 'center',
                        font = dict(
                            color = '#ffffff',
                            size = 12
                        )),
            cells=dict(values=np.array(results).T,
                       line = dict(width=0),
                       fill = dict(color=[['#EBF0F8', '#ffffff', '#EBF0F8', '#ffffff', '#EBF0F8']]),
                       align = 'center', 
                       height = 40),
            columnwidth=[0.3, 0.25, 0.3, 0.25, 0.5])
        layout = dict(
            height=300,
            margin=dict(
                l=5,
                r=5,
                t=30,
                b=0
            )
        )
        data = [trace]
        andar_table = dict(data=data, layout=layout)

        return iplot(andar_table)


def iqr(x):
    """
    Docstring
    
    The porpouse of this function is to check if a time series have outliers useing the 
    IQR criterion.
    
    Parameters
    ------------------
    x: time series
    
    Returns
    ------------------
    a dataframe with the values that are consider outliers.
    
     References
    ------------------
    https://www.statology.org/interquartile-range-python/
    """


    q1 = x['Actual '].quantile(0.25)
    q3 = x['Actual '].quantile(0.75)
    iqr = q3-q1
    lw_limit = q1 -1.5*iqr
    upper_limit = q3+1.5*iqr

    outliers = x[x['Actual ']<lw_limit].append(x[x['Actual ']>upper_limit])
    return outliers['Actual ']


def qq(x,qqplot_data):
    """
    Docstring
    
    The porpouse of this function is to plot a qqplot.
    
    Parameters
    --------------
    x: time series
    qqplot_data: figure
    
    Returns
    --------------
    A qqplot figure.
    
    References
    --------------
    https://plotly.com/python/v3/normality-test/
    """
    

    
    #qqplot_data=qqplot(x, line='s',ax=ax).gca().lines
    
    
    fig = go.Figure()

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[0].get_xdata(),
        'y': qqplot_data[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': '#19d3f3'
        }
    })

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[1].get_xdata(),
        'y': qqplot_data[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': '#636efa'
        }

    })


    fig['layout'].update({
        'title': 'Quantile-Quantile Plot',
        'xaxis': {
            'title': 'Theoritical Quantities',
            'zeroline': False
        },
        'yaxis': {
            'title': 'Sample Quantities'
        },
        'showlegend': False,
        'width': 800,
        'height': 700,
    })
    
    return fig.show()

def stationarity(x,alfa:float):
    """
    Docstring
    
    The porpouse of this function is to test if the the time series is stationary usign the Augmented Dickey Fuller test.
    
    Parameters
    -----------------
    x: time series
    alfa: 1-significance level
    
    Returns
    ----------------
    A chart with the results of the test.
    
    References
    ----------------
    https://machinelearningmastery.com/time-series-data-stationary-python/
    """
    stat = adfuller(x)
    c=0
    if stat[1] <= alfa:
        c='Is Stationary'
    else:
        c='Not Stationary'


 
    results = np.array(['ADF',stat[0],1-alfa,'Not Stationary',stat[1],c])
    #table plotly
    trace = go.Table(
        header=dict(values=['<b>Statistical test</b>', 
                            '<b>Statistic</b>','<b>Confidence level</b>','<b>Null Hypothesis</b>','<b>P-value</b>','<b>Comment</b>'],
                    line = dict(width=0),
                    fill = dict(color='rgba(42,63,95,0.8)'),
                    align = 'center',
                    font = dict(
                        color = '#ffffff',
                        size = 12
                    )),
        cells=dict(values=np.array(results).T,
                   line = dict(width=0),
                   fill = dict(color=[['#EBF0F8', '#ffffff', '#EBF0F8','#ffffff','#EBF0F8','#ffffff']]),
                   align = 'center', 
                   height = 40),
        columnwidth=[0.3, 0.25, 0.3])
    layout = dict(
        height=300,
        margin=dict(
            l=5,
            r=5,
            t=30,
            b=0
        )
    )
    data = [trace]
    andar_table = dict(data=data, layout=layout)
    
    return iplot(andar_table)

def validation(data,n_val):
    """
    This function returns a dataframe for a visual and empirical validation.
    
    Parameters
    ----------
    data : Dataframe
        Dataframe to validate
        
    n_val: numeric
        Defines the number of validation
        
    Return
    ----------
    Trial : Dataframe 

    """
        
    Trial = data.copy()
        
    if n_val==1:
        return Trial[6]   
    elif n_val==2:        
        return Trial[15]    
    elif n_val==3:        
        return Trial[27]  
    elif n_val==4:
        return Trial[18]
    else:        
        return Trial[34]
        
def empiric_trade(val):
    """
    This function creates an empirical strategy defining a direction, takeprofit, stoploss and volume. It also 
    indicates the strategy recomended after the visualization of each validation.
    
    Parameters
    ----------
    val : Dataframe
        Dataframe to validate
        
    Return
    ----------
    summary : Dataframe
        This dataframe contains a summary of the empirical strategy

    """
    np.random.seed(123)
    
    ind_date = val.iloc[0]['timestamp']+datetime.timedelta(minutes=30)
    ind_idx  = val.index[val['timestamp']==ind_date] #quitar un corchete
    ind_idx = ind_idx[0]
    rent =  val.iloc[-1]['close']-val.loc[ind_idx]['open']
    max_pip = np.round((-val.loc[ind_idx]['open']+val['open'].max())*10000)
    min_pip = np.round((val.loc[ind_idx]['open']-val['open'].min())*10000)
    
    
    if rent > 0: #Buy
        direction = np.sign(rent)
        operation = 'Buy'
        volume = np.median(val['volume'])
        takeprofit = + np.random.randint(0,max_pip)
        stoploss = np.random.randint(0,min_pip)
        price_tp = val.loc[ind_idx]['open']+(takeprofit/10000)
        price_sl = val.loc[ind_idx]['open']+(-stoploss/10000)
        profit = volume * (price_tp-val.loc[ind_idx]['open'])
        loss = volume * (price_sl-val.loc[ind_idx]['open'])
        
    elif rent < 0: #Sell
        direction = np.sign(rent)
        operation = 'Sell'
        volume = np.median(val['volume'])
        takeprofit = np.random.randint(0,min_pip)
        stoploss =  np.random.randint(0,max_pip)
        price_tp = val.loc[ind_idx]['open']+(-takeprofit/10000)
        price_sl = val.loc[ind_idx]['open']+(stoploss/10000)
        profit = volume * (val.loc[ind_idx]['open']-price_tp)
        loss = volume * (val.loc[ind_idx]['open']-price_sl)
        
    dict_metrics = {'Operation':operation,'Direction':direction, 'Volume':volume, 'Takeprofit(pip)':takeprofit, 
                    'Stoploss(pip)': stoploss,'Profit($)':profit, 'Loss($)':loss}
        
    summary = pd.DataFrame(dict_metrics, index=['Operation']) 
    return summary


def decisions(df_escenarios, usdmxn):
    """
    Function that creates a dataframe with the desingned
    strategy to place orders according to the scenario.

    Parameters
    ----------

    df_escenarios:dataframe

        'Datetime': timestamp, date of the indicator
        'Scenario': A, B, C or D
        'Direction': -1 if close price < open, 1 if close price > open
        'Bullish_Pip': diference between the highest price (t_0:t_30) and the open price t_0
        'Bear_Pip': diference between the open price t_0 and the lowest price (t_0:t_30)
        'Volatility': diference between the highest price and the lowest

     usdmxn:dataframe of the prices of the currency

    Returns
    -------
        df_de: dataframe
            dataframe with the following information
            'Scenario': A, B, C or D
            'Operation': Sell or Buy
            'SL': stop loss
            'TP': take profit
            'Volume': optimal volume


    """
    data_des = df_escenarios.copy()
    data_des = data_des.iloc[0:13, :]
    scen_a = data_des[data_des['Scenario'] == 'A']
    scen_b = data_des[data_des['Scenario'] == 'B']
    scen_c = data_des[data_des['Scenario'] == 'C']
    scen_d = data_des[data_des['Scenario'] == 'D']

    inx_scen = scen_a.index[scen_a['Scenario'] == 'A']
    volume_a = []
    for i in range(len(scen_a)):
        volume_a.append((usdmxn[usdmxn['timestamp'] == inx_scen[i]]['volume']).tolist())
    scen_a['Volume'] = [volume_a[i][0] for i in range(len(scen_a))]

    tp_a = scen_a['Bear_Pip'].mean()
    sl_a = scen_a['Bullish_Pip'].mean()
    vol_a = scen_a['Volume'].min()

    tp_b = scen_b['Bear_Pip'].mean()
    sl_b = scen_b['Bullish_Pip'].mean()
    vol_b = 47170

    inx_scen_c = scen_c.index[scen_c['Scenario'] == 'C']
    volume_c = []
    for i in range(len(scen_c)):
        volume_c.append((usdmxn[usdmxn['timestamp'] == inx_scen_c[i]]['volume']).tolist())
    scen_c['Volume'] = [volume_c[i][0] for i in range(len(scen_c))]

    tp_c = scen_c['Bullish_Pip'].mean()
    sl_c = scen_a['Bear_Pip'].mean()
    vol_c = scen_c['Volume'].min()

    inx_scen_d = scen_d.index[scen_d['Scenario'] == 'D']
    volume_d = []
    for i in range(len(scen_d)):
        volume_d.append((usdmxn[usdmxn['timestamp'] == inx_scen_d[i]]['volume']).tolist())
    scen_d['Volume'] = [volume_d[i][0] for i in range(len(scen_d))]

    tp_d = scen_b['Bullish_Pip'].mean()
    sl_d = scen_d['Bear_Pip'].mean()
    vol_d = scen_d['Volume'].min()

    dict_d = {'Scenario': ['A', 'B', 'C', 'D'], 'Operation': ['Sell', 'Sell', 'Buy', 'Buy'],
              'SL':np.round([sl_a, sl_b, sl_c, sl_d], 0),
              'TP':np.round([tp_a, tp_b, tp_c, tp_d], 0), 'Volume':[vol_a, vol_b, vol_c, vol_d]}

    df_de = pd.DataFrame(dict_d)

    return df_de


# ================================== Trading System Definition ============================================ #

### Trading system definition
def get_trading_summary(data: pd.DataFrame,
                        clasification: pd.DataFrame,
                        pip_up: float,
                        pip_down: float,
                        volume: float,
                        intial_cap: float,
                        scenario=None) -> pd.DataFrame:
    """
    Trading system definition based on the unemployment rate from the USA economy.
    The decisions are set from a previous knowledge on the behaviour and relationship
    between the USDMXN price and the reported value for the indicator. It summarize the
    trading operation and the capital evolution through the life of the operations

    Parameters
    ----------

    data: pd.DataFrame (default:None) --> Required parameter

        USDMXN prices on a minute granularity, it has to follow the next structure

        'timestamp': First column, correspond to the timestamp associated to each price
        'open': Second column, correspond to the open price for each timestamp associated
        'high': Third column, correspond to the high price for each timestamp associated
        'low': Fourth column, correspond to the low price for each timestamp associated
        'close': Fifth column, correspond to the close price for each timestamp associated
        'volume': Sixth column, correspond to the volume operated for each timestamp associated
        'date': Seventh column, correspond to the date in YYYY-MM-DD format for each timestamp

    clasification: pd.DataFrame (default:None) --> Required parameter

        USA unemployment rate reporting from Jan-2018 to Dec-2020 (monthly frequency)

        'Datetime': DataFrame index, correspond to the timestamp where the indicator was reported
        'Country/Region ': Region of origin (unique value "United States")
        'Event ': Indicator name
        'Importance ': Level of importance associated to the indicator within the USA economy
        'Period ': Reported period
        'Actual ': Reported value for the indicator
        'Consensus ': Economical expectations to the indicator report value
        'Prior ': Previous value corresponding to the indicator (previous month)
        'Scenario': Type of scenario definition

    pip_up: float (default:None) --> Required parameter

        Number of Pip's that will define an increase on USDMXN prices

    pip_down: float (default:None) --> Required parameter

        Number of Pip's that will define a downgrade on USDMXN prices

    volume: float (default:None) --> Required parameter

        Number of USDMXN to be trade by each operation

    intial_cap: float (default:None) --> Required parameter

        Initial capital for start the trading system (in USD)

    scenario: str (default:None) --> Optional parameter

        Scenario where the trading system wants to be analyzed. If none it will display all scenarios

    Returns
    -------

    trading_res: pd.DataFrame

        Final summary associated to the trading strategy it can correspond just to a single scenario or
        all of them contained in clasification data frame. It follows the next structure

        'Datetime': Index, timestamp where the indicator was reported
        'Scenario': Scenario associated to the trading decision within that timestamp
        'Operation': Signal detection (buy or sell)
        'Volume': Sell or buy volume associated to the trading decision
        'Result': Balance of the operation (won or lost)
        'Pip Up': The upper pip defined for that trading strategy
        'Pip Down': The lower pip defined for that trading strategy
        'Capital': Utility assigned to the operation
        'Cumulative Capital': Evolution of the invested capital within the whole period

    References
    ----------

    [1] https://pandas.pydata.org/docs/

    """

    # Data structure to return
    trading_res = pd.DataFrame(columns=['Scenario', 'Operation', 'Volume',
                                        'Result', 'Pip Up', 'Pip Down',
                                        'Capital', 'Cumulative Capital'])

    # First define it for all possible scenarios
    if scenario is None:

        scenario_list = []
        utility_list = []
        cum_capital = []
        status_operation = []

        for i, j in zip(clasification.index.values, range(0, len(clasification))):

            data_adder = lambda each_list, values: each_list.append(values)

            try:
                sub_set = data[(data['timestamp'] >= pd.to_datetime(i)) &
                               (data['timestamp'] < pd.to_datetime(clasification.index.values[j + 1]))]

            except IndexError:
                sub_set = data[(data['timestamp'] >= pd.to_datetime(i))]

            cases = clasification['Scenario'].iloc[j]

            # Sell definition
            if cases == 'A' or cases == 'B':

                data_adder(scenario_list, cases)  # Add scenario

                # Limit definition for this scenarios
                operation_price = sub_set['open'].iloc[1]
                take_profit = (sub_set['open'].iloc[1] - (pip_down / 10000))
                take_profit = (volume * (operation_price - take_profit)) / operation_price  # Take profit in USD
                stop_loss = (sub_set['open'].iloc[1] + (pip_up / 10000))
                stop_loss = (volume * (operation_price - stop_loss)) / operation_price  # Stop loss in USD

                # Stop loss boundary at a lost of 1000 USD
                if stop_loss <= -1000:
                    stop_loss = -1000

                else:
                    stop_loss = stop_loss

                sub_utility = []
                utility_evo = []

                for k in range(2, len(sub_set)):

                    check_price = sub_set.iloc[k].close
                    check_utility = ((operation_price - check_price) * volume) / operation_price  # Utility in USD
                    data_adder(utility_evo, check_utility)

                    if (check_utility >= take_profit) or (check_utility <= stop_loss):  # Close position

                        utility = (volume * (operation_price - check_price)) / operation_price
                        data_adder(utility_list, utility)

                        if utility < 0:
                            data_adder(status_operation, 'Lost')

                        elif utility >= 0:
                            data_adder(status_operation, 'Won')

                        capital = intial_cap + sum(utility_list)
                        data_adder(cum_capital, capital)

                        break

                    else:  # Open position

                        utility = (volume * (operation_price - check_price)) / operation_price
                        data_adder(sub_utility, utility)

                # Never touch limit of take profit or stop loss
                if len(utility_list) < j + 1:

                    data_adder(utility_list, sub_utility.pop())
                    capital = intial_cap + sum(utility_list)
                    data_adder(cum_capital, capital)

                    if sub_utility.pop() < 0:
                        data_adder(status_operation, 'Lost')

                    elif sub_utility.pop() >= 0:
                        data_adder(status_operation, 'Won')

            # Buy definition
            elif cases == 'C' or cases == 'D':

                data_adder(scenario_list, cases)  # Add scenario

                # Limit definition for this scenarios
                operation_price = sub_set.iloc[1].open
                take_profit = (sub_set['open'].iloc[1] + (pip_up / 10000))
                take_profit = (volume * (take_profit - operation_price)) / operation_price  # Take profit in USD
                stop_loss = (sub_set['open'].iloc[1] - (pip_down / 10000))
                stop_loss = (volume * (stop_loss - operation_price)) / operation_price  # Take profit in USD

                # Stop loss boundary at a lost of 1000 USD
                if stop_loss <= -1000:
                    stop_loss = -1000

                else:
                    stop_loss = stop_loss

                sub_utility = []
                utility_evo = []

                for k in range(2, len(sub_set)):

                    check_price = sub_set.iloc[k].close
                    check_utility = (volume * (check_price - operation_price)) / operation_price  # Utility in USD
                    data_adder(utility_evo, check_utility)

                    if (check_utility >= take_profit) or (check_utility <= stop_loss):  # Close position

                        utility = (volume * (check_price - operation_price)) / operation_price
                        data_adder(utility_list, utility)

                        if utility < 0:
                            data_adder(status_operation, 'Lost')

                        elif utility >= 0:
                            data_adder(status_operation, 'Won')

                        capital = intial_cap + sum(utility_list)
                        data_adder(cum_capital, capital)

                        break

                    else:  # Open position

                        utility = (volume * (check_price - operation_price)) / operation_price
                        data_adder(sub_utility, utility)

                # Never touch limit of take profit or stop loss
                if len(utility_list) < j + 1:

                    data_adder(utility_list, sub_utility.pop())
                    capital = intial_cap + sum(utility_list)
                    data_adder(cum_capital, capital)

                    if sub_utility.pop() < 0:
                        data_adder(status_operation, 'Lost')

                    elif sub_utility.pop() >= 0:
                        data_adder(status_operation, 'Won')

        # Final dataframe filling
        trading_res['Scenario'] = scenario_list
        trading_res['Operation'] = ['Sell' if i == 'A' or i == 'B' else 'Buy'
                                    for i in clasification['Scenario']]
        trading_res['Volume'] = [volume] * len(clasification)
        trading_res['Result'] = status_operation
        trading_res['Pip Up'] = [pip_up] * len(clasification)
        trading_res['Pip Down'] = [pip_down] * len(clasification)
        trading_res['Capital'] = utility_list
        trading_res['Cumulative Capital'] = cum_capital
        trading_res.index = clasification.index.values
        trading_res.index.name = 'Datetime'

        trading_res = trading_res[trading_res['Capital'] >= -1000]

        return trading_res

    # For the specified scenario to analyze
    else:

        scenario_list = []
        utility_list = []
        cum_capital = []
        status_operation = []

        clasification = clasification[clasification['Scenario'] == scenario]

        for i, j in zip(clasification.index.values, range(0, len(clasification))):

            data_adder = lambda each_list, values: each_list.append(values)

            try:
                sub_set = data[(data['timestamp'] >= pd.to_datetime(i)) &
                               (data['timestamp'] < pd.to_datetime(clasification.index.values[j + 1]))]

            except IndexError:
                sub_set = data[(data['timestamp'] >= pd.to_datetime(i))]

            # Sell definition
            if scenario == 'A' or scenario == 'B':

                data_adder(scenario_list, scenario)  # Add scenario

                # Limit definition for this scenarios
                operation_price = sub_set['open'].iloc[1]
                take_profit = (sub_set['open'].iloc[1] - (pip_down / 10000))
                take_profit = (volume * (operation_price - take_profit)) / operation_price  # Take profit in USD
                stop_loss = (sub_set['open'].iloc[1] + (pip_up / 10000))
                stop_loss = (volume * (operation_price - stop_loss)) / operation_price  # Stop loss in USD

                # Stop loss boundary at a lost of 1000 USD
                if stop_loss <= -1000:
                    stop_loss = -1000

                else:
                    stop_loss = stop_loss

                sub_utility = []
                utility_evo = []

                for k in range(2, len(sub_set)):

                    check_price = sub_set.iloc[k].close
                    check_utility = ((operation_price - check_price) * volume) / operation_price  # Utility in USD
                    data_adder(utility_evo, check_utility)

                    if (check_utility >= take_profit) or (check_utility <= stop_loss):  # Close position

                        utility = (volume * (operation_price - check_price)) / operation_price
                        data_adder(utility_list, utility)

                        if utility < 0:
                            data_adder(status_operation, 'Lost')

                        elif utility >= 0:
                            data_adder(status_operation, 'Won')

                        capital = intial_cap + sum(utility_list)
                        data_adder(cum_capital, capital)

                        break

                    else:  # Open position

                        utility = (volume * (operation_price - check_price)) / operation_price
                        data_adder(sub_utility, utility)

                # Never touch limit of take profit or stop loss
                if len(utility_list) < j + 1:

                    data_adder(utility_list, sub_utility.pop())
                    capital = intial_cap + sum(utility_list)
                    data_adder(cum_capital, capital)

                    if sub_utility.pop() < 0:
                        data_adder(status_operation, 'Lost')

                    elif sub_utility.pop() >= 0:
                        data_adder(status_operation, 'Won')

            # Buy definition
            elif scenario == 'C' or scenario == 'D':

                data_adder(scenario_list, scenario)  # Add scenario

                # Limit definition for this scenarios
                operation_price = sub_set.iloc[1].open
                take_profit = (sub_set['open'].iloc[1] + (pip_up / 10000))
                take_profit = (volume * (take_profit - operation_price)) / operation_price  # Take profit in USD
                stop_loss = (sub_set['open'].iloc[1] - (pip_down / 10000))
                stop_loss = (volume * (stop_loss - operation_price)) / operation_price  # Take profit in USD

                # Stop loss boundary at a lost of 1000 USD
                if stop_loss <= -1000:
                    stop_loss = -1000

                else:
                    stop_loss = stop_loss

                sub_utility = []
                utility_evo = []

                for k in range(2, len(sub_set)):

                    check_price = sub_set.iloc[k].close
                    check_utility = (volume * (check_price - operation_price)) / operation_price  # Utility in USD
                    data_adder(utility_evo, check_utility)

                    if (check_utility >= take_profit) or (check_utility <= stop_loss):  # Close position

                        utility = (volume * (check_price - operation_price)) / operation_price
                        data_adder(utility_list, utility)

                        if utility < 0:
                            data_adder(status_operation, 'Lost')

                        elif utility >= 0:
                            data_adder(status_operation, 'Won')

                        capital = intial_cap + sum(utility_list)
                        data_adder(cum_capital, capital)

                        break

                    else:  # Open position

                        utility = (volume * (check_price - operation_price)) / operation_price
                        data_adder(sub_utility, utility)

                # Never touch limit of take profit or stop loss
                if len(utility_list) < j + 1:

                    data_adder(utility_list, sub_utility.pop())
                    capital = intial_cap + sum(utility_list)
                    data_adder(cum_capital, capital)

                    if sub_utility.pop() < 0:
                        data_adder(status_operation, 'Lost')

                    elif sub_utility.pop() >= 0:
                        data_adder(status_operation, 'Won')

        # Final dataframe filling
        trading_res['Scenario'] = scenario_list
        trading_res['Operation'] = ['Sell' if i == 'A' or i == 'B' else 'Buy'
                                    for i in clasification['Scenario']]
        trading_res['Volume'] = [volume] * len(clasification)
        trading_res['Result'] = status_operation
        trading_res['Pip Up'] = [pip_up] * len(clasification)
        trading_res['Pip Down'] = [pip_down] * len(clasification)
        trading_res['Capital'] = utility_list
        trading_res['Cumulative Capital'] = cum_capital
        trading_res.index = clasification.index.values
        trading_res.index.name = 'Datetime'

        trading_res = trading_res[trading_res['Capital'] >= -1000]

        return trading_res

# ================================ Objective Function Definition ========================================== #

### Sharpe ratio as an objective function
def max_sharpe(iterable, *args):
    """
    Sharpe ratio metric definition for evaluating the final performance within an investment strategy.
    The final aim of this function it's to get maximized in order to find an optimal returns-risk balance

    Parameters
    ----------

    iterable
        Variable parameters to iterate in optimization process. (pip_up, pip_down, volume)

    args
        Fixed parameters to calculate the trading system definition

    Returns
    -------

    res
        Sharpe ratio calculations, it returns a negative result in order to minimize that value

    References
    ----------

    [1] https://cfastudyguide.com/performance-appraisal/

    """

    pip_up, pip_down, volume = iterable
    data, clasification, intial_cap, scenario, rf = args

    # Risk free definition
    rf = rf / 12  # Monthly

    # Expected returns definition
    capital = get_trading_summary(data, clasification, pip_up, pip_down, volume, intial_cap, scenario)[
        'Cumulative Capital']
    expected_ret = (capital / intial_cap) - 1
    rf = rf * np.sqrt(len(expected_ret) / 12)  # Risk free rate associated to the duration

    # Sharpe ratio definition
    sharpe_ratio = ((np.mean(expected_ret) * len(expected_ret)) - rf) / np.std(expected_ret)
    res = -sharpe_ratio

    return res

# ================================= Optimization PSO definition =========================================== #

### Particle Swarm Optimization
def get_pso(obj_fun,
            upper_limit:list,
            down_limit:list,
            args:tuple,
            iterations:float,
            minstep:float):

    """
    PSO optimization function definition for the trading system

    Parameters
    ----------

    obj_fun
        Objective function definition to be optimizated

    upper_limit (default:None) --> Required parameter
        Upper limit for all the iterable variables in the optimization problem. It has to follow
        a list type

    down_limit (default:None) --> Required parameter
        Down values limit for all the iterable variables in the optimization problem. It has to follow
        a list type

    args (default:None) --> Required parameter
        All the fixed paramaters to be using on the optimization problem

    iterations (default:None) --> Required parameter
        Number of iterations to be explored by the pso algorithm

    minstep (default:None) --> Required parameter
        Step size to be used between the upper and down limit

    Returns
    -------

    xopt
        Optimal value for all the iterable components in the problem definition

    fopt
        Optimal value for the global problem (value that it's wanted to be the best)

    References
    ----------

    [1] https://pyswarms.readthedocs.io/en/latest/
    """

    # Define the optimal process
    xopt, fopt = pso(obj_fun, upper_limit, down_limit, args=args, maxiter=iterations, minstep=minstep)

    return xopt, fopt
