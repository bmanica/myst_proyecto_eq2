
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
from plotly.offline import iplot


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

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#                           Statistical functions
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

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
        header=dict(values=['<b>Statistical test</b>', '<b>Significance Level</b>', '<b>P-Value</b>',
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
    The histogram of the time series.
    A chart with the normalility test results.
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
        header=dict(values=['<b>Statistical test</b>', '<b>Significance Level</b>', '<b>P-Value</b>',
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
    
    fig = go.Figure(data=[x])

    fig.update_layout(
       title_text='Unemployment Rate Histogram')
    fig.show()
   

    


    
   






    return iplot(andar_table)




def boxplot(x,title:str):
    """
    Docstring
    Function that plots the time series boxplot.
    
    Parameters
    --------------------
    x: time series.
    title: the title of the plot.
    
    Returns
    --------------------
    boxplot of the time series. 
    """
    fig = go.Figure(data=[go.Box(x=x,)])

    fig.update_layout(
        title_text=title)

    return fig.show()



def seasonality(x,alfa,m):
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
            header=dict(values=['<b>Statistical test</b>', '<b>Significance Level</b>', '<b>P-Value</b>',
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
    """


    q1 = x['Actual '].quantile(0.25)
    q3 = x['Actual '].quantile(0.75)
    iqr = q3-q1
    lw_limit = q1 -1.5*iqr
    upper_limit = q3+1.5*iqr

    outliers = x[x['Actual ']<lw_limit].append(x[x['Actual ']>upper_limit])
    return outliers


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

