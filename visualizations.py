
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: FUNDAMENTAL ANALISIS                                                          -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: lyhaesp, bmanica, JuanPablo2019, ACristinaVV.                                               -- #
# -- license: GNU General Public License v3.0                                                            -- #
# -- repository: https://github.com/bmanica/myst_proyecto_eq2                                            -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import plotly.graph_objects as go
import datetime

def val_graph(df):
    """
    This function plots the closing price time series. It also includes a line where the indicator was announced 
    
    Parameters
    ----------
    df : Dataframe
        Dataframe to validate
        
    Return
    ----------
    fig : Plot 

    """
    
    fig = go.Figure([go.Scatter(x=df['timestamp'], y=df['close'])])
    fig.update_layout(title_text="Visualization of the closing price for the selected scenario",
                     xaxis_title="Timestamp(Minutes)",yaxis_title="Price(USD)")
    
    
    fig.add_vline(x= df.loc[0]['timestamp'] + datetime.timedelta(minutes=30), line_width=1, line_dash="dash", line_color="red",
                 name='Indicator release')
    
                  
    return fig.show()


def hist(x,title:str):
    """
    Docstring
    Function that plots the time series Histogram.
    
    Parameters
    --------------------
    x: time series.
    title: the title of the plot.
    
    Returns
    --------------------
    boxplot of the time series. 
    """
    fig = go.Figure(data=[go.Histogram(x=x,)])

    fig.update_layout(
        title_text=title)

    return fig.show()





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