import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import os
import os.path
import sys
cwd = os.getcwd()
cwd
def nnmodel():
    
    # COVID time window
    COVID = pd.DataFrame({
        'holiday': 'COVID',
        'ds': pd.to_datetime(['2020-03-15']),
        'lower_window': -15,
        'upper_window': 15,    
        })
    events = COVID
    model = Prophet(holidays = events)
    stock_data= pd.read_csv(os.path.join(cwd, 'data\\interim\\stock_data.csv'))
    train_data = pd.read_csv(os.path.join(cwd, 'data\\interim\\train_data.csv'))
    model.fit(stock_data)
    future = model.make_future_dataframe(periods=92,include_history=True)
    forecast = model.predict(future)
    
    df_cv = cross_validation(model, initial='1457 days', period='7 days', horizon = '90 days')
    df_p = performance_metrics(df_cv)
    # Cross validation
    df_cv_  = df_cv[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_ = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_ = forecast[900:len(train_data)]
    frames = [forecast_, df_cv_]
    result = pd.concat(frames,ignore_index=True)
    
    fig = plot_cross_validation_metric(df_cv, metric='mape', rolling_window = .1)
    fig.savefig('reports\\figures\\forecast_performance', bbox_inches='tight')
    plot = model.plot(result)

    plt.xlim(pd.to_datetime(['2020-10-01', '2021-3-31']))
    plt.ylim(200,250)
    plt.show()
    plt.savefig('reports\\figures\\forecast_vs_actual_zoom', bbox_inches='tight')
    return(result)