import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# métricas
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# modelos
from prophet import Prophet
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

def predict_prophet(train, test):
    # Prophet
    train_prophet = train.copy()
    train_prophet = train_prophet.reset_index()
    train_prophet = train_prophet.rename(columns={'data': 'ds', 'fechamento': 'y'})

    test_prophet = test.copy()
    test_prophet = test_prophet.reset_index()
    test_prophet = test_prophet.rename(columns={'data': 'ds', 'fechamento': 'y'})

    model_prophet = Prophet(weekly_seasonality=False,
    yearly_seasonality=True,
    daily_seasonality=False)

    model_prophet.add_country_holidays(country_name='BR')
    model_prophet.add_regressor('abertura')

    model_prophet.fit(train_prophet)

    future_prophet= model_prophet.make_future_dataframe(periods=test_size, freq='B') # freq='B' (dias úteis - business days)
    future_prophet['abertura'] = pd.concat([train_prophet['abertura'], test_prophet['abertura'], test_prophet['abertura']], ignore_index=True)

    predict_prophet = model_prophet.predict(future_prophet)
    predict_prophet.sort_values(by='ds')
    predict_prophet_test = predict_prophet.tail(test_size)[['ds', 'yhat']].reset_index(drop=True)
    return test_prophet, predict_prophet_test

def predict_xgboost(train, test):
    train_xgb = train.copy()
    train_xgb = train_xgb.reset_index()
    train_xgb['ano'] = train_xgb['data'].dt.year
    train_xgb['mes'] = train_xgb['data'].dt.month
    train_xgb['dia'] = train_xgb['data'].dt.day
    train_xgb['diadasemana'] = train_xgb['data'].dt.dayofweek


    test_xgb = test.copy()
    test_xgb = test_xgb.reset_index()
    test_xgb['ano'] = test_xgb['data'].dt.year
    test_xgb['mes'] = test_xgb['data'].dt.month
    test_xgb['dia'] = test_xgb['data'].dt.day
    test_xgb['diadasemana'] = test_xgb['data'].dt.dayofweek
    
    FEATURES = ['ano', 'mes', 'dia', 'diadasemana', 'abertura']
    TARGET = 'fechamento'

    X_train_xgb, y_train_xgb = train_xgb[FEATURES], train_xgb[TARGET]
    X_test_xgb, y_test_xgb = test_xgb[FEATURES], test_xgb[TARGET]

    reg = xgb.XGBRegressor(objective='reg:squarederror')
    reg.fit(X_train_xgb, y_train_xgb)

    predict_xgb = reg.predict(X_test_xgb)

    return X_test_xgb, y_test_xgb, predict_xgb

def predict_sarimax(train, test):
    train_sarimax = train[['fechamento']].copy()
    exog_train_sarimax = train[['abertura']].copy()
    test_sarimax = test[['fechamento']].copy()
    exog_test_sarimax = test[['abertura']].copy()

    model_sarimax = SARIMAX(train_sarimax, exog=exog_train_sarimax, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fit_sarimax = model_sarimax.fit()

    predict_sarimax = fit_sarimax.get_forecast(steps=len(test_sarimax), exog=exog_test_sarimax).predicted_mean
    return test_sarimax, predict_sarimax

def plot_testpredict(model, x_test, y_test, x_predict, y_predict):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(x_test, y_test, label='Dados de Teste')
    plt.plot(x_predict, y_predict, label=f'Previsões {model}')

    plt.title(f'Previsão do Valor de Fechamento do Índice da Bolsa - {model}')
    plt.xlabel('Data')
    plt.ylabel('Valor de Fechamento')
    plt.legend(loc='best')
    return fig

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, mse, mape

df = pd.read_csv('dados_ibovespa.csv', sep=',', thousands='.', parse_dates=[0], date_format='%d.%m.%Y')
column_names = ['data', 'fechamento','abertura', 'maxima', 'minima', 'volume', 'variacao']
df.columns = column_names
df['volume'] = df['volume'].str.replace('B', 'e9').str.replace('M', 'e6').str.replace('K','e3').str.replace(',', '.')
df['volume'] = pd.to_numeric(df['volume'])
df['variacao'] = df['variacao'].str.replace('%', '').str.replace(',', '.').astype(float)
df = df.sort_values('data').reset_index(drop=True)
df = df.set_index('data')

min_date = df.index[0].to_pydatetime()
max_date = df.index[-1].to_pydatetime()

default_test_size = 30

# Configuração da página do Streamlit
st.set_page_config(page_title='Modelo preditivo do fechamento da IBOVESPA', page_icon=':chart_with_upwards_trend:', layout="wide")

# Cabeçalho
st.write("# Pós Tech - Data Analytics - 5DTAT")
st.write("## Fase 2 - Machine Learning and Time Series")
st.write('##### Tech Challenge - Modelo preditivo do fechamento da IBOVESPA')
st.write('''###### Grupo 46

Integrantes: 

* Alexandre Aquiles Sipriano da Silva (alexandre.aquiles@alura.com.br)
* Gabriel Machado Costa (gabrielmachado2211@gmail.com)
* Caio Martins Borges (caio.borges@bb.com.br)
        
Código disponível em: https://github.com/alexandreaquiles/postech-fiap-dtat-tech-challenge-fase2
''')

st.divider()

st.sidebar.title('Filtros')

start_date = st.sidebar.slider('Data Inicial', min_date, max_date, pd.to_datetime('2021-07-22').to_pydatetime())
st.sidebar.caption(f'Data a partir da qual queremos as previsões. Dados disponíveis de {min_date.strftime("%d/%m/%Y")} a {max_date.strftime("%d/%m/%Y")}.')

if start_date:
    df = df[df.index >= start_date]

test_size = st.sidebar.number_input('Tamanho do teste', min_value=1, max_value=180, value=default_test_size)
st.sidebar.caption(f'Tamanho do dataset separado para teste dos modelos. Por padrão, será considerado {default_test_size}.')

# criando dfs de treino e teste
df_modeling = df[['fechamento', 'abertura']]
train_size = df_modeling.shape[0] - test_size
train = df_modeling[:train_size]
test = df_modeling[train_size:]

aba1, aba2 = st.tabs(['Comparação de Modelos', 'Dados brutos'])

with aba1:

    st.header('Comparação dos modelos')

    # Executando modelos
    test_prophet, predict_prophet = predict_prophet(train, test)
    X_test_xgb, y_test_xgb, predict_xgb = predict_xgboost(train, test)
    test_sarimax, predict_sarimax = predict_sarimax(train, test)

    # Métricas
    metrics_prophet = calculate_metrics(test_prophet['y'].values, predict_prophet['yhat'].values)
    metrics_xgb = calculate_metrics(y_test_xgb, predict_xgb)
    metrics_sarimax = calculate_metrics(test_sarimax['fechamento'].values, predict_sarimax)
    df_metrics = pd.DataFrame(
        [metrics_prophet, metrics_xgb, metrics_sarimax],
        columns=['MAE', 'MSE', 'MAPE'],
        index=['Prophet', 'XGBoost', 'SARIMAX'],
    )
    st.dataframe(df_metrics, use_container_width=True, column_config={
        'MSE': st.column_config.NumberColumn(format="%d"),
    })

    st.header('Gráficos')
 
    st.subheader('Prophet')
    fig_prophet = plot_testpredict('Prophet', test_prophet['ds'], test_prophet['y'], predict_prophet['ds'], predict_prophet['yhat'])
    st.pyplot(fig_prophet)

    st.subheader('XGBoost')
    fig_xbg = plot_testpredict('XGBoost', X_test_xgb.index, y_test_xgb, X_test_xgb.index, predict_xgb)
    st.pyplot(fig_xbg)

    st.subheader('SARIMAX')
    fig_sarimax = plot_testpredict('SARIMAX', test_sarimax.index, test_sarimax['fechamento'], test_sarimax.index, predict_sarimax)
    st.pyplot(fig_sarimax)

with aba2:
    st.write('Dados históricos dos últimos 20 anos da IBOVESPA, considerando o período de 22/07/2004 até 22/07/2024 obtidos a partir do site Investing.com.')
    st.write('Fonte: https://br.investing.com/indices/bovespa-historical-data')
    st.dataframe(df, use_container_width=True, column_config={
        '_index': st.column_config.DatetimeColumn(format='DD/MM/YYYY'),
        'fechamento': st.column_config.NumberColumn(format="%d"),
        'abertura': st.column_config.NumberColumn(format="%d"),
        'maxima': st.column_config.NumberColumn(format="%d"),
        'minima': st.column_config.NumberColumn(format="%d"),
        'volume': st.column_config.NumberColumn(format="%d"),
    })