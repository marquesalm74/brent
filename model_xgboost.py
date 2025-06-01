import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import os
import joblib
from connect import conn, encerra_conn
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold, cross_validate, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ============================
# Funções auxiliares
# ============================

def criar_caracteristicas(brent):
    brent = brent.sort_values('data').reset_index(drop=True)

    # Criar colunas lag usando shift, para usar valores anteriores
    for lag in [1, 2, 3, 7, 14, 30]:
        brent[f'lag_{lag}'] = brent['brent_price'].shift(lag)

    # Converter colunas lag para numérico (precaução)
    lags = [f'lag_{lag}' for lag in [1, 2, 3, 7, 14, 30]]
    for lag in lags:
        brent[lag] = pd.to_numeric(brent[lag], errors='coerce')

    # Preencher valores ausentes nos lags (NaNs das primeiras linhas)
    brent[lags] = brent[lags].fillna(method='bfill')  # ou fillna(0) se preferir

    # Remover linhas com NaN em brent_price ou lags (se houver)
    brent = brent.dropna(subset=lags + ['brent_price']).reset_index(drop=True)

    return brent

def treinar_modelo_xgboost(brent):
    lags = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_30']
    for lag in lags:
        brent[lag] = pd.to_numeric(brent[lag], errors='coerce')
    brent[lags] = brent[lags].fillna(method='bfill')

    features = lags
    target = 'brent_price'

    brent = brent.dropna(subset=features + [target])

    X = brent[features]
    y = brent[target]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    import xgboost as xgb
    modelo = xgb.XGBRegressor()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    return y_test, y_pred, modelo, X_test, y_test

def avaliar_modelo(y_teste, y_pred):
    mse = mean_squared_error(y_teste, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_teste, y_pred)
    r2 = r2_score(y_teste, y_pred)

    st.subheader('Avaliação do Modelo XGBoost')
    st.write(f'MSE: {mse:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.2f}')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_teste, y=y_pred, mode='markers', name='Previsões'))
    fig.add_trace(go.Scatter(x=[min(y_teste), max(y_teste)], y=[min(y_teste), max(y_teste)], mode='lines', name='Ideal', line=dict(color='red')))
    fig.update_layout(title='Valores Reais vs. Previstos', xaxis_title='Valor Real', yaxis_title='Valor Previsto')
    st.plotly_chart(fig, use_container_width=True)
#########################################
# codigo novo previsão
def prever_futuro_xgboost1(brent, dias_previstos):

    try:
        modelo = joblib.load('modelos/xgboost_model.joblib')
    except FileNotFoundError:
        st.error("Modelo XGBoost não encontrado. Por favor, treine o modelo primeiro.")
        return

    # ✅ Conversão da coluna 'data' para datetime
    brent['data'] = pd.to_datetime(brent['data'])

    # ✅ Ordenar e resetar índice
    brent = brent.sort_values('data').reset_index(drop=True)

    ult_data = brent['data'].iloc[-1]
    ult_precos = brent['brent_price'].tolist()

    previsoes = []

    for i in range(dias_previstos):
        lags = []
        for lag in [1, 2, 3, 7, 14, 30]:
            idx = len(ult_precos) - lag
            if idx >= 0:
                lags.append(ult_precos[idx])
            else:
                lags.append(ult_precos[0])  # fallback

        X_pred = pd.DataFrame([lags], columns=[f'lag_{l}' for l in [1, 2, 3, 7, 14, 30]])
        X_pred = X_pred.astype(float)

        pred = modelo.predict(X_pred)[0]

        ult_precos.append(pred)
        previsoes.append(pred)

    # ✅ Gerar datas futuras corretamente a partir da última data do dataset
    datas_futuras = [ult_data + pd.Timedelta(days=i + 1) for i in range(dias_previstos)]

    if len(datas_futuras) != len(previsoes):
        st.error(f"Tamanho incompatível: datas_futuras={len(datas_futuras)}, previsoes={len(previsoes)}")
        return

    df_previsao = pd.DataFrame({'data': datas_futuras, 'previsao': previsoes})

    st.subheader(f'Previsão para os próximos {dias_previstos} dias')
    st.dataframe(df_previsao.set_index('data'))

    fig = px.line(df_previsao, x='data', y='previsao', title='Previsão XGBoost')
    st.plotly_chart(fig, use_container_width=True)

##############################################
def prever_futuro_xgboost(brent, dias_previstos):

    try:
        modelo = joblib.load('modelos/xgboost_model.joblib')
    except FileNotFoundError:
        st.error("Modelo XGBoost não encontrado. Por favor, treine o modelo primeiro.")
        return
    ##########
    brent['data'] = pd.to_datetime(brent['data'], errors='coerce')
    ##########
    #df = df.sort_values('data').reset_index(drop=True)

    #####################
    # ✅ Remover possíveis linhas com data inválida
    brent = brent.dropna(subset=['data'])

    brent = brent.sort_values('data').reset_index(drop=True)
    ########################################

    ult_data = brent['data'].iloc[-1]
    st.write(f"Última data no dataset: {ult_data}")
    ult_precos = brent['brent_price'].tolist()

    previsoes = []

    for i in range(dias_previstos):
        lags = []
        for lag in [1, 2, 3, 7, 14, 30]:
            idx = len(ult_precos) - lag
            if idx >= 0:
                lags.append(ult_precos[idx])
            else:
                lags.append(ult_precos[0])  # fallback

        X_pred = pd.DataFrame([lags], columns=[f'lag_{l}' for l in [1, 2, 3, 7, 14, 30]])
        X_pred = X_pred.astype(float)

        pred = modelo.predict(X_pred)[0]

        ult_precos.append(pred)
        previsoes.append(pred)

    datas_futuras = [ult_data + pd.Timedelta(days=i+1) for i in range(dias_previstos)]

    # Verificação de segurança
    if len(datas_futuras) != len(previsoes):
        st.error(f"Tamanho incompatível: datas_futuras={len(datas_futuras)}, previsoes={len(previsoes)}")
        return

    df_previsao = pd.DataFrame({'data': datas_futuras, 'previsao': previsoes})

    st.subheader(f'Previsão para os próximos {dias_previstos} dias')
    st.dataframe(df_previsao.set_index('data'))

    fig = px.line(df_previsao, x='data', y='previsao', title='Previsão XGBoost')
    st.plotly_chart(fig, use_container_width=True)
    
###################
def validar_xgboost_timeseries(brent):
    st.subheader('📊 Validação Cruzada com TimeSeriesSplit - XGBoost')

    lags = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_30']
    brent = brent.dropna(subset=lags + ['brent_price']).copy()

    X = brent[lags]
    y = brent['brent_price']

    modelo = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    pipeline = make_pipeline(StandardScaler(), modelo)

    tscv = TimeSeriesSplit(n_splits=5)

    resultados = cross_validate(
        pipeline, X, y,
        cv=tscv,
        scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
        return_train_score=False
    )

    mse = -resultados['test_neg_mean_squared_error'].mean()
    mae = -resultados['test_neg_mean_absolute_error'].mean()
    r2 = resultados['test_r2'].mean()
    rmse = mse ** 0.5

    st.write(f'MSE: {mse:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R² Médio: {r2:.2f}')

#####################
def validar_modelo_xgboost_com_crossval(brent):
    lags = [f'lag_{lag}' for lag in [1, 2, 3, 7, 14, 30]]

    # Remove valores ausentes
    brent = brent.dropna(subset=lags + ['brent_price'])

    X = brent[lags]
    y = brent['brent_price']

    modelo_xgb = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )

    pipeline = make_pipeline(StandardScaler(), modelo_xgb)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    resultados = cross_validate(
        pipeline, X, y,
        cv=kf,
        scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
        return_train_score=False
    )

    mse = -resultados['test_neg_mean_squared_error'].mean()
    mae = -resultados['test_neg_mean_absolute_error'].mean()
    r2 = resultados['test_r2'].mean()
    rmse = mse ** 0.5

    st.subheader('📊 Validação Cruzada (5-fold) - XGBoost')
    st.write(f'MSE: {mse:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R² Médio: {r2:.2f}')

# ============================
# Aplicativo principal
# ============================

def execute_xgboost():
    st.title('📈 Previsão de Preço do Brent')
    st.write("Este app utiliza Random Forest e XGBoost para prever preços futuros com base na base de dados do Petróleo Bruto - Brent.")

    try:
        connection = conn()
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM safra_cana.tbl_brent')
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=col_names)
        cursor.close()
        encerra_conn(connection)
    except Exception as e:
        st.error(f"Erro ao acessar o banco de dados: {e}")
        return

    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    
    st.subheader('🔎 Visualização dos dados')

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Início do DataFrame**")
        st.write(df.head())

    with col2:
        st.write("**Fim do DataFrame**")
        st.write(df.tail())
    
    df = criar_caracteristicas(df)

    dias_previstos = st.slider('Selecione o número de dias para prever:', 1, 60, 7, help='Arraste para ver a previsão')

    # Só roda XGBoost agora, sem seleção de modelo
    if st.button('Treinar modelo XGBoost'):
        y_teste, y_pred, modelo, X_test, y_test = treinar_modelo_xgboost(df)
        avaliar_modelo(y_teste, y_pred)
        validar_modelo_xgboost_com_crossval(df) # nova função
        validar_xgboost_timeseries(df)
        st.write('A avaliação do Modelo XGBoost apresentou métricas (MSE, RMSE, MAE e R²) bem distintas.\
                  Quando analisado o Desempenho no Holdout (dados de teste) as métricas apresentam erros altos e\
                      R² Forte, o que indicam que o modelo está explicando quase toda a variabilidade dos dados.\
                          Contudo, a Validação Cruzada (k-fold tradicional) gerou saídas mais altas para erros e baixo R² também mais alto.\
                              Isto é, na Validação Cruzada o desempenho melhorou o que não é comun, sugere um desempenho muito ruim. O modelo não explica a variância dos dados. A validação com TimeSeriesSplit\
                                      é mais apropriada para analise de séries temporais e só veio a confirmar que o modelo pode estar \
                                          superajustando os dados ao conjunto de teste e não generalizando bem os períodos futuros.  Erros\
                                  significamente maiores sugerem overfitting ou dependência temporal. Portanto,\
                                              o modelo que deve ser escolhido nesta availiação é o RANDOM-FOREST.')
        
        if not os.path.exists('modelos'):
            os.makedirs('modelos')
        joblib.dump(modelo, 'modelos/xgboost_model.joblib')  # SALVA O MODELO TREINADO
        st.success("Modelo treinado e salvo com sucesso!")
    
    #if st.button('Prever futuro com XGBoost'):
    #       prever_futuro_xgboost(df, dias_previstos)