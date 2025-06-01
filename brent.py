import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import openpyxl
import psycopg2 as ps
import plotly.graph_objs as go
import plotly.figure_factory as ff

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import joblib
import warnings
warnings.filterwarnings('ignore')
from connect import conn, encerra_conn
import modelo as md

geopolitica = pd.read_excel('quadro_geoeconomico_brent.xlsx', skiprows=3)
geopolitica0 = geopolitica.iloc[1:,:]

def mostrar_brent():
    connection = conn()
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM safra_cana.tbl_brent')
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description]
    brent = pd.DataFrame(rows, columns=col_names)

    st.write(' ##### Análise exploratória de dados (EDA) e cria visualizações')

    st.write('Gráfico Diário: Evolução Diária dos Preços do Petróleo Brent')
    brent['brent_price'] = pd.to_numeric(brent['brent_price'], errors='coerce')
    figura = px.line(brent, x="data", y="brent_price", 
                     labels={"data": "Data - Diário", "brent_price": "Preço Petróleo Bruto"},
                     title="Evolução Diária dos Preços do Petróleo Brent")
    st.line_chart(data=brent, x="data", y='brent_price', use_container_width=True)
    #st.plotly_chart(figura, use_container_width=True)

    st.write('Estatística da Série Temporal do Brent (FOB) - DIÁRIO')
    stats = brent['brent_price'].describe().to_frame().T
    st.dataframe(stats.round(2))

    brent['data'] = pd.to_datetime(brent['data'], errors='coerce')
    brent['AnoMes'] = brent['data'].dt.to_period('M')
    media_mensal = brent.groupby('AnoMes')['brent_price'].mean().reset_index()
    media_mensal['AnoMes'] = media_mensal['AnoMes'].astype(str)

    st.write('Gráfico Mensal: Evolução dos Preços Médios do Petróleo Brent')
    fig_mensal = px.line(media_mensal, x="AnoMes", y="brent_price",
                         labels={"AnoMes": "Data - Mensal", "brent_price": "Preço Médio do Petróleo"},
                         title="Evolução Mensal dos Preços Médios do Brent")
    st.plotly_chart(fig_mensal, use_container_width=True)

    st.write('Estatística da Série Temporal do Brent (FOB) - MENSAL')
    stats1 = media_mensal['brent_price'].describe().to_frame().T
    st.dataframe(stats1.round(2))

    hist_data = [
        brent['brent_price'].dropna().tolist(),
        media_mensal['brent_price'].dropna().tolist()
    ]   
    group_labels = ['Mensal','Diario']
    fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])
    fig.update_layout(title_text="Preços Diários e Mensais Médios")
    st.plotly_chart(fig)

    brent['Mes'] = brent['data'].dt.month
    sazonal = brent.groupby('Mes')['brent_price'].mean().reset_index()
    fig1 = px.bar(sazonal, x='Mes', y='brent_price', text_auto='.2s',
                  title="Preços Médios do Brent - Sazonalidade")
    fig1.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig1.update_xaxes(tickmode='linear', title_text="Meses")
    fig1.update_yaxes(title_text="Preços Médios Brent")
    st.plotly_chart(fig1, theme="streamlit")

    st.write('RESUMO DAS QUESTÕES GEOPOLÍTICAS E ECONÔMICAS')
    st.dataframe(geopolitica0, hide_index=True)

    encerra_conn(connection)
    return brent

def criar_caracteristicas(brent):
    for lag in [1, 2, 3, 7, 14, 30]:
        brent[f'lag_{lag}'] = brent['brent_price'].shift(lag)
    for janela in [7, 14, 30]:
        brent[f'media_movel_{janela}'] = brent['brent_price'].rolling(window=janela).mean()
    brent['dia_da_semana'] = brent['data'].dt.dayofweek
    brent['mes'] = brent['data'].dt.month
    brent['ano'] = brent['data'].dt.year
    return brent.dropna()

def treinar_modelo_ml(brent):
    caracteristicas = [col for col in brent.columns if col.startswith('lag_') or col.startswith('media_movel_') or
                       col in ['dia_da_semana', 'mes', 'ano']]
    X = brent[caracteristicas]
    y = brent['brent_price']
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
    escalador = StandardScaler()
    X_treino_escalado = escalador.fit_transform(X_treino)
    X_teste_escalado = escalador.transform(X_teste)
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(X_treino_escalado, y_treino)
    y_pred = modelo_rf.predict(X_teste_escalado)
    joblib.dump(modelo_rf, 'rf_model.joblib')
    joblib.dump(escalador, 'scaler.joblib')
    return X_treino_escalado, X_teste_escalado, y_treino, y_teste, y_pred, modelo_rf, escalador, caracteristicas

def avaliar_modelo(y_teste, y_pred):
    mse = mean_squared_error(y_teste, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_teste, y_pred)
    r2 = r2_score(y_teste, y_pred)
    print("\n=== Avaliação do Modelo ===")
    print(f"Erro Quadrático Médio: {mse:.2f}")
    print(f"Raiz do Erro Quadrático Médio: {rmse:.2f}")
    print(f"Erro Absoluto Médio: {mae:.2f}")
    print(f"Pontuação R²: {r2:.2f}")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_teste, y_pred, alpha=0.5)
    plt.plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], 'r--', lw=2)
    plt.xlabel('Preço Real')
    plt.ylabel('Preço Previsto')
    plt.title('Preços Reais vs Previstos')
    st.pyplot(plt.gcf())
    return mse, rmse, mae, r2

def criar_previsao_series_temporais(df):
    df.columns = [col.strip() for col in df.columns]  # Remover espaços extras
    if 'DATA' not in df.columns or 'PREÇO - PETRÓLEO BRUTO' not in df.columns:
        st.error("Colunas esperadas 'DATA' e 'PREÇO - PETRÓLEO BRUTO' não encontradas no DataFrame.")
        return None, None
    prophet_df = df[['DATA', 'PREÇO - PETRÓLEO BRUTO']].rename(columns={'DATA': 'ds', 'PREÇO - PETRÓLEO BRUTO': 'y'})
    modelo = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    modelo.fit(prophet_df)
    futuro = modelo.make_future_dataframe(periods=30)
    previsao = modelo.predict(futuro)
    st.pyplot(modelo.plot(previsao))
    st.pyplot(modelo.plot_components(previsao))
    return modelo, previsao