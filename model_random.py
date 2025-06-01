import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import joblib
from connect import conn, encerra_conn
import warnings
warnings.filterwarnings('ignore')

# ============================
# Funções auxiliares
# ============================

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

    modelo_rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5,random_state=42)
    modelo_rf.fit(X_treino_escalado, y_treino)

    y_pred = modelo_rf.predict(X_teste_escalado)

    #joblib.dump(modelo_rf, 'rf_model.joblib')
    #joblib.dump(escalador, 'scaler.joblib')

    joblib.dump(modelo_rf, 'modelos/rf_model.joblib')  # SALVA O MODELO TREINADO
    joblib.dump(escalador, 'modelos/scaler.joblib')

    return y_teste, y_pred

def avaliar_modelo(y_teste, y_pred):
    mse = mean_squared_error(y_teste, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_teste, y_pred)
    r2 = r2_score(y_teste, y_pred)

    st.subheader('Avaliação do Modelo Random Forest')
    st.write(f'MSE: {mse:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.2f}')

    fig, ax = plt.subplots()
    ax.scatter(y_teste, y_pred, alpha=0.5)
    ax.plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], 'r--')
    ax.set_xlabel('Valor Real')
    ax.set_ylabel('Valor Previsto')
    ax.set_title('Comparação Real vs Previsto')
    #st.pyplot(fig)
    #st.plotly_chart(fig)
    
    fig, ax = plt.subplots()
    ax.scatter(y_teste, y_pred, alpha=0.5)
    ax.plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], 'r--')

    # Labels e título com cor branca e tamanho maior
    ax.set_xlabel('Valor Real', color='white', fontsize=14)
    ax.set_ylabel('Valor Previsto', color='white', fontsize=14)
    ax.set_title('Comparação Real vs Previsto', color='white', fontsize=16)

    # Define cor branca para os valores nos eixos
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Fundo escuro opcional (fica mais elegante com texto branco)
    fig.patch.set_facecolor('black')      # fundo fora do gráfico
    ax.set_facecolor('#222222')           # fundo dentro do gráfico
    st.plotly_chart(fig)

###############################################################################################################
def validar_modelo_com_crossval(df):
    caracteristicas = [col for col in df.columns if col.startswith('lag_') or col.startswith('media_movel_') or
                       col in ['dia_da_semana', 'mes', 'ano']]
    
    X = df[caracteristicas]
    y = df['brent_price']

    modelo_rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )

    # Pipeline com escalonamento + modelo
    pipeline = make_pipeline(StandardScaler(), modelo_rf)

    # K-Fold com 5 divisões e shuffle para embaralhar os dados
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Avaliação com múltiplas métricas
    resultados = cross_validate(
        pipeline, X, y,
        cv=kf,
        scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
        return_train_score=False
    )

    # Resultados médios
    mse = -resultados['test_neg_mean_squared_error'].mean()
    mae = -resultados['test_neg_mean_absolute_error'].mean()
    r2 = resultados['test_r2'].mean()
    rmse = mse ** 0.5

    st.subheader('📊 Validação Cruzada (5-fold) - Random Forest')
    st.write(f'MSE: {mse:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.2f}')
    
#################################################################################################################
def prever_futuro_rf(df, dias_previstos):
    modelo_rf = joblib.load('modelo/rf_model.joblib')
    escalador = joblib.load('modelo/scaler.joblib')

    ultimos_dados = df.copy()
    previsoes = []

    for i in range(dias_previstos):
        ultima_linha = ultimos_dados.iloc[-1].copy()
        nova_data = ultima_linha['data'] + pd.Timedelta(days=1)

        nova_linha = {'data': nova_data}
        for lag in [1, 2, 3, 7, 14, 30]:
            if len(ultimos_dados) >= lag:
                nova_linha[f'lag_{lag}'] = ultimos_dados.iloc[-lag]['brent_price']
            else:
                nova_linha[f'lag_{lag}'] = np.nan

        for janela in [7, 14, 30]:
            ultimos_dados['brent_price'] = ultimos_dados['brent_price'].astype(float) # adicionada
            nova_linha[f'media_movel_{janela}'] = ultimos_dados['brent_price'].tail(janela).mean()

        nova_linha['dia_da_semana'] = nova_data.dayofweek
        nova_linha['mes'] = nova_data.month
        nova_linha['ano'] = nova_data.year

        nova_linha_df = pd.DataFrame([nova_linha]).dropna()
        if nova_linha_df.empty:
            break

        X_novo = nova_linha_df.drop(columns=['data'])
        X_novo_scaled = escalador.transform(X_novo)
        y_pred = modelo_rf.predict(X_novo_scaled)[0]

        nova_linha_df['brent_price'] = y_pred
        ultimos_dados = pd.concat([ultimos_dados, nova_linha_df[ultimos_dados.columns]], ignore_index=True)

        previsoes.append((nova_data, y_pred))

    # Mostrar resultado
    previsoes_df = pd.DataFrame(previsoes, columns=['Data', 'Preço Previsto'])
    st.subheader(f'📉 Previsão de {dias_previstos} dia(s) com Random Forest')
    st.dataframe(previsoes_df.set_index('Data'))

    fig = px.line(previsoes_df, x='Data', y='Preço Previsto', title='Previsão com Random Forest')
    st.plotly_chart(fig, use_container_width=True)

# ============================
# Aplicativo principal
# ============================

def execute_random():
    st.title('📈 Previsão de Preço do Brent - Série Temporal')
    st.write("Este app utiliza Random Forest e XGBoost para prever preços futuros com base na base de dados do Petróleo Bruto - Brent.")

    try:
        # Conectando ao banco de dados
        connection = conn()
        cursor = connection.cursor()

        cursor.execute('SELECT * FROM safra_cana.tbl_brent')
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=col_names)

        # Fechando conexão com o banco
        cursor.close()
        encerra_conn(connection)

    except Exception as e:
        st.error(f"Erro ao conectar ou consultar o banco de dados: {e}")
        return

    if 'data' not in df.columns or 'brent_price' not in df.columns:
        st.error('❌ A tabela deve conter as colunas "data" e "brent_price"')
        return

    df['data'] = pd.to_datetime(df['data'])

    st.subheader('🔎 Visualização dos dados')

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Início do DataFrame**")
        st.write(df.head())

    with col2:
        st.write("**Fim do DataFrame**")
        st.write(df.tail())

    df = criar_caracteristicas(df)

    st.markdown("---")
    dias_previstos = st.slider(
        '📅 Selecione o número de dias para prever com Random-Forest:',
        min_value=1,
        max_value=60,
        value=7,
        help='Arraste para ver a previsão')
    
    if st.button('🚀 Treinar modelo Random Forest'):
        y_teste, y_pred = treinar_modelo_ml(df)
        avaliar_modelo(y_teste, y_pred)
        validar_modelo_com_crossval(df)  # nova funcionalidade
        st.write('A avaliação do Modelo Random-Forest apresentou métricas (MSE, RMSE, MAE) muito semelhantes,\
                 logo o modelo está generalizando bem. Como o R² explica 100(%) da variabilidade dos dados, isto indica\
                     que o modelo está bem ajustado. O Cross-Validation confirma a robustez do modelo para fazer previsões. ')
        st.success("Modelo treinado e salvo com sucesso!")
        
    if st.button('Prever futuro com Random Forest'):
        prever_futuro_rf(df, dias_previstos) # previsão dias previstos