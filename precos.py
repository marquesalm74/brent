import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import psycopg2 as ps
from connect import conn, encerra_conn


# conecção db
def mostrar_preco():
    
    connection =  conn()
    cursor = connection.cursor()
    
    cursor.execute('SELECT regiao, estado, municipio, produto, data_coleta, valor_venda, bandeira FROM safra_cana.tbl_anprecos')
    rows = cursor.fetchall()
    
    # Obtem os nomes das colunas
    col_names = [desc[0] for desc in cursor.description]
    data = pd.DataFrame(rows, columns = col_names)
    
    data['data_coleta'] = pd.to_datetime(data['data_coleta'], errors='coerce')
    data['ano'] = data['data_coleta'].dt.year
    
    ####
    resultado = data.groupby(['ano','produto'])['valor_venda'].mean().round(2).reset_index()
    
    
    #################### Filtros - Segmentação #############
    anos = sorted(data['ano'].unique())
    produtos = sorted(data['produto'].unique())

    # Slider para selecionar o ano
    ano_selecionado = st.select_slider('Selecione o ano', options=anos)

    # Filtrar os dados do ano selecionado
    dados_ano = resultado[resultado['ano'] == ano_selecionado].drop_duplicates(subset='produto').reset_index(drop=True)

    # Primeira linha: 3 colunas
    col1, col2, col3 = st.columns(3)
    # Segunda linha: 3 colunas
    col4, col5, col6 = st.columns(3)

    # Lista com todas as colunas
    colunas = [col1, col2, col3, col4, col5, col6]

    # Título
    st.write(f" Preços médios dos combustíveis em {ano_selecionado}")

    # Exibir métricas em linhas de 3 colunas
    for i in range(0, len(dados_ano), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(dados_ano):
                row = dados_ano.iloc[i + j]
                col.metric(label=row['produto'], value=f"R$ {row['valor_venda']:.2f}")
    
    
    ########################## Figuras ###############################################
    fig0 = px.line(resultado, x="ano", y="valor_venda", color="produto",
               line_group="produto", hover_name="produto",
               line_shape="spline", render_mode="svg")
    
    media_por_estado = (data.groupby(['estado', 'produto'])['valor_venda'].mean().reset_index())
    media_por_estado['valor_venda'] = media_por_estado['valor_venda'].astype(float).round(2)
    fig1 = px.treemap(media_por_estado,
                    path=[px.Constant('Brasil'), 'estado', 'produto'],
                    values=media_por_estado['valor_venda'],
                    color='estado',
                    title="Preço médio por Região e Estado")
    fig1.update_traces(root_color="lightgrey")
    fig1.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    
    #repr_bandeira = data.groupby(['estado', 'bandeira'])['bandeira'].value_counts()
    #fig2 = px.histogram(data, x="bandeira", color="produto", marginal="rug", hover_data=data.columns)
    
    repr_bandeira = data.groupby(['estado', 'bandeira']).size().reset_index(name='frequencia')
    fig2 = px.bar(repr_bandeira, y='frequencia', x='bandeira', color = 'estado', text_auto='.2s',
                title="Frequência de Bandeiras por Estado")
    fig2.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    #fig1 = px.sunburst(data, path=['regiao', 'bandeira'], values=data['valor_venda'].mean())
    #fig1 = px.bar(resultado, y='estado', x='bandeira', text_auto='.2s',title="Representatividade das Bandeiras")
    #fig1.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

    tab1, tab2, tab3 = st.tabs(["Preços no Tempo", "Heatmap Regiao-Estado", "Representação Bandeiras"]) #, tab2, tab3 , "Bandeiras", "Heatmap Regiao-Estado"
    with tab1:
        st.plotly_chart(fig0, theme="streamlit")
    with tab2:
        st.plotly_chart(fig1, theme='streamlit')
    with tab3:
        st.plotly_chart(fig2, theme='streamlit')

    
       
    #################################################################################
    st.write("Tabela Preços de Combustível [min, medio, max] - ANP")
    anp = data[['ano', 'data_coleta','regiao','estado', 'municipio','produto', 'bandeira', 'valor_venda']]
    anp_agrupado = anp.groupby(['ano', 'produto']).agg({'valor_venda' : ['min', 'mean', 'max', 'std']}).round(2)
    
    st.dataframe(anp_agrupado)
    
    encerra_conn(connection)
    
#if __name__ == "__main__":
#    main()    
