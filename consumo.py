import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import psycopg2 as ps
from connect import conn, encerra_conn

def mostrar_consumo():
    # Conexão com banco
    connection = conn()
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM safra_cana.tbl_canadem')
    rows = cursor.fetchall()

    # Obtem os nomes das colunas
    col_names = [desc[0] for desc in cursor.description]
    data = pd.DataFrame(rows, columns=col_names)
    
    # Converte colunas de data
    data['ano'] = pd.to_datetime(data['ano'], errors='coerce').dt.year
    
    # Resultado agregado
    resultado = data.groupby(['ano','regioes', 'tp_produto'])[['vendas_m3','vendas_litros']].mean().reset_index()

    resultado1 = data.groupby(['ano', 'uf', 'regioes', 'tp_produto'])[['vendas_m3', 'vendas_litros']].mean().reset_index()
    
    data1 = data.groupby('ano')[['vendas_m3', 'vendas_litros']].sum()
    
    reg_norte = resultado1[resultado1['regioes'] == 'REGIÃO NORTE']
    reg_nordeste = resultado1[resultado1['regioes'] == 'REGIÃO NORDESTE']
    reg_sudeste = resultado1[resultado1['regioes'] == 'REGIÃO SUDESTE']
    reg_co = resultado1[resultado1['regioes'] == 'REGIÃO CENTRO-OESTE']
    reg_sul = resultado1[resultado1['regioes'] == 'REGIÃO SUL']

    def formatar_br(numero):
        return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    
    
    # Título
    #st.header("DASHBOARD DO CONSUMO DE COMBUSTÍVEIS ANP :fuelpump:")
    
    ########################### Metricas
    # Layout com 6 colunas
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    # Coluna 1 
    with col1:
        ano20 = data1.loc[2020, 'vendas_litros'] / 1_000_000_000
        col1.metric("Consumo em 2020 - bl", formatar_br(ano20), ' ')

    # Coluna 2 
    with col2:
        ano21 = data1.loc[2021, 'vendas_litros'] / 1_000_000_000
        varperc21 = (ano21 - ano20) / ano21 * 100
        col2.metric("Consumo em 2021 - bl", formatar_br(ano21), f"{varperc21:.2f}%")

    # Coluna 3
    with col3:
        ano22 = data1.loc[2022, 'vendas_litros'] / 1_000_000_000
        varperc22 = (ano22 - ano21) / ano22 * 100
        col3.metric("Consumo em 2022 - bl", formatar_br(ano22), f"{varperc22:.2f}%")

    # Coluna 4
    with col4:
        ano23 = data1.loc[2023, 'vendas_litros'] / 1_000_000_000
        varperc23 = (ano23 - ano22) / ano23 * 100
        col4.metric("Consumo em 2023 - bl", formatar_br(ano23), f"{varperc23:.2f}%")

    # Coluna 5 - Nova métrica (Exemplo: Produção de Etanol por hectare)
    with col5:
        ano24 = data1.loc[2021, 'vendas_litros'] / 1_000_000_000
        varperc24 = (ano24 - ano23) / ano24 * 100
        col5.metric("Consumo em 2024 - bl", formatar_br(ano24), f"{varperc24:.2f}%")
            
    # Coluna 6 - Nova métrica (Exemplo: Produtividade total)
    with col6:
        ano25 = data1.loc[2025, 'vendas_litros'] / 1_000_000_000
        varperc25 = (ano25 - ano24) / ano25 * 100
        col6.metric("Consumo em 2025 - bl", formatar_br(ano25), f"{varperc25:.2f}%")
    
    #################################################################################
    st.dataframe(resultado, hide_index=True)
    
    ########################### Figuras
    fig0 = px.pie(resultado,
             values='vendas_litros',
             names='regioes',
             title='Consumo de Combustível por Região')

    fig1 = px.bar(data,
              x='tp_produto',
              y='vendas_litros',
              text_auto='.2s',
              color = 'tp_produto',
              labels={'vendas_litros': 'Litros', 'tp_produto': 'Produtos'},
              title='Combustíveis Vendidos por Produto')
    
    fig2 = px.bar(
        reg_norte,
        x="uf",
        y="vendas_litros",
        color="tp_produto",
        barmode="group",
        title="Distribuição de Vendas por Produto - Região Norte"
    )
    
    
    fig3 = px.bar(
        reg_nordeste,
        x="uf",
        y="vendas_litros",
        color="tp_produto",
        barmode="group",
        title="Distribuição de Vendas por Produto - Região Centro-Oeste"
    )
    
    
    fig4 = px.bar(
        reg_co,
        x="uf",
        y="vendas_litros",
        color="tp_produto",
        barmode="group",
        title="Distribuição de Vendas por Produto - Região Centro-Oeste"
    )
    
    
    fig5 = px.bar(
        reg_sudeste,
        x="uf",
        y="vendas_litros",
        color="tp_produto",
        barmode="group",
        title="Distribuição de Vendas por Produto - Região Sudeste"
    )
    
    fig6 = px.bar(
        reg_sul,
        x="uf",
        y="vendas_litros",
        color="tp_produto",
        barmode="group",
        title="Distribuição de Vendas por Produto - Região Sul"
    )
    
    #####################################################################
    # Informação sobre combustível
    
    st.write('O óleo diesel é um combustível utilizado para motores de combustão por compressão (motores diesel).')
    
    st.write('O óleo combustível é utilizado em grandes equipamentos industriais, como caldeiras e usinas termelétricas, para gerar calor ou eletricidade.')
    
    ######################################################################
    # Exibição em abas com temas diferentes
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Consumo/Região", 'Produtos Comercializados',"Reg. Norte", 'Reg. Nordeste', 'Reg. Centro-Oeste', 'Reg. Sudeste', 'Reg. Sul'])
    with tab1:
        st.plotly_chart(fig0, theme="streamlit")
    with tab2:
        st.plotly_chart(fig1, theme="streamlit")
    with tab3:
        st.plotly_chart(fig2, theme="streamlit")
    with tab4:
        st.plotly_chart(fig3, theme="streamlit")
    with tab5:
        st.plotly_chart(fig4, theme="streamlit")
    with tab6:
        st.plotly_chart(fig5, theme="streamlit")
    with tab7:
        st.plotly_chart(fig6, theme="streamlit")


    encerra_conn(connection)

#if __name__ == "__main__":
#    main()
