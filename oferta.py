import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import psycopg2 as ps
import pydeck as pdk
from connect import conn, encerra_conn

# st.write("Localização das Usinas e Destilarias: Produção de Açúcar e Etanol - Brasil e UF's.")

def mostrar_oferta():
    # Conexão com o banco de dados
    connection = conn()
    cursor = connection.cursor()

    cursor.execute('SELECT * FROM safra_cana.tbl_canaof')
    rows = cursor.fetchall()

    # Obtem os nomes das colunas
    col_names = [desc[0] for desc in cursor.description]
    data = pd.DataFrame(rows, columns=col_names)

    # Título
    #st.header("DASHBOARD DA OFERTA DE ETANOL :fuelpump: E AÇÚCAR :ear_of_rice:")

    def formatar_br(numero):
        return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    # Filtro Ano-Safra
    ano_safra = st.select_slider("Selecione o Ano-Safra", data['ano_safra'].unique())

    # Filtro de levantamento por ano
    if ano_safra == '2020-2021':
        opcoes_levantamento = [4]
    elif ano_safra in ['2021-2022', '2022-2023', '2023-2024']:
        opcoes_levantamento = sorted(data[data['ano_safra'] == ano_safra]['levantamento'].unique())
    elif ano_safra == '2024-2025':
        levantamentos = data[data['ano_safra'] == '2024-2025']['levantamento'].unique()
        opcoes_levantamento = sorted(levantamentos) if len(levantamentos) > 0 else []
    else:
        opcoes_levantamento = sorted(data[data['ano_safra'] == ano_safra]['levantamento'].unique())

    #levantamento = st.select_slider("Selecione o Levantamento de Safra", options=opcoes_levantamento)
    if len(opcoes_levantamento) > 1:
        levantamento = st.select_slider("Selecione o Levantamento de Safra", options=opcoes_levantamento)
    elif len(opcoes_levantamento) == 1:
        levantamento = opcoes_levantamento[0]
        st.info(f"Levantamento único disponível: {levantamento}")
    else:
        st.warning("⚠️ Nenhum levantamento disponível para essa safra.")
        return


    # Filtra os dados
    dados_filtrados = data[(data['ano_safra'] == ano_safra) & (data['levantamento'] == levantamento)]

    if dados_filtrados.empty:
        st.warning("⚠️ Nenhum dado disponível para essa safra e levantamento selecionados.")
        return

    # Formatação dos dados
    df_formatado = dados_filtrados.copy()
    colunas_numericas = ['area_propria_ha', 'area_fornecedor_ha', 'area_total_ha',
                         'producao_etanol_m3', 'producao_acucar_ton', 'atr', 'tch']

    for coluna in colunas_numericas:
        df_formatado[coluna] = df_formatado[coluna].apply(formatar_br)

    # Layout com colunas
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    with col1:
        etanol = dados_filtrados['producao_etanol_m3'].sum() / 1000
        col1.metric("Produção de Etanol (Mil m³)", f"{formatar_br(etanol)}")

    with col2:
        acucar = dados_filtrados['producao_acucar_ton'].sum() / 1000
        col2.metric("Produção de Açúcar - Mt", f"{formatar_br(acucar)}")

    with col3:
        cana_acucar = dados_filtrados['producao_acucar_ton'].sum() / 115
        cana_etanol = dados_filtrados['producao_etanol_m3'].sum() / 85
        tot_cana = cana_acucar + cana_etanol

        if tot_cana > 0:
            mix_acucar = cana_acucar / tot_cana
            mix_etanol = cana_etanol / tot_cana
            if mix_acucar > mix_etanol:
                col3.metric("Mix de Produção - Açúcar", f"{(mix_acucar * 100):.1f}%")
            else:
                col3.metric("Mix de Produção - Etanol", f"{(mix_etanol * 100):.1f}%")
        else:
            col3.metric("Mix de Produção", "Sem dados")

    with col4:
        tch = dados_filtrados['tch'].mean()
        col4.metric("Produtividade - TCH (t/ha)", formatar_br(tch))

    with col5:
        atr = dados_filtrados['atr'].mean()
        col5.metric("ATR Médio", formatar_br(atr))

    with col6:
        produtividade_total = dados_filtrados['tch'].sum()
        col6.metric("Produtividade Total", f"{formatar_br(produtividade_total)} t")

    if mix_acucar > mix_etanol:
        df_sorted = dados_filtrados.sort_values(by='producao_acucar_ton', ascending=False)
    else:
        df_sorted = dados_filtrados.sort_values(by='producao_etanol_m3', ascending=False)

    # Agrupamento por roteiro
    roteiros = dados_filtrados.groupby(['roteiro']).agg({
        'area_propria_ha': 'sum',
        'area_fornecedor_ha': 'sum',
        'area_total_ha': 'sum',
        'producao_mil_ton': 'sum',
        'producao_etanol_m3': 'sum',
        'producao_acucar_ton': 'sum',
        'atr': 'mean',
        'tch': 'mean'
    }).sort_values(by='area_total_ha', ascending=False)

    st.dataframe(
        roteiros.style
        .highlight_max(axis=0)
        .format({col: formatar_br for col in roteiros.columns})
    )

    st.dataframe(df_sorted[['cod_firma', 'area_propria_ha', 'area_fornecedor_ha',
                            'producao_mil_ton', 'producao_etanol_m3', 'producao_acucar_ton',
                            'atr', 'tch']], hide_index=True)

    # Mapa
    st.write("Localização das Usinas e Destilarias: Produção de Açúcar e Etanol - Brasil e UF's.")
    if 'latitude' in dados_filtrados.columns and 'longitude' in dados_filtrados.columns:
        map_data = (
            dados_filtrados.groupby(['cod_firma', 'latitude', 'longitude'], as_index=False) # descr_firma
            .agg({'producao_etanol_m3': 'sum', 'producao_acucar_ton': 'sum'})
        )
        map_data = map_data.dropna(subset=['latitude', 'longitude', 'producao_etanol_m3', 'producao_acucar_ton'])
        map_data['producao_etanol_m3'] = map_data['producao_etanol_m3'].astype(float)
        map_data['producao_acucar_ton'] = map_data['producao_acucar_ton'].astype(float)

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position='[longitude, latitude]',
            get_radius=["producao_etanol_m3", "producao_acucar_ton"],
            radius_scale=0.05,
            get_fill_color='[0, 102, 204, 160]',
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=map_data["latitude"].mean(),
            longitude=map_data["longitude"].mean(),
            zoom=4,
            pitch=0
        )

        tooltip = {
            "html": "<b>{cod_firma}</b><br/>Etanol: {producao_etanol_m3} m³<br/>Açúcar: {producao_acucar_ton} ton", #descr_firma x cod_firma
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
    else:
        st.warning("Colunas de latitude e longitude ausentes no DataFrame.")

    # Fecha conexão
    encerra_conn(connection)