import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import psycopg2 as ps

from oferta import mostrar_oferta
from precos import mostrar_preco
from brent import mostrar_brent
from consumo import mostrar_consumo

from modelo.model_random import execute_random
from modelo.model_xgboost import execute_xgboost

from connect import conn, encerra_conn

#oferta_page = st.Page("oferta.py", title="OFERTA CANA CONAB")
#demanda_page = st.Page("consumo.py", title="CONSUMO COMBUSTÍVEL ANP")
#anprecos_page = st.Page('precos.py', title = 'PREÇOS COMBUSTÍVEL ANP')
#brent_page = st.Page('brent.py', title= 'PREÇOS BRENT')

#pg = st.navigation([oferta_page, demanda_page, anprecos_page, brent_page])
#st.set_page_config(page_title="Data manager")

#pg.run()

st.set_page_config(page_title="Data Manager")

pagina = st.sidebar.selectbox(
    "Selecione a página",
    ["Oferta", "Demanda", "Preços ANP", "Brent", "Modelos"]
)

if pagina == "Modelos":
    # Menu secundário aparece só quando "Brent" está selecionado
    #modelo = st.sidebar.radio("Escolha o modelo",["Random Forest", "Prophet"])
    modelo = st.sidebar.selectbox("Escolha o modelo", ["", "Random Forest", "XGBoost"], index=0)  # Começa selecionado na opção vazia)
    
else:
    modelo = None  # Não há modelo quando não é Brent


if pagina == "Oferta":
    # Conteúdo da oferta
    #st.title("OFERTA CANA CONAB")
    st.title("DASHBOARD DA OFERTA DE ETANOL :fuelpump: E AÇÚCAR :ear_of_rice:")
    mostrar_oferta()
elif pagina == "Demanda":
    #st.title("CONSUMO COMBUSTÍVEL ANP")
    st.title("DASHBOARD DO CONSUMO DE COMBUSTÍVEIS ANP :fuelpump:")
    mostrar_consumo()
elif pagina == "Preços ANP":
    #st.title("PREÇOS COMBUSTÍVEL ANP")
    st.title("DASHBOARD DE PREÇOS DE COMBUSTÍVEIS ANP :fuelpump:")
    mostrar_preco()
elif pagina == "Brent":
    #st.title("PREÇOS BRENT")
    st.title("DASHBOARD DE PREÇOS DO PETRÓLEO BRUTO (FOB) :fuelpump:")
    mostrar_brent()
elif pagina == 'Modelos':    
    if modelo == "":
        st.write("Por favor, selecione um modelo no menu lateral.")
    elif modelo == "Random Forest":
        st.subheader("Modelo Random Forest")
        execute_random()
    elif modelo == "XGBoost":
        st.subheader("Modelo XGBoost")
        execute_xgboost()