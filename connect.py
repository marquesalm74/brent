import psycopg2 as ps
from psycopg2 import Error
from dotenv import load_dotenv
import os

load_dotenv()

def conn():
    try:
        pwd = os.getenv('DB_PASSWORD')
        conecta = ps.connect(
            
            user = 'postgres',
            password = pwd,
            host = 'localhost',
            port = 5432,
            database = 'db_segeo')
        
        print('Conectado com Sucesso')
    
        return conecta
    
    except Error as e:
        print(f'Ocorreu um erro ao tentar conectar no DB {e}')
        
def encerra_conn(conecta):
    if conecta:
        conecta.close()