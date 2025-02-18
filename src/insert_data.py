import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

# Configuração do banco de dados PostgreSQL
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

##cod para escrever dados n o meu postgresl deploy no render
df = pd.read_csv('/Users/pedroh.mello/Desktop/MESTRADO_MCDE/SEMINARIO/PROJETO_CONCLUSAO_MESTRADO/data/books_data_ph.csv')

conn_str = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}'
engine = create_engine(conn_str)

# inserir dados na tabl 'cursos'
if df.to_sql('livros', engine, if_exists='append', index=False):
    print('Dados inseridos com sucesso!')
else:
    print('algo deu errado!')
