from BERT_recomend_construct import Recommender
import os
from dotenv import load_dotenv

load_dotenv()

model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
emb_cols = ['apresentacao', 'conteudo_programatico']

recomender = Recommender(
    DB_USER = os.getenv("DB_USER"),
    DB_PASS=os.getenv("DB_PASS"),
    DB_NAME=os.getenv("DB_NAME"),
    DB_HOST=os.getenv("DB_HOST"),
    model_name=model,
    emb_cols=emb_cols,
    id_col= 'id_curso',
    item_name_col = 'nome_curso'
)

##carrego a tabela do DB
recomender.carrega_dados('cursos')
##aplico limpeza simples aos dados
recomender.limpa_dados()
##extraio os embeddings
embeddings = recomender.embeddings_extract()

# Verifique se embeddings foram gerados corretamente
if embeddings is not None:
    recomendacoes = recomender.recomendar_itens(390, 4)
else:
    print("Erro: Os embeddings não foram gerados, verifique os dados.")

