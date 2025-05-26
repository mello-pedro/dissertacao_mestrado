import pandas as pd
from sqlalchemy.engine import create_engine
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import os
import warnings
from dotenv import load_dotenv

pd.set_option('display.max_colwidth', None)
warnings.filterwarnings("ignore")
load_dotenv()

class Recommender:
    def __init__(self, DB_USER, DB_PASS, DB_NAME, DB_HOST, model_name: str, emb_cols, id_col: str, item_name_col: str):
        self.DB_USER = DB_USER
        self.DB_PASS = DB_PASS
        self.DB_HOST = DB_HOST
        self.DB_NAME = DB_NAME
        self.engine = create_engine(f'postgresql://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:5432/{self.DB_NAME}')

        ##chama o modelo
        self.model = SentenceTransformer(model_name)

        ##chama cols que serao usadas p/ extrair texto p/ vetores de embeddings
        self.emb_cols = emb_cols
        self.id_col = id_col
        self.item_name_col = item_name_col
        self.stop_words = set(stopwords.words('portuguese'))
        self.embeddings = None
        self.df_text = None
    
    ##func para carregar os dados (somente cols de interesse)
    def carrega_dados(self, tabela: str):
        try:
            #cols_str = ', '.join(self.emb_cols) + f', {self.item_name_col}, {self.id_col}'
            cols_str = ', '.join(self.emb_cols) + f', {self.item_name_col}, {self.id_col}, {self.item_name_col} AS nome_original'
            query = f'SELECT {cols_str} FROM {tabela}'
            self.df_text = pd.read_sql(query, self.engine)
            # coluna para armazenar os nomes dos cursos originais(util na hora de puxar recomends)
            print("Dados textuais carregados com sucesso \n")
            #print(self.df_text.head())
        except Exception as e:
            print(f"Erro ao carregar os dados! {e}")


    def limpa_dados(self):
        all_cols_to_clean = self.emb_cols + [self.item_name_col]
        for col in all_cols_to_clean:
            if col in self.df_text.columns:
                self.df_text[col] = (
                    self.df_text[col]
                    .fillna("")
                    .str.strip()
                    .str.lower()
                    .apply(lambda x: ' '.join(
                        [word for word in x.split() if word not in self.stop_words]
                    ) if isinstance(x, str) else x)
                )
        # junta o texto das colunas em uma nova coluna 'compilado_textual'
        self.df_text['compilado_textual'] = self.df_text[self.item_name_col] + ' ' + self.df_text[self.emb_cols].agg(' '.join, axis=1)
        print("Dados limpos, padronizados e texto compilado em uma única coluna.\n")
        return self.df_text
    

    def embeddings_extract(self, batch_size=64):
    # inicializar a coluna 'embeddings' com uma lista vazia
        self.df_text['embeddings'] = [None] * len(self.df_text)

        with torch.no_grad():
            # process em batches para economizar RAM
            for start in tqdm(range(0, len(self.df_text), batch_size), desc="Processando Embeddings"):
                end = min(start + batch_size, len(self.df_text))
                batch_texts = self.df_text['compilado_textual'][start:end].tolist()  # Usando a coluna compilado_textual
                
                # valida se há texto para processar
                if len(batch_texts) == 0:
                    print("Nenhum texto para processar no lote.")
                    continue
                
                # gera embeddings de cada lote
                try:
                    batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=True, normalize_embeddings=True)
                    #print(f"Embeddings do lote gerados com sucesso: {batch_embeddings}")(p/ teste apenas)

                    # corrige atribuição para evitar ChainedAssignmentError
                    self.df_text['embeddings'][start:end] = [emb.tolist() for emb in batch_embeddings]

                except Exception as e:
                    print(f"Erro ao gerar embeddings: {e}")
                    break  # Saia do loop se houver um erro

        try:
            # verificar se a coluna embeddings gerada não contém vazios
            if self.df_text['embeddings'].isnull().all():
                raise ValueError("Todos os embeddings são None. Verifique os dados.")

            # empilhar os embeddings em um tensor 2d(matriz)
            self.embeddings = torch.stack([torch.tensor(emb) for emb in self.df_text['embeddings'] if emb is not None])
            print("Vetores de representação textual gerados com sucesso pelo PyTorch!\n\n")
            #print(f"Shape dos embeddings gerados: {self.embeddings.shape}")
            return self.embeddings
        except Exception as e:
            print(f"Não foi possível gerar os vetores de representação textual. Erro: {e}")
            self.embeddings = None  

    
    ### FUNCIONA, PORÉM ATIVAR SOMENTE QND FOR COMPARAR 3 MEDIDAS DE SIMILARIDADE
    def recomendar_itens(self, id_item, top_n: int = 3):
        """Retorna recomendações baseadas em Similaridade do Cosseno, Distância Euclidiana e Distância de Manhattan."""
        id_item = str(id_item)
        self.df_text[self.id_col] = self.df_text[self.id_col].astype(str)

        if id_item not in self.df_text[self.id_col].values:
            print(f"ID {id_item} não encontrado na base de dados.")
            return None

            # Reset index and map IDs to positions
        df_reset = self.df_text.reset_index(drop=True)
        indice = pd.Series(df_reset.index, index=df_reset[self.id_col])
        idx = indice[id_item]

            # ------------------ Métricas ------------------
            # 1) Similaridade do Cosseno (quanto maior, mais similar)
        sim_scores = util.cos_sim(self.embeddings[idx], self.embeddings).squeeze().tolist()
            # 2) Distância Euclidiana (quanto menor, mais similar)
        eucl_distances = torch.norm(self.embeddings - self.embeddings[idx], dim=1).tolist()
            # 3) Distância de Manhattan (L1) (quanto menor, mais similar)
        manh_distances = torch.sum(torch.abs(self.embeddings - self.embeddings[idx]), dim=1).tolist()

            # ------------------ Ordenação ------------------
        sim_sorted = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)[1:top_n+1]
        eucl_sorted = sorted(enumerate(eucl_distances), key=lambda x: x[1])[1:top_n+1]
        manh_sorted = sorted(enumerate(manh_distances), key=lambda x: x[1])[1:top_n+1]

        sim_idx   = [i[0] for i in sim_sorted]
        eucl_idx  = [i[0] for i in eucl_sorted]
        manh_idx  = [i[0] for i in manh_sorted]

            # ------------------ DataFrames de Recomendação ------------------
        recomendacao_cos   = pd.DataFrame({
            'Item Recomendado': df_reset.loc[sim_idx, 'nome_original'].values,
            'Sim_Cosseno':      [score[1] for score in sim_sorted]
        }).reset_index(drop=True)

        recomendacao_eucl  = pd.DataFrame({
            'Item Recomendado': df_reset.loc[eucl_idx, 'nome_original'].values,
            'Dist_Euclidiana':  [dist[1] for dist in eucl_sorted]
        }).reset_index(drop=True)

        recomendacao_manh  = pd.DataFrame({
            'Item Recomendado': df_reset.loc[manh_idx, 'nome_original'].values,
            'Dist_Manhattan':   [dist[1] for dist in manh_sorted]
        }).reset_index(drop=True)

            # Ajuste de índice para apresentação (inicia em 1)
        recomendacao_cos.index += 1
        recomendacao_eucl.index += 1
        recomendacao_manh.index += 1

            # Nome do item consultado
        item_name = df_reset.loc[df_reset[self.id_col] == id_item, 'nome_original'].values[0].title()

        print(f"Recomendações para o item '{item_name}' (ID: {id_item}):\n")
        # print("▶️ Similaridade do Cosseno:")
        # print(recomendacao_cos)
        # print("\n▶️ Distância Euclidiana:")
        # print(recomendacao_eucl)
        # print("\n▶️ Distância de Manhattan:")
        # print(recomendacao_manh)
        
        return {
                "cosine": recomendacao_cos,
                "euclidean": recomendacao_eucl,
                "manhattan": recomendacao_manh
            }

        #return recomendacao_cos, recomendacao_eucl, recomendacao_manh


    
    #### FUNCIONA PERFEITAMENTE. VOU TENTAR ACIMA IMPLEMENTAR DIST EUCLIDEANA TB.
    # def recomendar_itens(self, id_item, top_n: int = 3):
    #     id_item = str(id_item)

    #     ## Garante que os IDs da base também são strings
    #     self.df_text[self.id_col] = self.df_text[self.id_col].astype(str)

    #     if id_item not in self.df_text[self.id_col].values:
    #         print(f"ID {id_item} não encontrado na base de dados.")
    #         return None
        
    #     df_reset = self.df_text.reset_index()
    #     df_reset[self.id_col] = df_reset[self.id_col].astype(str)  # Garante que o ID está correto

    #     # Construção do índice
    #     indice = pd.Series(df_reset.index, index=df_reset[self.id_col])
    #     idx = indice[id_item]

                # if id_item not in self.df_text[self.id_col].values:
                #     print(f"ID {id_item} não encontrado na base de dados.")
                #     return None
                
                # df_reset = self.df_text.reset_index()
                # indice = pd.Series(df_reset.index, index=df_reset[self.id_col].astype(str))
                # idx = indice[id_item]

                # print(f"Embeddings do item {id_item}: {self.embeddings[idx]}")
        # sim_score = util.cos_sim(self.embeddings[idx], self.embeddings).squeeze()
        # sim_score = sim_score.tolist()

        # sim_score = sorted(enumerate(sim_score), key=lambda x: x[1], reverse=True)[1:top_n+1]
        # sim_index = [i[0] for i in sim_score]

        # recomendacao = pd.DataFrame({
        #     #'Item Recomendado': self.df_text[self.item_name_col].iloc[sim_index],
        #     'Item Recomendado': self.df_text['nome_original'].iloc[sim_index],
        #     'Similaridade Cosseno': [score[1] for score in sim_score]
        # }).reset_index(drop=True)

        # # Adjust index to start at 1
        # recomendacao.index += 1

        # # Get original item name
        # item_name = self.df_text.loc[self.df_text[self.id_col] == id_item, 'nome_original'].values[0].title()

        # if recomendacao.empty:
        #     print("Nenhuma recomendação encontrada.")
        # else:
        #     print(f"As recomendações mais similares ao item '{item_name}' são:\n")
        #     #print(f"As recomendações mais similares ao item '{self.df_text.loc[idx, self.original_nome_item]}' são:\n")
        # return recomendacao
       
        