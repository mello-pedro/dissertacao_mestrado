import pandas as pd
import re
import nltk 
from sqlalchemy.engine import create_engine
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, euclidean_distances, manhattan_distances
from tqdm import tqdm
pd.set_option('display.max_colwidth', None)

class TfidfRecommender:
    def __init__(self, DB_USER, 
                 DB_PASS, 
                 DB_NAME, 
                 DB_HOST, 
                 emb_cols, 
                 id_col: str, 
                 item_name_col: str, 
                 stopwords_extra=None):
        self.DB_USER = DB_USER
        self.DB_PASS = DB_PASS
        self.DB_HOST = DB_HOST
        self.DB_NAME = DB_NAME
        self.engine = create_engine(f'postgresql://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:5432/{self.DB_NAME}')

        ##cols que serao usadas p/ extrair texto p/ vetores de embeddings
        self.emb_cols = emb_cols
        self.id_col = id_col
        self.item_name_col = item_name_col
        self.stop_words = set(stopwords.words('portuguese'))

        # add stopwords personalizadas se fornecidas como argumento
        if stopwords_extra is not None:
            self.stop_words.update(stopwords_extra)
        
        #Inicialização do Tfidf
        self.tfidf_vectorizer = TfidfVectorizer()
        
        # Dados e manipulação dos vetores
        self.df_text = None
        self.embeddings = None
        self.cosine_sim = None
        self.euclidean_dist = None
        self.manhattan_dist = None
        self.indice = None
        self.tf_matrix = None
    
    ##func para carregar os dados (somente cols de interesse)
    def carrega_dados(self, tabela: str):
        try:
            #cols_str = ', '.join(self.emb_cols) + f', {self.item_name_col}, {self.id_col}, nome_curso AS nome_original'
            cols_str = ', '.join(self.emb_cols) + f', {self.item_name_col}, {self.id_col}, {self.item_name_col} AS nome_original'
            query = f'SELECT {cols_str} FROM {tabela}'
            self.df_text = pd.read_sql(query, self.engine)
            # coluna para armazenar os nomes dos cursos originais(util na hora de puxar recomends)
            print("Dados textuais carregados com sucesso \n")
        except Exception as e:
            print(f"Erro ao carregar os dados! {e}")

    def _limpar_e_compilar_texto(self):
        """Limpeza e preparação dos textos das colunas especificadas para embeddings."""

        # tradução para remov acent
        def traducao():
            origem = 'ãÃâÂáÁàÀêÊéÉèÈîÎíÍìÌïÏõÕôÔóÓòÒûÛúÚùÙüÜçÇñÑ/\'-"<>,.?!'
            destino = 'aaaaaaaaeeeeeeiiiiiiiioooooooouuuuuuuuccnn          '
            return str.maketrans(origem, destino)
            
        # limpeza de texto COM remoção de stopwords (nao inicializa-las junto com tfidf)
        def clean_text(data):
            if isinstance(data, str):
                # Remove acentuação e converte para minúsculas
                data = data.translate(traducao()).lower()
                # Remove stopwords
                data = ' '.join([word for word in data.split() if word not in self.stop_words])
                # Remove pontuação extra
                data = re.sub(r'[;,.]', '', data)
            return data
        
        #Limpeza adicional e compilação de colunas para vetorização
        all_cols_to_clean = self.emb_cols + [self.item_name_col]
        for col in all_cols_to_clean:
            if col in self.df_text.columns:
                self.df_text[col] = self.df_text[col].apply(clean_text)

        # cria a coluna compilada para vetorização
        self.df_text['compilado_textual'] = self.df_text[all_cols_to_clean].fillna('').agg(' '.join, axis=1)
        

    def gerar_embeddings(self):
        """Gera embeddings de TF-IDF e calcula as similaridades com uma barra de progresso."""
        
        self._limpar_e_compilar_texto()
        
        # Adiciona a barra de progresso para a geração da matriz TF-IDF
        with tqdm(total=1, desc="Gerando TF-IDF embeddings") as pbar:
            # self.tfidf_vectorizer.set_params(stop_words=set(list(self.portuguese_stopwords)))
            self.tf_matrix = self.tfidf_vectorizer.fit_transform(self.df_text['compilado_textual'])
            pbar.update(1)
        
        # print(f"Shape da matriz TF-IDF: {self.tf_matrix.shape}")
        # print("🔍 Trecho da matriz TF-IDF:")
        # print(self.tf_matrix)
        
        # Adiciona a barra de progresso para o cálculo da similaridade do cosseno
        with tqdm(total=1, desc="Calculando similaridade do cosseno") as pbar:
            self.cosine_sim = linear_kernel(self.tf_matrix, self.tf_matrix)
            pbar.update(1)

        ##ATIVAR SOMENTE QND USAR FUNCAO QUE COMPARA RECOMENDS DAS 3 DISTANCIAS ABAIXO!!
        # with tqdm(total=1, desc="Calculando distância Euclidiana") as pbar:
        #     self.euclidean_dist = euclidean_distances(self.tf_matrix, self.tf_matrix)
        #     pbar.update(1)
        
        # with tqdm(total=1, desc="Calculando distância Manhattan") as pbar:
        #     self.manhattan_dist = manhattan_distances(self.tf_matrix, self.tf_matrix)
        #     pbar.update(1)
        ##ATIVAR SOMENTE QND USAR FUNCAO QUE COMPARA RECOMENDS DAS 3 DISTANCIAS ABAIXO!!
        
        # print(f"Shape da matriz de similaridade do cosseno: {self.cosine_sim.shape}")
        # print("🔍 Trecho da matriz de similaridade do cosseno:")
        # print(self.cosine_sim)

        self.indice = pd.Series(self.df_text.index, index=self.df_text[self.id_col])
        print("Embeddings TF-IDF gerados com sucesso!\n")
    
    
    ### FUNCIONA, PORÉM ATIVAR SOMENTE QND FOR COMPARAR 3 MEDIDAS DE SIMILARIDADE
    # def recomendar(self, id_item, top_n=3):
    #     if self.cosine_sim is None or self.euclidean_dist is None:
    #         raise ValueError("A similaridade ainda não foi calculada. Certifique-se de que os embeddings foram gerados.")
        
    #     idx = self.indice[id_item]
    #     item_name = self.df_text.loc[self.df_text[self.id_col] == id_item, self.item_name_col].values[0].title()
        
    #     # Similaridade de Cosseno
    #     sim_cosine = list(enumerate(self.cosine_sim[idx]))
    #     sim_cosine = sorted(sim_cosine, key=lambda x: x[1], reverse=True)[1:top_n+1]  # Remove o próprio item
        
    #     # Dist Euclidiana
    #     sim_euclidean = list(enumerate(self.euclidean_dist[idx]))  # Usa a distância invertida
    #     sim_euclidean = sorted(sim_euclidean, key=lambda x: x[1])[1:top_n+1]  # Remove o próprio item

    #     # Dist Manhattan
    #     sim_manhattan = list(enumerate(self.manhattan_dist[idx]))
    #     sim_manhattan = sorted(sim_manhattan, key=lambda x: x[1])[1:top_n+1]

    #     # Criando DataFrame de recomendações para ambas as métricas
    #     recomendacao_cos = pd.DataFrame({
    #         'Item Recomendado': self.df_text[self.item_name_col].iloc[[i[0] for i in sim_cosine]],
    #         'Similaridade_Cosseno': [i[1] for i in sim_cosine]
    #     }).reset_index(drop=True)
        
    #     recomendacao_eucl = pd.DataFrame({
    #         'Item Recomendado': self.df_text[self.item_name_col].iloc[[i[0] for i in sim_euclidean]],
    #         'Distancia_Euclidiana': [i[1] for i in sim_euclidean]
    #     }).reset_index(drop=True)

    #     recomendacao_man = pd.DataFrame({
    #         'Item Recomendado': self.df_text[self.item_name_col].iloc[[i[0] for i in sim_manhattan]],
    #         'Distancia_Manhattan': [i[1] for i in sim_manhattan]
    #     }).reset_index(drop=True)

    #     # Exibindo as recomendações separadas por métrica
    #     print(f"As recomendações mais similares ao item '{item_name}' são:\n")
        
    #     print("\n **Baseado em Similaridade do Cosseno:**")
    #     print(recomendacao_cos)

    #     print("\n Baseado em Distância Euclidiana:")
    #     print(recomendacao_eucl)

    #     print("\n Baseado em Distância Manhattan:")
    #     print(recomendacao_man)


    # ##funciona mas vou tentar v2 acima juntando eucliden + cosine. NAO USAR!!
    # def recomendar(self, id_item, top_n=3, similarity_metric='cosine'):
    #     if self.cosine_sim is None or self.euclidean_dist is None:
    #         raise ValueError("A similaridade ainda não foi calculada. Certifique-se de que os embeddings foram gerados.")
        
    #     idx = self.indice[id_item]
    #     item_name = self.df_text.loc[self.df_text[self.id_col] == id_item, self.item_name_col].values[0].title()
        
    #     if similarity_metric == 'cosine':
    #         sim_score = list(enumerate(self.cosine_sim[idx]))
    #     elif similarity_metric == 'euclidean':
    #         sim_score = list(enumerate(-self.euclidean_dist[idx]))  # Usa -distância para inverter a ordem
    #     else:
    #         raise ValueError("Métrica inválida! Escolha entre 'cosine' ou 'euclidean'.")
        
    #     sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)[1:top_n+1]
    #     sim_index = [i[0] for i in sim_score]
        
    #     recomendacoes = pd.DataFrame({
    #         'Item Recomendado': self.df_text[self.item_name_col].iloc[sim_index],
    #         'Similaridade': [score[1] for score in sim_score]
    #     }).reset_index(drop=True)
        
    #     print(f"As recomendações mais similares ao item '{item_name}' usando {similarity_metric} são:\n")
    #     return print(recomendacoes)
    
    ##FUNCIONA PERFEITAMENTE. VOU ADAPTAR APENAS PARA ACOMODAR DIST EUCLIDEANA
    def recomendar(self, id_item, top_n=3):
        """Recomenda itens similares com base no ID do item de entrada."""
        if self.cosine_sim is None:
            raise ValueError("A similaridade ainda não foi calculada. Certifique-se de que os embeddings foram gerados.")
        
        idx = self.indice[id_item]
        item_name = self.df_text.loc[self.df_text[self.id_col] == id_item, 'nome_original'].values[0].title()
        #item_name = self.df_text.loc[self.df_text[self.id_col] == id_item, self.item_name_col].values[0].title()
        
        # Calcular scores de similaridade
        sim_score = list(enumerate(self.cosine_sim[idx]))
        sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)[1:top_n+1]
        sim_index = [i[0] for i in sim_score]
        
        recomendacoes = pd.DataFrame({
            'Item Recomendado': self.df_text['nome_original'].iloc[sim_index],
            # 'Item Recomendado': self.df_text[self.item_name_col].iloc[sim_index],
            'Similaridade Cosseno': [score[1] for score in sim_score]
        }).reset_index(drop=True)
        
        recomendacoes.index += 1
        print(f"As recomendações mais similares ao item '{item_name}' são:\n")
        return recomendacoes
        #return print(recomendacoes)