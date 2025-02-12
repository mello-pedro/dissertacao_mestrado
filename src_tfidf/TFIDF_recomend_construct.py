import pandas as pd
import re
import nltk 
from sqlalchemy.engine import create_engine
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

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
        self.indice = None
        self.tf_matrix = None
    
    ##func para carregar os dados (somente cols de interesse)
    def carrega_dados(self, tabela: str):
        try:
            cols_str = ', '.join(self.emb_cols) + f', {self.item_name_col}, {self.id_col}'
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
        """Gera embeddings de TF-IDF e calcula a similaridade com uma barra de progresso."""
        
        self._limpar_e_compilar_texto()
        
        # Adiciona a barra de progresso para a geração da matriz TF-IDF
        with tqdm(total=1, desc="Gerando TF-IDF embeddings") as pbar:
            # self.tfidf_vectorizer.set_params(stop_words=set(list(self.portuguese_stopwords)))
            self.tf_matrix = self.tfidf_vectorizer.fit_transform(self.df_text['compilado_textual'])
            pbar.update(1)
        
        print(f"Shape da matriz TF-IDF: {self.tf_matrix.shape}")
        print("🔍 Trecho da matriz TF-IDF:")
        print(self.tf_matrix)
        
        # Adiciona a barra de progresso para o cálculo da similaridade do cosseno
        with tqdm(total=1, desc="Calculando similaridade do cosseno") as pbar:
            self.cosine_sim = linear_kernel(self.tf_matrix, self.tf_matrix)
            pbar.update(1)
        
        print(f"Shape da matriz de similaridade do cosseno: {self.cosine_sim.shape}")
        print("🔍 Trecho da matriz de similaridade do cosseno:")
        print(self.cosine_sim)

        self.indice = pd.Series(self.df_text.index, index=self.df_text[self.id_col])
        print("Embeddings TF-IDF gerados com sucesso!\n")
    
    def recomendar(self, id_item, top_n=3):
        """Recomenda itens similares com base no ID do item de entrada."""
        if self.cosine_sim is None:
            raise ValueError("A similaridade ainda não foi calculada. Certifique-se de que os embeddings foram gerados.")
        
        idx = self.indice[id_item]
        item_name = self.df_text.loc[self.df_text[self.id_col] == id_item, self.item_name_col].values[0].title()
        
        # Calcular scores de similaridade
        sim_score = list(enumerate(self.cosine_sim[idx]))
        sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)[1:top_n+1]
        sim_index = [i[0] for i in sim_score]
        
        recomendacoes = pd.DataFrame({
            'Item Recomendado': self.df_text[self.item_name_col].iloc[sim_index],
            'Similaridade': [score[1] for score in sim_score]
        }).reset_index(drop=True)
        
        print(f"As recomendações mais similares ao item '{item_name}' são:\n")
        return print(recomendacoes)