import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Base consolidada baseada na Tabela Oficial
cartas_dataset = [
    # --- COMUNS ---
    {"nome": "Sara", "raridade": "Comum", "hp": 800, "armadura": 100, "dano": 150, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "reduz armadura de 1 adversario em 10%"},
    {"nome": "Alexandre", "raridade": "Comum", "hp": 1000, "armadura": 200, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta a armadura de 1 aliado em 10%"},
    {"nome": "Ana", "raridade": "Comum", "hp": 600, "armadura": 100, "dano": 200, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta o proprio dano em 10%"},
    {"nome": "Igor", "raridade": "Comum", "hp": 600, "armadura": 150, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "imobiliza 1 adversario por 1 rodada"},
    {"nome": "Alan", "raridade": "Comum", "hp": 1200, "armadura": 200, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta a propria armadura em 10%"},
    {"nome": "Marcos", "raridade": "Comum", "hp": 600, "armadura": 100, "dano": 100, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "adiciona 1 carta comum a sua mao"},
    {"nome": "João", "raridade": "Comum", "hp": 700, "armadura": 150, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "protege a si proprio por 1 rodada"},
    {"nome": "Raul", "raridade": "Comum", "hp": 200, "armadura": 400, "dano": 200, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta armadura em 50% e absorve dano"},
    {"nome": "Juliana", "raridade": "Comum", "hp": 800, "armadura": 100, "dano": 150, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "protege 1 aliado por 1 rodada"},
    {"nome": "Peter", "raridade": "Comum", "hp": 600, "armadura": 100, "dano": 200, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "cede 50% do seu dano a 1 aliado"},

    # --- RAROS ---
    {"nome": "Carlos", "raridade": "Raro", "hp": 900, "armadura": 150, "dano": 200, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "apos cair renasce com 50% hp e armadura"},
    {"nome": "Láis", "raridade": "Raro", "hp": 1500, "armadura": 200, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta sua armadura mais 1 aliado em 10%"},
    {"nome": "Sofia", "raridade": "Raro", "hp": 700, "armadura": 150, "dano": 150, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "imobiliza adversario com menor vida por 1 rodada"},
    {"nome": "Adriano", "raridade": "Raro", "hp": 800, "armadura": 100, "dano": 150, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "silencia impede ultimate de 1 adversario por 1 rodada"},
    {"nome": "Vitor", "raridade": "Raro", "hp": 600, "armadura": 200, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta todos os hps dos aliados em 10%"},
    {"nome": "Beatriz", "raridade": "Raro", "hp": 100, "armadura": 200, "dano": 150, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "troca de lugar com aliado restaurando hp"},
    {"nome": "Alice", "raridade": "Raro", "hp": 700, "armadura": 200, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "reduz o dano de 1 carta inimiga em 10%"},
    {"nome": "Vitória", "raridade": "Raro", "hp": 1000, "armadura": 200, "dano": 100, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "resiste apos ser finalizada por 1 rodada"},

    # --- ÉPICOS ---
    {"nome": "SintetiLáis", "raridade": "Épico", "hp": 2200, "armadura": 300, "dano": 150, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta armadura mais 2 aliados em 20%"},
    {"nome": "AnotherChara", "raridade": "Épico", "hp": 1200, "armadura": 100, "dano": 200, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "protege a si proprio mais 1 aliado por 1 rodada"},
    {"nome": "Betty Noire", "raridade": "Épico", "hp": 800, "armadura": 150, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "imobiliza 2 adversarios por 1 rodada"},
    {"nome": "Lyra", "raridade": "Épico", "hp": 800, "armadura": 100, "dano": 100, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "adiciona 1 carta epica a sua mao"},
    {"nome": "Ink Carlos", "raridade": "Épico", "hp": 900, "armadura": 200, "dano": 200, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "apos cair renasce com amplificacao de dano de 10%"},
    {"nome": "Dream Beatriz", "raridade": "Épico", "hp": 900, "armadura": 100, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "restaura o hp de um aliado com hp inferior a 50%"},

    # --- LENDÁRIOS ---
    {"nome": "Ink Sans", "raridade": "Lendário", "hp": 800, "armadura": 100, "dano": 150, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "adiciona 1 carta de qualquer raridade a sua mao para jogar na mesa"},
    {"nome": "Error Sans", "raridade": "Lendário", "hp": 800, "armadura": 150, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "reduz dano e armadura de 2 adversarios em 10%"},
    {"nome": "Dream Sans", "raridade": "Lendário", "hp": 900, "armadura": 100, "dano": 300, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "restaura o hp de todos os aliados"},
    {"nome": "Nightmare Sans", "raridade": "Lendário", "hp": 800, "armadura": 200, "dano": 300, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "imobiliza 2 adversarios por 2 rodadas"},

    # --- SUPORTES (Uso Único) ---
    {"nome": "Felipe", "raridade": "Suporte", "hp": 0, "armadura": 0, "dano": 0, "fora_de_turno": 0, "suporte_uso_unico": 1, "efeito": "aumenta o dano de 1 aliado em 10%"},
    {"nome": "Camila", "raridade": "Suporte", "hp": 0, "armadura": 0, "dano": 0, "fora_de_turno": 0, "suporte_uso_unico": 1, "efeito": "restaura o hp de 1 aliado"},
    {"nome": "SintetiClara", "raridade": "Suporte", "hp": 0, "armadura": 0, "dano": 0, "fora_de_turno": 0, "suporte_uso_unico": 1, "efeito": "dobra os valores da ultimate de 1 aliado"},
    {"nome": "Cross Sans", "raridade": "Suporte", "hp": 0, "armadura": 0, "dano": 0, "fora_de_turno": 0, "suporte_uso_unico": 1, "efeito": "altera os valores de 1 carta aleatoria entre 100 e 200"},
    {"nome": "RevengeLáis", "raridade": "Suporte", "hp": 0, "armadura": 0, "dano": 0, "fora_de_turno": 0, "suporte_uso_unico": 1, "efeito": "ressuscita 1 aliado"}
]

class PreprocessadorColapseTale:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.vectorizer = TfidfVectorizer()
        
    def fit_transform(self, df):
        # 1. Normalização dos atributos numéricos
        num_cols = ["hp", "armadura", "dano"]
        num_matrix = self.scaler.fit_transform(df[num_cols])
        
        # 2. One-Hot Encoding da Raridade
        raridade_matrix = self.encoder.fit_transform(df[["raridade"]])
        
        # 3. TF-IDF das mecânicas das Ultimates
        tfidf_matrix = self.vectorizer.fit_transform(df["efeito"]).toarray()
        
        # 4. Flags binárias (Fora de Turno e Suporte Uso Único)
        flags_matrix = df[["fora_de_turno", "suporte_uso_unico"]].to_numpy()
        
        # 5. Concatenação do espaço vetorial completo
        X = np.hstack([num_matrix, raridade_matrix, tfidf_matrix, flags_matrix])
        return X

# Instanciação e geração da matriz de features
df_cartas = pd.DataFrame(cartas_dataset)
preprocessador = PreprocessadorColapseTale()
matriz_features = preprocessador.fit_transform(df_cartas)

print(f"Dataset carregado com {df_cartas.shape[0]} cartas.")
print(f"Dimensão do espaço vetorial gerado: {matriz_features.shape[1]} dimensões por carta.")