import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class MotorRecomendacaoFBC:
    def __init__(self, df_cartas, matriz_features):
        self.df = df_cartas.reset_index(drop=True)
        self.X = matriz_features
        # Pré-normalização L2 das linhas da matriz de cartas
        normas = np.linalg.norm(self.X, axis=1, keepdims=True)
        normas[normas == 0] = 1.0
        self.X_norm = self.X / normas

    def ranquear_cartas(self, vetor_perfil, top_n=5, excluir_nomes=None, filtro_raridade=None):
        """
        Calcula a similaridade de cosseno do perfil contra todo o catálogo e retorna o Top-N.
        
        :param vetor_perfil: Vetor 1D do perfil do jogador (R^d).
        :param top_n: Quantidade de recomendações a retornar.
        :param excluir_nomes: Lista de cartas que já estão na mão/deck para não duplicar.
        :param filtro_raridade: 'Comum', 'Raro', 'Épico', 'Lendário', 'Suporte' ou None.
        :return: DataFrame ordenado com as Top-N recomendações e scores de similaridade.
        """
        excluir_nomes = [nome.lower() for nome in (excluir_nomes or [])]
        
        # Garante normalização do vetor de consulta
        norma_u = np.linalg.norm(vetor_perfil)
        if norma_u == 0:
            raise ValueError("O vetor de perfil do usuário não pode ser nulo.")
        u_norm = (vetor_perfil / norma_u).reshape(1, -1)

        # Similaridade matricial por Cosseno: shape (1, 33)
        scores = cosine_similarity(u_norm, self.X_norm)[0]

        # Criação de tabela de pontuação
        df_ranking = self.df.copy()
        df_ranking["similaridade"] = scores

        # Aplica filtros de exclusão e raridade
        if excluir_nomes:
            df_ranking = df_ranking[~df_ranking["nome"].str.lower().isin(excluir_nomes)]

        if filtro_raridade:
            df_ranking = df_ranking[df_ranking["raridade"].str.lower() == filtro_raridade.lower()]

        # Ordenação decrescente por score de similaridade
        df_ranking = df_ranking.sort_values(by="similaridade", ascending=False).reset_index(drop=True)

        colunas_exibicao = ["nome", "raridade", "hp", "armadura", "dano", "efeito", "similaridade"]
        return df_ranking[colunas_exibicao].head(top_n)