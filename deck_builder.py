import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class MontadorDeckColapseTale:
    def __init__(self, df_cartas, matriz_features):
        self.df = df_cartas.reset_index(drop=True)
        self.X = matriz_features
        # Vetores normalizados para cálculo rápido de similaridade
        normas = np.linalg.norm(self.X, axis=1, keepdims=True)
        normas[normas == 0] = 1.0
        self.X_norm = self.X / normas

    def montar_deck_balanceado(
        self,
        vetor_perfil,
        cartas_iniciais=None,
        tamanho_total=6,
        max_lendarias=1,
        max_epicas=2,
        qtd_suporte=1,
        lambda_div=0.70
    ):
        """
        Monta um deck completo aplicando restrições de cotas e Maximal Marginal Relevance (MMR).
        """
        cartas_iniciais = cartas_iniciais or []
        deck_indices = []

        # 1. Carrega as cartas iniciais já escolhidas pelo jogador
        for nome in cartas_iniciais:
            idx = self.df.index[self.df["nome"].str.lower() == nome.lower()].tolist()
            if idx:
                deck_indices.append(idx[0])

        # Normaliza vetor do perfil do jogador
        u_norm = (vetor_perfil / np.linalg.norm(vetor_perfil)).reshape(1, -1)
        sim_perfil = cosine_similarity(u_norm, self.X_norm)[0]

        # 2. Loop iterativo até completar o tamanho do deck
        while len(deck_indices) < tamanho_total:
            candidatos_validos = []
            
            # Contadores atuais de cotas
            raridades_no_deck = self.df.iloc[deck_indices]["raridade"].tolist()
            qtd_lendarias_atuais = raridades_no_deck.count("Lendário")
            qtd_epicas_atuais = raridades_no_deck.count("Épico")
            qtd_suporte_atuais = raridades_no_deck.count("Suporte")

            precisa_suporte = (tamanho_total - len(deck_indices)) <= (qtd_suporte - qtd_suporte_atuais)

            for i in range(len(self.df)):
                if i in deck_indices:
                    continue  # Já está no deck

                carta = self.df.iloc[i]
                raridade = carta["raridade"]

                # Validação de regras de suporte
                if raridade == "Suporte" and qtd_suporte_atuais >= qtd_suporte:
                    continue
                if precisa_suporte and raridade != "Suporte":
                    continue

                # Validação de tetos de raridade
                if raridade == "Lendário" and qtd_lendarias_atuais >= max_lendarias:
                    continue
                if raridade == "Épico" and qtd_epicas_atuais >= max_epicas:
                    continue

                candidatos_validos.append(i)

            if not candidatos_validos:
                break  # Não há mais candidatos respeitando as cotas

            # 3. Cálculo do Score MMR
            melhor_score = -np.inf
            melhor_candidato = None

            for c in candidatos_validos:
                rel = sim_perfil[c]
                
                if deck_indices:
                    # Máxima similaridade entre o candidato c e as cartas já selecionadas no deck
                    sim_com_deck = cosine_similarity(self.X_norm[c].reshape(1, -1), self.X_norm[deck_indices])[0]
                    redundancia = np.max(sim_com_deck)
                else:
                    redundancia = 0.0

                score_mmr = (lambda_div * rel) - ((1.0 - lambda_div) * redundancia)

                if score_mmr > melhor_score:
                    melhor_score = score_mmr
                    melhor_candidato = c

            deck_indices.append(melhor_candidato)

        # Montagem do DataFrame final com métricas do deck
        df_deck = self.df.iloc[deck_indices].copy().reset_index(drop=True)
        df_deck["sim_perfil"] = sim_perfil[deck_indices]
        
        return df_deck[["nome", "raridade", "hp", "armadura", "dano", "efeito", "sim_perfil"]]