import numpy as np
import pandas as pd

class PerfilJogadorFBC:
    def __init__(self, preprocessador, df_cartas, matriz_features):
        self.prep = preprocessador
        self.df = df_cartas.reset_index(drop=True)
        self.X = matriz_features
        self.feature_names = self._mapear_indices_features()

    def _mapear_indices_features(self):
        """Mapeia os nomes e índices de cada dimensão no espaço vetorial."""
        nomes = ["hp", "armadura", "dano"]
        
        # Categorias da raridade
        nomes.extend([f"raridade_{r}" for r in self.prep.encoder.categories_[0]])
        
        # Vocabulário TF-IDF
        nomes.extend(self.prep.vectorizer.get_feature_names_out())
        
        # Flags
        nomes.extend(["fora_de_turno", "suporte_uso_unico"])
        return nomes

    def criar_perfil_por_favoritos(self, escolhas_jogador):
        """
        Gera o vetor médio ponderado a partir de cartas que o jogador gosta/utiliza.
        escolhas_jogador: lista de tuplas [("NomeDaCarta", peso)]
        Exemplo: [("Carlos", 1.0), ("Ink Sans", 1.2), ("Alan", 0.8)]
        """
        vetores = []
        pesos = []

        for nome, peso in escolhas_jogador:
            idx = self.df.index[self.df["nome"].str.lower() == nome.lower()].tolist()
            if idx:
                vetores.append(self.X[idx[0]])
                pesos.append(peso)
            else:
                print(f"[Aviso] Carta '{nome}' não encontrada no dataset.")

        if not vetores:
            raise ValueError("Nenhuma carta válida foi fornecida para construir o perfil.")

        vetores = np.array(vetores)
        pesos = np.array(pesos).reshape(-1, 1)

        # Média ponderada
        vetor_perfil = np.sum(vetores * pesos, axis=0) / np.sum(pesos)
        
        # Normalização L2 para garantir magnitude unitária
        norma = np.linalg.norm(vetor_perfil)
        if norma > 0:
            vetor_perfil = vetor_perfil / norma
            
        return vetor_perfil

    def criar_perfil_por_arquetipo(self, foco_ataque=0.0, foco_defesa=0.0, foco_controle=0.0, foco_suporte=0.0):
        """
        Gera um perfil sintético a partir de diretrizes táticas desejadas (0.0 a 1.0).
        """
        u = np.zeros(self.X.shape[1])
        
        # Atributos numéricos normalizados
        # idx 0: hp, idx 1: armadura, idx 2: dano
        u[0] = foco_defesa
        u[1] = foco_defesa
        u[2] = foco_ataque

        # Sinergia de palavras-chave TF-IDF
        termos_controle = ["imobiliza", "stun", "silencia", "reduz"]
        termos_suporte = ["restaura", "cura", "protege", "adiciona", "dobra", "ressuscita"]

        for i, nome_feat in enumerate(self.feature_names):
            if any(termo in nome_feat for termo in termos_controle):
                u[i] += foco_controle
            if any(termo in nome_feat for termo in termos_suporte):
                u[i] += foco_suporte

        norma = np.linalg.norm(u)
        if norma > 0:
            u = u / norma
            
        return u

    def inspecionar_perfil(self, vetor_perfil, top_k=6):
        """Exibe as dimensões e características mais valorizadas pelo perfil."""
        indices = np.argsort(vetor_perfil)[::-1][:top_k]
        print("\n--- Assinatura Vetorial do Jogador (Top Dimensões) ---")
        for idx in indices:
            print(f" • {self.feature_names[idx]}: {vetor_perfil[idx]:.4f}")