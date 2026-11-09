import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

app = Flask(__name__)

# ==========================================
# 1. DATASET OFICIAL (COLAPSE!TALE GAME CARD)
# ==========================================
cartas_dataset = [
    # --- COMUNS ---
    {"id": 1, "nome": "Sara", "raridade": "Comum", "hp": 800, "armadura": 100, "dano": 150, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "reduz armadura de 1 adversario em 10%"},
    {"id": 2, "nome": "Alexandre", "raridade": "Comum", "hp": 1000, "armadura": 200, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta a armadura de 1 aliado em 10%"},
    {"id": 3, "nome": "Ana", "raridade": "Comum", "hp": 600, "armadura": 100, "dano": 200, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta o proprio dano em 10%"},
    {"id": 4, "nome": "Igor", "raridade": "Comum", "hp": 600, "armadura": 150, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "imobiliza 1 adversario por 1 rodada"},
    {"id": 5, "nome": "Alan", "raridade": "Comum", "hp": 1200, "armadura": 200, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta a propria armadura em 10%"},
    {"id": 6, "nome": "Marcos", "raridade": "Comum", "hp": 600, "armadura": 100, "dano": 100, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "adiciona 1 carta comum a sua mao"},
    {"id": 7, "nome": "João", "raridade": "Comum", "hp": 700, "armadura": 150, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "protege a si proprio por 1 rodada"},
    {"id": 8, "nome": "Raul", "raridade": "Comum", "hp": 200, "armadura": 400, "dano": 200, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta armadura em 50% e absorve dano"},
    {"id": 9, "nome": "Juliana", "raridade": "Comum", "hp": 800, "armadura": 100, "dano": 150, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "protege 1 aliado por 1 rodada"},
    {"id": 10, "nome": "Peter", "raridade": "Comum", "hp": 600, "armadura": 100, "dano": 200, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "cede 50% do seu dano a 1 aliado"},

    # --- RAROS ---
    {"id": 11, "nome": "Carlos", "raridade": "Raro", "hp": 900, "armadura": 150, "dano": 200, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "apos cair renasce com 50% hp e armadura"},
    {"id": 12, "nome": "Láis", "raridade": "Raro", "hp": 1500, "armadura": 200, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta sua armadura mais 1 aliado em 10%"},
    {"id": 13, "nome": "Sofia", "raridade": "Raro", "hp": 700, "armadura": 150, "dano": 150, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "imobiliza adversario com menor vida por 1 rodada"},
    {"id": 14, "nome": "Adriano", "raridade": "Raro", "hp": 800, "armadura": 100, "dano": 150, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "silencia impede ultimate de 1 adversario por 1 rodada"},
    {"id": 15, "nome": "Vitor", "raridade": "Raro", "hp": 600, "armadura": 200, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta todos os hps dos aliados em 10%"},
    {"id": 16, "nome": "Beatriz", "raridade": "Raro", "hp": 100, "armadura": 200, "dano": 150, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "troca de lugar com aliado restaurando hp"},
    {"id": 17, "nome": "Alice", "raridade": "Raro", "hp": 700, "armadura": 200, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "reduz o dano de 1 carta inimiga em 10%"},
    {"id": 18, "nome": "Vitória", "raridade": "Raro", "hp": 1000, "armadura": 200, "dano": 100, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "resiste apos ser finalizada por 1 rodada"},

    # --- ÉPICOS ---
    {"id": 19, "nome": "SintetiLáis", "raridade": "Épico", "hp": 2200, "armadura": 300, "dano": 150, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "aumenta armadura mais 2 aliados em 20%"},
    {"id": 20, "nome": "AnotherChara", "raridade": "Épico", "hp": 1200, "armadura": 100, "dano": 200, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "protege a si proprio mais 1 aliado por 1 rodada"},
    {"id": 21, "nome": "Betty Noire", "raridade": "Épico", "hp": 800, "armadura": 150, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "imobiliza 2 adversarios por 1 rodada"},
    {"id": 22, "nome": "Lyra", "raridade": "Épico", "hp": 800, "armadura": 100, "dano": 100, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "adiciona 1 carta epica a sua mao"},
    {"id": 23, "nome": "Ink Carlos", "raridade": "Épico", "hp": 900, "armadura": 200, "dano": 200, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "apos cair renasce com amplificacao de dano de 10%"},
    {"id": 24, "nome": "Dream Beatriz", "raridade": "Épico", "hp": 900, "armadura": 100, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "restaura o hp de um aliado com hp inferior a 50%"},

    # --- LENDÁRIOS ---
    {"id": 25, "nome": "Ink Sans", "raridade": "Lendário", "hp": 800, "armadura": 100, "dano": 150, "fora_de_turno": 1, "suporte_uso_unico": 0, "efeito": "adiciona 1 carta de qualquer raridade a sua mao para jogar na mesa"},
    {"id": 26, "nome": "Error Sans", "raridade": "Lendário", "hp": 800, "armadura": 150, "dano": 100, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "reduz dano e armadura de 2 adversarios em 10%"},
    {"id": 27, "nome": "Dream Sans", "raridade": "Lendário", "hp": 900, "armadura": 100, "dano": 300, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "restaura o hp de todos os aliados"},
    {"id": 28, "nome": "Nightmare Sans", "raridade": "Lendário", "hp": 800, "armadura": 200, "dano": 300, "fora_de_turno": 0, "suporte_uso_unico": 0, "efeito": "imobiliza 2 adversarios por 2 rodadas"},

    # --- SUPORTES (Uso Único) ---
    {"id": 29, "nome": "Felipe", "raridade": "Suporte", "hp": 0, "armadura": 0, "dano": 0, "fora_de_turno": 0, "suporte_uso_unico": 1, "efeito": "aumenta o dano de 1 aliado em 10%"},
    {"id": 30, "nome": "Camila", "raridade": "Suporte", "hp": 0, "armadura": 0, "dano": 0, "fora_de_turno": 0, "suporte_uso_unico": 1, "efeito": "restaura o hp de 1 aliado"},
    {"id": 31, "nome": "SintetiClara", "raridade": "Suporte", "hp": 0, "armadura": 0, "dano": 0, "fora_de_turno": 0, "suporte_uso_unico": 1, "efeito": "dobra os valores da ultimate de 1 aliado"},
    {"id": 32, "nome": "Cross Sans", "raridade": "Suporte", "hp": 0, "armadura": 0, "dano": 0, "fora_de_turno": 0, "suporte_uso_unico": 1, "efeito": "altera os valores de 1 carta aleatoria entre 100 e 200"},
    {"id": 33, "nome": "RevengeLáis", "raridade": "Suporte", "hp": 0, "armadura": 0, "dano": 0, "fora_de_turno": 0, "suporte_uso_unico": 1, "efeito": "ressuscita 1 aliado"}
]

df_cartas = pd.DataFrame(cartas_dataset)

# ==========================================
# 2. PRÉ-PROCESSAMENTO E ESPAÇO VETORIAL
# ==========================================
scaler = MinMaxScaler()
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
vectorizer = TfidfVectorizer()

num_matrix = scaler.fit_transform(df_cartas[["hp", "armadura", "dano"]])
raridade_matrix = encoder.fit_transform(df_cartas[["raridade"]])
tfidf_matrix = vectorizer.fit_transform(df_cartas["efeito"]).toarray()
flags_matrix = df_cartas[["fora_de_turno", "suporte_uso_unico"]].to_numpy()

X = np.hstack([num_matrix, raridade_matrix, tfidf_matrix, flags_matrix])

normas = np.linalg.norm(X, axis=1, keepdims=True)
normas[normas == 0] = 1.0
X_norm = X / normas

feature_names = ["hp", "armadura", "dano"]
feature_names.extend([f"raridade_{r}" for r in encoder.categories_[0]])
feature_names.extend(vectorizer.get_feature_names_out())
feature_names.extend(["fora_de_turno", "suporte_uso_unico"])

# ==========================================
# 3. FUNÇÕES AUXILIARES DE MODELAGEM
# ==========================================
def calcular_vetor_perfil(cartas_selecionadas, pesos=None, arquetipo=None):
    vetores = []
    pesos_lista = []

    if cartas_selecionadas:
        for idx_cart, nome in enumerate(cartas_selecionadas):
            p = pesos[idx_cart] if pesos and idx_cart < len(pesos) else 1.0
            loc = df_cartas.index[df_cartas["nome"].str.lower() == nome.lower()].tolist()
            if loc:
                vetores.append(X[loc[0]])
                pesos_lista.append(p)

    if vetores:
        vetores = np.array(vetores)
        pesos_arr = np.array(pesos_lista).reshape(-1, 1)
        u = np.sum(vetores * pesos_arr, axis=0) / np.sum(pesos_arr)
    else:
        u = np.zeros(X.shape[1])

    if arquetipo:
        atk = float(arquetipo.get("ataque", 0.0))
        defs = float(arquetipo.get("defesa", 0.0))
        ctrl = float(arquetipo.get("controle", 0.0))
        sup = float(arquetipo.get("suporte", 0.0))

        u[0] += defs
        u[1] += defs
        u[2] += atk

        termos_ctrl = ["imobiliza", "stun", "silencia", "reduz"]
        termos_sup = ["restaura", "cura", "protege", "adiciona", "dobra", "ressuscita"]

        for idx, feat in enumerate(feature_names):
            if any(t in feat for t in termos_ctrl):
                u[idx] += ctrl
            if any(t in feat for t in termos_sup):
                u[idx] += sup

    norma_u = np.linalg.norm(u)
    if norma_u > 0:
        u = u / norma_u
    return u

# ==========================================
# 4. ROTAS WEB & ENDPOINTS REST
# ==========================================

@app.route("/")
def index():
    """Serve a interface visual HTML do Colapse!Tale."""
    return render_template("index.html")


@app.route("/api/cartas", methods=["GET"])
def listar_cartas():
    raridade_filtro = request.args.get("raridade")
    df_resp = df_cartas
    if raridade_filtro:
        df_resp = df_resp[df_resp["raridade"].str.lower() == raridade_filtro.lower()]
    return jsonify(df_resp.to_dict(orient="records")), 200


@app.route("/api/perfil", methods=["POST"])
def obter_perfil():
    dados = request.get_json() or {}
    cartas = dados.get("cartas", [])
    pesos = dados.get("pesos", [])
    arquetipo = dados.get("arquetipo", {})

    u = calcular_vetor_perfil(cartas, pesos, arquetipo)
    if np.all(u == 0):
        return jsonify({"erro": "Nenhuma carta ou atributo válido para construir o perfil."}), 400

    top_dims_idx = np.argsort(u)[::-1][:6]
    assinatura = [
        {"dimensao": feature_names[i], "peso": float(round(u[i], 4))}
        for i in top_dims_idx if u[i] > 0
    ]

    return jsonify({"dimensao_total": len(u), "top_atributos": assinatura}), 200


@app.route("/api/recomendar", methods=["POST"])
def recomendar():
    dados = request.get_json() or {}
    cartas_base = dados.get("cartas_base", [])
    pesos = dados.get("pesos", [])
    arquetipo = dados.get("arquetipo", {})
    modo = dados.get("modo", "deck_balanceado")
    top_n = int(dados.get("top_n", 5))
    tamanho_deck = int(dados.get("tamanho_deck", 6))
    lambda_div = float(dados.get("lambda_div", 0.70))

    u = calcular_vetor_perfil(cartas_base, pesos, arquetipo)
    if np.all(u == 0):
        return jsonify({"erro": "Informe cartas base ou arquetipo para gerar recomendação."}), 400

    sim_perfil = cosine_similarity(u.reshape(1, -1), X_norm)[0]

    if modo == "top_n":
        nomes_base_lower = [c.lower() for c in cartas_base]
        candidatos = [i for i in range(len(df_cartas)) if df_cartas.iloc[i]["nome"].lower() not in nomes_base_lower]
        candidatos_ordenados = sorted(candidatos, key=lambda i: sim_perfil[i], reverse=True)[:top_n]
        
        resultado = []
        for i in candidatos_ordenados:
            c = df_cartas.iloc[i].to_dict()
            c["similaridade"] = float(round(sim_perfil[i], 4))
            resultado.append(c)

        return jsonify({"modo": "top_n", "recomendacoes": resultado}), 200

    deck_indices = []
    for nome in cartas_base:
        loc = df_cartas.index[df_cartas["nome"].str.lower() == nome.lower()].tolist()
        if loc and loc[0] not in deck_indices:
            deck_indices.append(loc[0])

    max_lendarias = int(dados.get("max_lendarias", 1))
    max_epicas = int(dados.get("max_epicas", 2))
    qtd_suporte = int(dados.get("qtd_suporte", 1))

    while len(deck_indices) < tamanho_deck:
        candidatos_validos = []
        raridades_atuais = df_cartas.iloc[deck_indices]["raridade"].tolist()
        n_lend = raridades_atuais.count("Lendário")
        n_epic = raridades_atuais.count("Épico")
        n_sup = raridades_atuais.count("Suporte")

        precisa_suporte = (tamanho_deck - len(deck_indices)) <= (qtd_suporte - n_sup)

        for i in range(len(df_cartas)):
            if i in deck_indices:
                continue

            r = df_cartas.iloc[i]["raridade"]
            if r == "Suporte" and n_sup >= qtd_suporte:
                continue
            if precisa_suporte and r != "Suporte":
                continue
            if r == "Lendário" and n_lend >= max_lendarias:
                continue
            if r == "Épico" and n_epic >= max_epicas:
                continue

            candidatos_validos.append(i)

        if not candidatos_validos:
            break

        melhor_score = -np.inf
        melhor_candidato = None

        for c in candidatos_validos:
            rel = sim_perfil[c]
            if deck_indices:
                sim_deck = cosine_similarity(X_norm[c].reshape(1, -1), X_norm[deck_indices])[0]
                redundancia = np.max(sim_deck)
            else:
                redundancia = 0.0

            score_mmr = (lambda_div * rel) - ((1.0 - lambda_div) * redundancia)
            if score_mmr > melhor_score:
                melhor_score = score_mmr
                melhor_candidato = c

        deck_indices.append(melhor_candidato)

    deck_final = []
    for idx in deck_indices:
        item = df_cartas.iloc[idx].to_dict()
        item["similaridade"] = float(round(sim_perfil[idx], 4))
        deck_final.append(item)

    return jsonify({
        "modo": "deck_balanceado",
        "total_cartas": len(deck_final),
        "lambda_diversidade": lambda_div,
        "deck": deck_final
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)