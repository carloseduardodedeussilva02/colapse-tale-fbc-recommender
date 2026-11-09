# Colapse!Tale — Sistema de Recomendação Baseado em Conteúdo (FBC) & Deck Builder

> **Trabalho Prático 2 — Disciplina de Organização e Disseminação de Dados (ODS)**  
> **Tema:** Filtragem Baseada em Conteúdo (Content-Based Filtering — FBC) aplicada à composição estratégica de baralhos no jogo *Colapse!Tale*.

---

## 📌 Sumário Executivo

Este projeto implementa um pipeline completo de **Filtragem Baseada em Conteúdo (FBC)** para o jogo de cartas tático *Colapse!Tale* (composto por 33 cartas originais). 

Diferente da Filtragem Colaborativa (abordada no Trabalho 1), a FBC permite contornar integralmente o problema de **Cold-Start** (novos jogadores sem histórico ou novas cartas recém-adicionadas ao catálogo), operando sobre os atributos intrínsecos de cada carta:
1. **Atributos Numéricos:** Pontos de Vida (HP), Armadura (ARM) e Dano (ATK), escalonados linearmente via `MinMaxScaler`.
2. **Atributos Categóricos:** Níveis de Raridade (`Comum`, `Raro`, `Épico`, `Lendário`, `Suporte`) codificados via `OneHotEncoder`.
3. **Mecânicas e Habilidades Textuais:** Processamento de Linguagem Natural (PLN) aplicado ao texto das Ultimates via `TfidfVectorizer` (capturando termos-chave como *imobiliza*, *restaura*, *armadura*, *renasce*, etc.).
4. **Flags de Ação:** Atuação fora de turno e identificação de cartas de Suporte de uso único.

Para superar a superespecialização e o afunilamento tático (*filter bubble*), o sistema incorpora o algoritmo **Maximal Marginal Relevance (MMR)** combinado com **restrições duras de baralho (Hard Constraints)**, garantindo baralhos competitivos de exatamente 6 cartas com sinergia, diversidade e conformidade com as regras do jogo.

---

## 🏛️ Arquitetura do Sistema

```
colapse-tale-fbc/
├── app.py                  # API REST em Flask e motor vetorial integrado
├── requirements.txt        # Dependências do ecossistema Python
├── README.md               # Documentação técnica e guia de execução
├── relatorio_fbc.html      # Relatório formal acadêmico em padrão A4
└── templates/
    └── index.html          # Front-end responsivo (SPA) Dark Mode
```

### Fluxo de Dados e Pipeline Matemático

```
[ Catálogo 33 Cartas ] ──> [ MinMax + OneHot + TF-IDF ] ──> [ Matriz X ] ──> [ Normalização L2: X_norm ]
                                                                                   │
[ Seleção Manual ] + [ Sliders Táticos ] ──> [ Vetor Perfil Jogador: u ] ─────────┤
                                                                                   ▼
                                                                       [ Similaridade por Cosseno ]
                                                                                   │
                                                                                   ▼
                                                                       [ Otimização MMR (λ=0.70) ]
                                                                       + [ Filtro de Cotas (1L, 2E, 1S) ]
                                                                                   │
                                                                                   ▼
                                                                       [ Deck Otimizado (6 Cartas) ]
```

---

## 📐 Formulação Matemática

### 1. Espaço Vetorial Unificado
Cada carta $c_i$ ($i \in \{1, \dots, 33\}$) é projetada em um vetor $v(c_i) \in \mathbb{R}^D$:

$$v(c_i) = \big[\, v_{\text{num}}(c_i) \;\|\; v_{\text{cat}}(c_i) \;\|\; v_{\text{txt}}(c_i) \;\|\; v_{\text{flag}}(c_i) \,\big]$$

Para eliminar distorções de magnitude, todos os vetores são normalizados pela norma euclidiana:

$$\hat{x}_i = \frac{v(c_i)}{\|v(c_i)\|_2}$$

### 2. Vetor de Perfil do Jogador ($\vec{u}$)
Dado um conjunto de cartas-base $S_{\text{base}}$ escolhidas pelo usuário e preferências táticas contínuas (Ataque, Defesa, Controle, Suporte):

$$\vec{u} = \text{Norm}_{L2} \left( \sum_{k \in S_{\text{base}}} w_k \hat{x}_k + \vec{u}_{\text{tático}} \right)$$

### 3. Similaridade por Cosseno
A aderência da carta $c$ ao perfil $\vec{u}$ corresponde ao produto interno direto entre vetores unitários:

$$\text{Sim}(\vec{u}, c) = \vec{u} \cdot \hat{x}_c$$

### 4. Maximal Marginal Relevance (MMR)
Para selecionar iterativamente a próxima carta a compor o deck $S$ até $|S| = 6$:

$$c^* = \arg\max_{c \in C \setminus S} \left[ \lambda \cdot \text{Sim}(\vec{u}, c) - (1 - \lambda) \cdot \max_{s \in S} \text{Sim}(\hat{x}_c, \hat{x}_s) \right]$$

* **$\lambda = 0.70$ (padrão):** Prioriza a aderência ao estilo do jogador sem permitir duplicação de funções redundantes.
* **Restrições de Cotas Aplicadas no Espaço de Busca:**
  * Máximo de **1 Lendária** por baralho.
  * Máximo de **2 Épicas** por baralho.
  * Exatamente **1 Suporte de uso único** obrigatório por baralho.

---

## 🚀 Instalação e Execução

### Pré-requisitos
* Python 3.10, 3.11 ou 3.12 (Recomendado ambiente Linux / Ubuntu).
* Gerenciador de pacotes `pip`.

### 1. Clonar o Repositório
```bash
git clone https://github.com/seu-usuario/colapse-tale-fbc.git
cd colapse-tale-fbc
```

### 2. Configurar o Ambiente Virtual (Recomendado no Ubuntu/Debian)
```bash
# Criação do ambiente virtual
python3 -m venv venv

# Ativação do ambiente
source venv/bin/activate
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4. Executar a Aplicação Flask
```bash
python3 app.py
```

O servidor iniciará localmente e estará acessível em:
```text
http://127.0.0.1:5000
```
*(ou pelo endereço de rede local exibido no terminal, ex: `http://192.168.100.13:5000`)*.

---

## 📡 Documentação dos Endpoints REST

### `GET /`
* **Descrição:** Serve a interface web interativa (SPA) em HTML5/JavaScript.

---

### `GET /api/cartas`
* **Descrição:** Retorna o catálogo estruturado com as 33 cartas do jogo.
* **Parâmetros de Query (Opcionais):** `raridade` (ex: `GET /api/cartas?raridade=Lendário`).
* **Resposta de Exemplo (200 OK):**
```json
[
  {
    "id": 11,
    "nome": "Carlos",
    "raridade": "Raro",
    "hp": 900,
    "armadura": 150,
    "dano": 200,
    "fora_de_turno": 1,
    "suporte_uso_unico": 0,
    "efeito": "apos cair renasce com 50% hp e armadura"
  }
]
```

---

### `POST /api/perfil`
* **Descrição:** Inspeciona a assinatura vetorial gerada pelas cartas e diretrizes táticas.
* **Payload:**
```json
{
  "cartas": ["Carlos"],
  "pesos": [1.0],
  "arquetipo": {
    "defesa": 0.8,
    "ataque": 0.2,
    "controle": 0.0,
    "suporte": 0.3
  }
}
```
* **Resposta de Exemplo (200 OK):**
```json
{
  "dimensao_total": 45,
  "top_atributos": [
    { "dimensao": "armadura", "peso": 0.4125 },
    { "dimensao": "hp", "peso": 0.3950 },
    { "dimensao": "dano", "peso": 0.2210 },
    { "dimensao": "restaura", "peso": 0.1840 }
  ]
}
```

---

### `POST /api/recomendar`
* **Descrição:** Gera uma recomendação Top-N por cosseno puro ou monta um deck balanceado completo (6 cartas) com MMR e cotas.
* **Payload:**
```json
{
  "cartas_base": ["Carlos"],
  "arquetipo": { "defesa": 0.8, "ataque": 0.2 },
  "modo": "deck_balanceado",
  "tamanho_deck": 6,
  "lambda_div": 0.70
}
```
* **Resposta de Exemplo (200 OK):**
```json
{
  "modo": "deck_balanceado",
  "total_cartas": 6,
  "lambda_diversidade": 0.70,
  "deck": [
    { "nome": "Carlos", "raridade": "Raro", "similaridade": 1.0 },
    { "nome": "SintetiLáis", "raridade": "Épico", "similaridade": 0.8421 },
    { "nome": "Sara", "raridade": "Comum", "similaridade": 0.7610 },
    { "nome": "Dream Sans", "raridade": "Lendário", "similaridade": 0.7105 },
    { "nome": "Dream Beatriz", "raridade": "Épico", "similaridade": 0.6940 },
    { "nome": "Camila", "raridade": "Suporte", "similaridade": 0.5823 }
  ]
}
```

---

## 📊 Validação Experimental (Perfis Sintéticos)

O motor FBC foi submetido a testes empíricos com 4 perfis sintéticos de jogadores:

| Perfil Sintético | Entrada / Foco Tático | Composição Final Gerada (6 Cartas) | Afinidade Média | Cotas Válidas? |
| :--- | :--- | :--- | :---: | :---: |
| **P1: Guardião / Defesa** | Carlos (Base) + Defesa (0.9) | Carlos (R), SintetiLáis (Ép), Sara (C), Dream Sans (L), Dream Beatriz (Ép), **Camila (Sup)** | **69,82%** | ✅ Sim (1L, 2E, 1S) |
| **P2: Agressivo / Burst** | Ana, Peter (Base) + Dano (0.9) | Ana (C), Peter (C), Raul (C), SintetiLáis (Ép), Nightmare Sans (L), **Felipe (Sup)** | **68,45%** | ✅ Sim (1L, 1E, 1S) |
| **P3: Controle / Debuff** | Igor (Base) + Controle (0.9) | Igor (C), Betty Noire (Ép), Sofia (R), Error Sans (L), Adriano (R), **RevengeLáis (Sup)** | **64,10%** | ✅ Sim (1L, 1E, 1S) |
| **P4: Suporte / Cura** | Dream Beatriz + Suporte (0.9) | Dream Beatriz (Ép), AnotherChara (Ép), Dream Sans (L), Vitor (R), Láis (R), **SintetiClara (Sup)** | **63,20%** | ✅ Sim (1L, 2E, 1S) |

* **Cobertura de Catálogo:** **60,61%** (20 das 33 cartas do jogo foram exploradas na amostragem).
* **Conformidade de Cotas:** 100% de obediência às restrições de raridade e suporte único.

---

## 👥 Autores & Créditos

* **Trabalho Desenvolvido para a Disciplina de ODS (Oficina de Desenvolvimento de Sistemas)**.
* **Jogo Base:** *Colapse!Tale* Game Card.