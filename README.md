# 🃏 Colapse!Tale — Jogo de Cartas & Sistema de Recomendação Baseado em Conteúdo (FBC)

Projeto prático desenvolvido para a disciplina **Oficina de Desenvolvimento de Sistemas I (ODS I)** da **Universidade do Estado do Amazonas (UEA)**. 

O sistema une o jogo de estratégia por turnos **Colapse!Tale** (com suporte a modo Solo contra Bots e Multijogador em tempo real via WebSockets) a um **Motor de Recomendação Baseado em Conteúdo (FBC)** com **Maximal Marginal Relevance (MMR)** para montagem tática e balanceada de baralhos.

🔗 **Demo Online (Deploy em Nuvem):** [https://colapse-tale-game-card.onrender.com](https://colapse-tale-game-card.onrender.com)  
*(Nota: Por utilizar instância gratuita no Render, o primeiro acesso após inatividade pode levar ~45 segundos para iniciar o servidor).*

---

## 🎯 Funcionalidades Principais

1. **Sistema de Recomendação Baseado em Conteúdo (FBC + MMR):**
   - Modelagem vetorial das **33 cartas** do catálogo combinando atributos numéricos normalizados (`HP`, `Armadura`, `Dano`), raridades (`One-Hot Encoding`) e mecânicas textuais (`TF-IDF`).
   - Cálculo de afinidade via **Similaridade por Cosseno** contra perfis táticos (*Ataque, Defesa, Controle e Suporte*).
   - Algoritmo **Maximal Marginal Relevance (MMR)** calibrado com $\lambda = 0.70$ (70% relevância tática / 30% penalização de redundância) respeitando regras competitivas de composição (6 cartas de combate + 1 Suporte, teto de 1 Lendária e 2 Épicas).

2. **Motor de Combate & Mecânica de Banco:**
   - Batalhas em até 4 vias com sistema de **Mesa (3 ativas)** e **Banco de Reservas (3 cartas)** com substituição automática ao cair.
   - Efeitos ativos de turno (*Ultimates*) e efeitos passivos/automáticos fora de turno (ex: renascimento de *Carlos* e *Ink Carlos*, troca tática de *Beatriz*, doação de dano de *Peter*).

3. **Renderização Procedural 3D -> 8-Bit:**
   - Geração dinâmica de avatares tridimensionais utilizando **Three.js**, convertidos em tempo de execução via `<canvas>` para sprites *pixel art* 8-bit com contornos e iluminação estilizada.

4. **Arquitetura Multijogador em Tempo Real (WebSockets):**
   - Comunicação bidirecional de baixa latência (`ws://` em rede local Wi-Fi e `wss://` em produção via HTTPS) utilizando `Flask-Sock` e `Gevent`.
   - Sistema de criação de salas por código de 4 caracteres com sincronização autoritativa pelo anfitrião (*Host*) e preenchimento automático de vagas vazias com Inteligência Artificial.

---

## 🛠️ Tecnologias Utilizadas

- **Back-end & Rede:** Python 3, Flask, Flask-Sock (WebSockets), Gunicorn + Gevent.
- **Ciência de Dados & Recomendação:** NumPy, Pandas, Scikit-Learn (`TfidfVectorizer`, `MinMaxScaler`, `OneHotEncoder`, `cosine_similarity`).
- **Front-end & Gráficos:** HTML5, CSS3 (Animações 8-bit & 3D Tilt), JavaScript (ES6+), Three.js.
- **Infraestrutura & Versionamento:** Git, GitHub e Render Cloud.

---

## 📂 Estrutura do Repositório

```text
├── app.py                  # Servidor Flask, hub de salas WebSocket e rotas da aplicação
├── templates/
│   └── index.html          # Interface SPA, motor de combate, renderizador 3D e FBC/MMR
├── recommender_fbc.py      # Módulo analítico de vetorização híbrida (TF-IDF + Numérico)
├── preprocessing.py        # Pipeline de normalização e codificação de atributos
├── deck_builder.py         # Algoritmo MMR e validação de restrições de raridade
├── user_profile.py         # Modelagem de centróide e preferências do jogador
├── requirements.txt        # Dependências do projeto para execução local e nuvem
└── README.md               # Documentação técnica do projeto
```

---

## 🚀 Como Executar Localmente

### 1. Clonar o repositório
```bash
git clone https://github.com/carloseduardodedeussilva02/colapse-tale-fbc-recommender.git
cd colapse-tale-fbc-recommender
```

### 2. Criar e ativar o ambiente virtual (Linux/Ubuntu)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 4. Iniciar o servidor
```bash
python3 app.py
```
- **Acesso Local:** Abra `http://127.0.0.1:5000` no navegador.
- **Multijogador no mesmo Wi-Fi:** Utilize o IP da rede local exibido no terminal (ex: `http://192.168.100.13:5000`) nos celulares ou notebooks conectados à mesma rede.

---

## 👨‍💻 Autor

**Carlos Eduardo de Deus Silva**  
Estudante de Engenharia / Computação — Universidade do Estado do Amazonas (UEA)  
Disciplina: *Oficina de Desenvolvimento de Sistemas I (ODS I)*