# Tech Challenge - Fase 3 | FIAP — Pós-Tech AI Scientist

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Concluído-success?style=for-the-badge" alt="Status: Concluído">
</p>

## 📋 Sobre o projeto

Este repositório contém a entrega oficial do **Tech Challenge da Fase 3** da **Pós-Tech em AI Scientist da FIAP**.

O projeto tem como objetivo aplicar técnicas de **Machine Learning** para prever se uma criança será considerada alfabetizada até o **2º ano do Ensino Fundamental**.

## 🛠️ Tecnologias e ferramentas utilizadas

- **Linguagem:** Python 3.12+
- **Machine Learning:** Scikit-learn e Joblib
- **Interface web:** Streamlit
- **Manipulação e análise de dados:** Pandas e NumPy
- **Visualização de dados:** Matplotlib e Seaborn

## 📂 Estrutura do repositório

```text
.
├── data/                  # Diretório para as bases de dados
├── notebooks/             # Notebooks Jupyter com análises e experimentos
├── src/                   # Códigos-fonte modulares da aplicação
├── main.py                # Execução da pipeline de modelagem
├── streamlit.py           # Aplicação web desenvolvida com Streamlit
├── requirements.txt       # Dependências do projeto
└── README.md              # Documentação principal
```

## 🖥️ Aplicação Streamlit

A interface interativa foi desenvolvida utilizando **Streamlit**, permitindo testar o modelo de forma visual e intuitiva.

🔗 [Acessar a aplicação online](https://tech-challenge-fase-3-alfabetizacao.streamlit.app/)

## 🚀 Como executar o projeto

### 1. Clone o repositório

```bash
git clone URL_DO_REPOSITORIO
cd NOME_DO_REPOSITORIO
```

### 2. Crie e ative um ambiente virtual

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux ou macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute a aplicação

```bash
streamlit run streamlit.py
```

A aplicação será disponibilizada localmente no endereço exibido pelo Streamlit, geralmente:

```text
http://localhost:8501
```
