import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title='Previsão de Alfabetização Infantil',
    page_icon='📚',
    layout='centered',
)

model = joblib.load(
    r'best.pkl'
)

st.title('Previsão de Alfabetização Infantil')
st.markdown(
    'Esta ferramenta utiliza modelos de aprendizado de máquina para prever a probabilidade de alfabetização de crianças com base em indicadores socioeconômicos, educacionais e características do domicílio extraídos de dados da PNAD.'
)
st.markdown('---')

st.subheader('👧 Dados e Escola da Criança')
st.markdown('Insira abaixo as informações referentes à criança e à sua vida escolar:')

col1, col2 = st.columns(2)

with col1:
  opcoes_sexo = {'Menino': 1, 'Menina': 2}
  sel_sexo = st.selectbox('Sexo', list(opcoes_sexo.keys()))

  opcoes_cor = {
      'Branca': 1,
      'Preta': 2,
      'Amarela': 3,
      'Parda': 4,
      'Indígena': 5,
  }
  sel_cor = st.selectbox('Cor ou Raça', list(opcoes_cor.keys()))

with col2:
  opcoes_escola = {'Rede privada': 1, 'Rede pública': 2}
  sel_escola = st.selectbox('Tipo de Escola que Frequenta', list(opcoes_escola.keys()))

  opcoes_serie = {'1º Ano': 1, '2º Ano': 2}
  sel_serie = st.selectbox('Qual Série Frequenta', list(opcoes_serie.keys()))

st.markdown('---')

st.subheader('🏠 Onde a criança mora?')
st.markdown('Informe os detalhes sobre o domicílio e a família:')

col3, col4 = st.columns(2)

with col3:
  opcoes_domicilio = {'Urbana': 1, 'Rural': 2}
  sel_domicilio = st.selectbox('Situação do Domicílio', list(opcoes_domicilio.keys()))

  opcoes_tipo_pessoa = {
    #   'Pessoa responsável pelo domicílio': 1,
    #   'Cônjuge ou companheiro(a) de sexo diferente': 2,
    #   'Cônjuge ou companheiro(a) do mesmo sexo': 3,
      'Filho(a) do responsável e do cônjuge': 4,
      'Filho(a) somente do responsável': 5,
      'Enteado(a)': 6,
    #   'Genro ou nora': 7,
    #   'Pai, mãe, padrasto ou madrasta': 8,
    #   'Sogro(a)': 9,
      'Neto(a)': 10,
      'Bisneto(a)': 11,
      'Irmão ou irmã': 12,
    #   'Avô ou avó': 13,
    #   'Outro parente': 14,
    #   'Agregado(a) - Não parente que não compartilha despesas': 15,
    #   'Convivente - Não parente que compartilha despesas': 16,
    #   'Pensionista': 17,
    #   'Empregado(a) doméstico(a)': 18,
    #   'Parente do(a) empregado(a) doméstico(a)': 19,
  }
  sel_tipo_pessoa = st.selectbox(
      'Condição da criança no lar', list(opcoes_tipo_pessoa.keys())
  )

with col4:
  opcoes_renda = {
      'Sem rendimento': 0,
      'Até 0,5 SM': 1,
      '0,5 a 1 SM': 2,
      '1 a 2 SM': 3,
      '2 a 3 SM': 4,
      '3 a 5 SM': 5,
      '5 a 10 SM': 6,
      '10 a 20 SM': 7,
      'Mais de 20 SM': 8,
  }
  sel_renda = st.selectbox(
      'Faixa de rendimento do lar onde a criança reside', list(opcoes_renda.keys())
  )

  numero_pessoas = st.number_input(
      'Número de Pessoas no do lar onde a criança reside', min_value=1, max_value=20, value=4
  )

st.markdown('---')

sexo = opcoes_sexo[sel_sexo]
cor_raca = opcoes_cor[sel_cor]
situacao_domicilio = opcoes_domicilio[sel_domicilio]
tipo_pessoa = opcoes_tipo_pessoa[sel_tipo_pessoa]
faixa_rendimento = opcoes_renda[sel_renda]
tipo_escola = opcoes_escola[sel_escola]
qual_serie = opcoes_serie[sel_serie]

mapeamento_renda_mediana = {
    0: 0.0,
    1: 0.25,
    2: 0.75,
    3: 1.5,
    4: 2.5,
    5: 4.0,
    6: 7.5,
    7: 15.0,
    8: 25.0,
}

renda_aprox = mapeamento_renda_mediana.get(faixa_rendimento, 0.0)
proxy_renda_per_capita = renda_aprox / numero_pessoas

flag_baixa_renda = 1 if proxy_renda_per_capita < 0.5 else 0
flag_distorcao_idade_serie = 1 if '1' in str(qual_serie) else 0

indice_vulnerabilidade = (
    (1 if situacao_domicilio == 2 else 0)
    + flag_baixa_renda
    + (1 if tipo_escola == 2 else 0)
    + flag_distorcao_idade_serie
)
escola_x_vulnerabilidade = f'{tipo_escola}_{indice_vulnerabilidade}'
esta_em_regiao_metropolitana = 0
metropolitana_x_renda = f'{esta_em_regiao_metropolitana}_{faixa_rendimento}'

input_dict = {
    'NUMERO_PESSOAS_DOMICILIO': [numero_pessoas],
    'PROXY_RENDA_PER_CAPITA': [proxy_renda_per_capita],
    'FLAG_DISTORCAO_IDADE_SERIE': [flag_distorcao_idade_serie],
    'INDICE_VULNERABILIDADE': [indice_vulnerabilidade],
    'ESTA_EM_REGIAO_METROPOLITANA': [esta_em_regiao_metropolitana],
    'SEXO_' + str(sexo): [1] if str(sexo) != '1' else [0],
    'COR_OU_RACA_' + str(cor_raca): [1] if str(cor_raca) != '1' else [0],
    'SITUACAO_DOMICILIO_' + str(situacao_domicilio): [1]
    if str(situacao_domicilio) != '1'
    else [0],
    'TIPO_PESSOA_NO_DOMICILIO_' + str(tipo_pessoa): [1]
    if str(tipo_pessoa) != '1'
    else [0],
    'FAIXA_RENDIMENTO_BRUTO_' + str(faixa_rendimento): [1]
    if str(faixa_rendimento) != '0'
    else [0],
    'TIPO_ESCOLA_' + str(tipo_escola): [1] if str(tipo_escola) != '1' else [0],
    'QUAL_SERIE_FREQUENTA_' + str(qual_serie): [1]
    if str(qual_serie) != '1'
    else [0],
    'ESCOLA_X_VULNERABILIDADE_' + str(escola_x_vulnerabilidade): [1],
    'METROPOLITANA_X_RENDA_' + str(metropolitana_x_renda): [1],
}

input_data = pd.DataFrame(input_dict)

if hasattr(model, 'feature_names_in_'):
  for col in model.feature_names_in_:
    if col not in input_data.columns:
      input_data[col] = 0
  input_data = input_data[model.feature_names_in_]

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
  executar_previsao = st.button(
      'Prever Alfabetização', use_container_width=True
  )

if executar_previsao:
  pred = model.predict(input_data)[0]
  proba = model.predict_proba(input_data)[0][1]

  st.markdown('---')
  if pred == 1:
    st.success(f'Previsão: Sabe ler e escrever (Probabilidade: {proba:.2%})')
  else:
    st.warning(
        f'Previsão: Não sabe ler e escrever (Probabilidade: {1 - proba:.2%})'
    )