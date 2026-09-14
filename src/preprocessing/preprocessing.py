import os
import pandas as pd
import numpy as np

def run_preprocessing():
    df = pd.read_excel(r'././data/base.xlsx')
    
    df['ESTA_EM_REGIAO_METROPOLITANA'] = df['ESTA_EM_REGIAO_METROPOLITANA'].notna().astype(int)
    
    df_feat = df.copy()
    
    mapeamento_renda_mediana = {
        0: 0.0,
        1: 0.25,
        2: 0.75,
        3: 1.5,
        4: 2.5,
        5: 4.0,
        6: 7.5,
        7: 15.0,
        8: 25.0
    }
    
    df_feat['RENDA_NUMERICA_APROX'] = df_feat['FAIXA_RENDIMENTO_BRUTO'].map(mapeamento_renda_mediana)
    df_feat['NUMERO_PESSOAS_DOMICILIO'] = pd.to_numeric(df_feat['NUMERO_PESSOAS_DOMICILIO'], errors='coerce')
    df_feat['PROXY_RENDA_PER_CAPITA'] = df_feat['RENDA_NUMERICA_APROX'] / df_feat['NUMERO_PESSOAS_DOMICILIO']
    
    df_feat['FLAG_BAIXA_RENDA'] = (df_feat['PROXY_RENDA_PER_CAPITA'] < 0.5).astype(int)
    df_feat['FLAG_DISTORCAO_IDADE_SERIE'] = df_feat['QUAL_SERIE_FREQUENTA'].astype(str).apply(lambda x: 1 if '1' in x else 0)
    
    df_feat['INDICE_VULNERABILIDADE'] = (
        (df_feat['SITUACAO_DOMICILIO'] == 2).astype(int) +
        df_feat['FLAG_BAIXA_RENDA'] +
        (df_feat['TIPO_ESCOLA'] == 2).astype(int) +
        df_feat['FLAG_DISTORCAO_IDADE_SERIE']
    )
    
    df_feat['ESCOLA_X_RENDA'] = df_feat['TIPO_ESCOLA'].astype(str) + "_" + pd.qcut(df_feat['PROXY_RENDA_PER_CAPITA'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop').astype(str)
    df_feat['ESCOLA_X_VULNERABILIDADE'] = df_feat['TIPO_ESCOLA'].astype(str) + "_" + df_feat['INDICE_VULNERABILIDADE'].astype(str)
    df_feat['METROPOLITANA_X_RENDA'] = df_feat['ESTA_EM_REGIAO_METROPOLITANA'].astype(str) + "_" + df_feat['FAIXA_RENDIMENTO_BRUTO'].astype(str)
    
    features_finais_categoricas = [
        'SEXO',
        'COR_OU_RACA',
        'SITUACAO_DOMICILIO',
        'TIPO_PESSOA_NO_DOMICILIO',
        'FAIXA_RENDIMENTO_BRUTO',
        'TIPO_ESCOLA',
        'QUAL_SERIE_FREQUENTA',
        'ESCOLA_X_VULNERABILIDADE'
    ]
    
    features_finais_numericas = [
        'NUMERO_PESSOAS_DOMICILIO',
        'PROXY_RENDA_PER_CAPITA',
        'FLAG_DISTORCAO_IDADE_SERIE',
        'INDICE_VULNERABILIDADE'
    ]
    
    X_cat = pd.get_dummies(df_feat[features_finais_categoricas].astype(str), drop_first=True)
    X_num = df_feat[features_finais_numericas].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    X = pd.concat([X_num, X_cat], axis=1)
    y = pd.to_numeric(df_feat['SABE_LER_E_ESCREVER'], errors='coerce')
    y = y.apply(lambda val: 1 if val == 1 else (0 if val == 2 else val))
    
    df_processed = pd.concat([X, y.rename('target')], axis=1)
    df_processed.to_excel(r'src/preprocessing/base_processada.xlsx', index=False)

if __name__ == '__main__':
    run_preprocessing()