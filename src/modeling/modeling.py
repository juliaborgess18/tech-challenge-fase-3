import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier

def run_modeling():
    df_processed = pd.read_excel(r'src/preprocessing/base_processada.xlsx')
    
    X = df_processed.drop(columns=['target'])
    y = df_processed['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    modelos = {
        'Random Forest (SMOTE)': RandomForestClassifier(
            n_estimators=500, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1
        ),
        'LightGBM (Balanced)': lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=63, class_weight='balanced', random_state=42, verbose=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42
        ),
        'Decision Tree (SMOTE)': DecisionTreeClassifier(
            max_depth=15, random_state=42
        ),
    }
    
    modelos_treinados = {}
    for nome, modelo in modelos.items():
        if 'SMOTE' in nome or 'Decision Tree' in nome:
            modelo.fit(X_train_smote, y_train_smote)
        else:
            modelo.fit(X_train, y_train)
        modelos_treinados[nome] = modelo

    joblib.dump((modelos_treinados, X_test, y_test), r'src/modeling/trained_models.pkl')

if __name__ == '__main__':
    run_modeling()