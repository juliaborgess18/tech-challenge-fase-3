import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    fbeta_score
)

def run_evaluation():
    modelos_treinados, X_test, y_test = joblib.load(r'src/modeling/trained_models.pkl')
    
    for nome, modelo in modelos_treinados.items():
        preds_proba = modelo.predict_proba(X_test)[:, 1]
        
        thresholds = np.linspace(0.01, 0.99, 200)
        best_thresh = 0.5
        best_score = 0
        
        for t in thresholds:
            preds_t = (preds_proba >= t).astype(int)
            score = fbeta_score(y_test, preds_t, beta=1.5, pos_label=0)
            if score > best_score:
                best_score = score
                best_thresh = t
                
        preds = (preds_proba >= best_thresh).astype(int)

    melhor_score = -1
    melhor_modelo = None

    for nome, modelo in modelos_treinados.items():
        preds_proba = modelo.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, preds_proba)
        if score > melhor_score:
            melhor_score = score
            melhor_modelo = modelo

    joblib.dump(
        melhor_modelo,
        'best.pkl',
    )

if __name__ == '__main__':
    run_evaluation()