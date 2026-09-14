import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, RocCurveDisplay, fbeta_score

def run_visualization():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('images', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusão - {nome}')
        
        nome_arquivo_cm = f"matriz_confusao_{nome.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(os.path.join(output_dir, nome_arquivo_cm), bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(8, 6))
    for nome, modelo in modelos_treinados.items():
        RocCurveDisplay.from_estimator(
            modelo, X_test, y_test, name=nome, ax=plt.gca()
        )

    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.50)')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC - Comparativo de Modelos')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'curva_roc_comparativo.png'), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    run_visualization()