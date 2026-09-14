import os
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.preprocessing import run_preprocessing
from src.modeling.modeling import run_modeling
from src.evaluation.evaluation import run_evaluation
from src.visualization.visualization import run_visualization

def main():
    print("Executando a etapa de pré-processamento...")
    run_preprocessing()
    
    print("Executando a etapa de modelagem...")
    run_modeling()
    
    print("Executando a etapa de avaliação...")
    run_evaluation()
    
    print("Executando a etapa de visualização...")
    run_visualization()
    
    print("Processo finalizado!")

if __name__ == '__main__':
    main()