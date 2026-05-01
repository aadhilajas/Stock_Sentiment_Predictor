import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys, os

NOTEBOOKS = [
    "notebooks/01_data_pipeline.ipynb",
    "notebooks/02_lstm_model.ipynb",
    "notebooks/03_finbert_sentiment.ipynb",
    "notebooks/04_fusion_evaluation.ipynb",
]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

for nb in NOTEBOOKS:
    path = os.path.join(PROJECT_ROOT, nb)
    print('\n' + '='*70)
    print(f"RUNNING: {nb}")
    print('='*70 + '\n')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            nb_node = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb_node, {'metadata': {'path': os.path.dirname(path) or '.'}})
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(nb_node, f)
        print(f"✓ COMPLETED: {nb}")
    except Exception as e:
        print(f"*** FAILED: {nb} — {e}")
        sys.exit(1)

print('\nALL NOTEBOOKS COMPLETED SUCCESSFULLY!')
