"""
Run all 4 notebooks in order using nbconvert.
Equivalent to executing each notebook top-to-bottom in Jupyter.
"""

import subprocess
import sys
import os

NOTEBOOKS = [
    "notebooks/01_data_pipeline.ipynb",
    "notebooks/02_lstm_model.ipynb",
    "notebooks/03_finbert_sentiment.ipynb",
    "notebooks/04_fusion_evaluation.ipynb",
]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

for nb in NOTEBOOKS:
    nb_path = os.path.join(PROJECT_ROOT, nb)
    print(f"\n{'='*70}")
    print(f"  RUNNING: {nb}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",
            "--ExecutePreprocessor.kernel_name=python3",
            nb_path,
        ],
        cwd=os.path.join(PROJECT_ROOT, "notebooks"),
        capture_output=False,
    )
    
    if result.returncode != 0:
        print(f"\n*** FAILED: {nb} (exit code {result.returncode}) ***")
        sys.exit(1)
    else:
        print(f"\n✓ COMPLETED: {nb}")

print(f"\n{'='*70}")
print("  ALL NOTEBOOKS COMPLETED SUCCESSFULLY!")
print(f"{'='*70}")
