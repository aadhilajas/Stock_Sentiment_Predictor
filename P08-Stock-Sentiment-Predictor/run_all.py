"""
run_all.py — Run all 4 notebooks as Python scripts in order.
Extracts code cells from each .ipynb and executes them sequentially.
"""

import json
import sys
import os
import time

# ── Set up project root ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# Ensure src is importable
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

NOTEBOOKS = [
    "notebooks/01_data_pipeline.ipynb",
    "notebooks/02_lstm_model.ipynb",
    "notebooks/03_finbert_sentiment.ipynb",
    "notebooks/04_fusion_evaluation.ipynb",
]


def extract_code_cells(nb_path: str) -> str:
    """Extract all code cells from a Jupyter notebook as a single Python script."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_lines = []
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            code_lines.append(source)
            code_lines.append('\n')

    return '\n'.join(code_lines)


def run_notebook(nb_path: str, nb_num: int):
    """Extract and execute a single notebook."""
    full_path = os.path.join(PROJECT_ROOT, nb_path)
    
    print(f"\n{'█' * 70}")
    print(f"  [{nb_num}/4] RUNNING: {nb_path}")
    print(f"{'█' * 70}\n")
    
    # Change to notebooks dir so relative paths work (e.g., os.path.join(os.getcwd(), '..'))
    os.chdir(os.path.join(PROJECT_ROOT, 'notebooks'))
    
    code = extract_code_cells(full_path)
    
    # Replace display() calls which only work in IPython/Jupyter
    code = code.replace(
        "display(df.head()) if hasattr(__builtins__, '__IPYTHON__') else print(df.head())",
        "print(df.head())"
    )
    
    # Execute in a namespace that persists across cells
    namespace = {
        '__name__': '__main__',
        '__file__': full_path,
    }
    
    start = time.time()
    try:
        exec(code, namespace)
    except Exception as e:
        print(f"\n{'!'*70}")
        print(f"  ERROR in {nb_path}: {type(e).__name__}: {e}")
        print(f"{'!'*70}\n")
        import traceback
        traceback.print_exc()
        raise
    
    elapsed = time.time() - start
    print(f"\n{'─' * 70}")
    print(f"  ✓ COMPLETED: {nb_path}  ({elapsed:.1f}s)")
    print(f"{'─' * 70}")
    
    # Return to project root
    os.chdir(PROJECT_ROOT)


if __name__ == '__main__':
    total_start = time.time()
    
    print(f"{'═' * 70}")
    print(f"  P08 — Running all 4 notebooks in order")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"{'═' * 70}")
    
    for i, nb in enumerate(NOTEBOOKS, 1):
        run_notebook(nb, i)
    
    total_elapsed = time.time() - total_start
    
    print(f"\n\n{'═' * 70}")
    print(f"  ✅ ALL 4 NOTEBOOKS COMPLETED SUCCESSFULLY!")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'═' * 70}")
