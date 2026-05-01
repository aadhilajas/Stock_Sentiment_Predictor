#!/usr/bin/env python3
"""
GitHub push setup script for P08 Stock Sentiment Predictor
Initializes git, configures credentials, and pushes to GitHub.
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent
GITHUB_USERNAME = "aadhilajas"
GITHUB_EMAIL = "aadhilajasakr@gmail.com"
GITHUB_REPO = "Stock_Sentiment_Predictor"
GITHUB_URL = f"https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"

# Files/folders to exclude from git
GITIGNORE_CONTENT = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
pip-log.txt
pip-delete-this-directory.txt

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db

# Project specific
results/*.pth
results/*.npy
results/finbert_finetuned/
data/processed/
notebooks/.ipynb_checkpoints/
.streamlit/

# Large files
*.tar.gz
*.zip

# Data cache (keep CSV and small files)
!results/experiment_results.csv
!results/backtest_*.png
!results/closing_prices_chart.png
!results/rsi_macd_aapl.png
!results/class_balance.png
"""


def run_command(cmd, description=""):
    """Run a shell command and return the result."""
    print(f"\n{'='*70}")
    if description:
        print(f"📋 {description}")
    print(f"   $ {cmd}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print(f"✓ Output:\n{result.stdout}")
        if result.stderr and result.returncode != 0:
            print(f"✗ Error:\n{result.stderr}")
        
        return result.returncode == 0, result.stdout, result.stderr
    
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False, "", str(e)


def main():
    print("\n" + "="*70)
    print("🚀 P08 Stock Sentiment Predictor — GitHub Push Setup")
    print("="*70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"GitHub URL: {GITHUB_URL}")
    print(f"User: {GITHUB_USERNAME} ({GITHUB_EMAIL})")
    
    # Step 1: Create .gitignore
    print("\n\n[Step 1/5] Creating .gitignore...")
    gitignore_path = PROJECT_ROOT / ".gitignore"
    if not gitignore_path.exists():
        with open(gitignore_path, "w") as f:
            f.write(GITIGNORE_CONTENT)
        print("✓ .gitignore created")
    else:
        print("⊘ .gitignore already exists")
    
    # Step 2: Initialize git repository
    print("\n[Step 2/5] Initializing git repository...")
    git_dir = PROJECT_ROOT / ".git"
    if git_dir.exists():
        print("⊘ Git repository already initialized")
    else:
        success, out, err = run_command("git init", "Initialize git repo")
        if not success:
            print("✗ Failed to initialize git. Make sure git is installed and in PATH.")
            print("   Install from: https://git-scm.com/download/win")
            return False
    
    # Step 3: Configure git user
    print("\n[Step 3/5] Configuring git user...")
    run_command(f'git config user.email "{GITHUB_EMAIL}"', f"Set git email to {GITHUB_EMAIL}")
    run_command(f'git config user.name "{GITHUB_USERNAME}"', f"Set git user to {GITHUB_USERNAME}")
    
    # Step 4: Add files and create initial commit
    print("\n[Step 4/5] Adding files and creating commit...")
    run_command("git add .", "Add all files")
    
    commit_msg = "Initial commit: P08 Stock Sentiment Predictor - LSTM + FinBERT fusion dashboard"
    success, out, err = run_command(
        f'git commit -m "{commit_msg}"',
        f"Commit with message: {commit_msg[:50]}..."
    )
    
    if not success:
        if "nothing to commit" in err:
            print("⊘ No changes to commit (repo already up to date)")
        else:
            print(f"✗ Commit failed: {err}")
            return False
    
    # Step 5: Add remote and push
    print("\n[Step 5/5] Setting up remote and pushing to GitHub...")
    
    run_command(f'git remote remove origin', "Remove existing remote if any")
    success, out, err = run_command(
        f'git remote add origin "{GITHUB_URL}"',
        f"Add remote: {GITHUB_URL}"
    )
    
    if not success:
        print(f"⊘ Remote may already exist, continuing...")
    
    print("\n" + "="*70)
    print("📤 PUSHING TO GITHUB...")
    print("="*70)
    print("\n⚠️  You will be prompted for GitHub authentication.")
    print("   Use your GitHub Personal Access Token or SSH key.")
    print("   If using HTTPS, create a PAT at: https://github.com/settings/tokens")
    print("\n")
    
    success, out, err = run_command(
        "git push -u origin master",
        "Push to GitHub (main branch)"
    )
    
    if not success:
        if "error" in err.lower():
            print("\n✗ Push failed. Trying 'main' branch...")
            success, out, err = run_command(
                "git push -u origin main",
                "Push to GitHub (main branch)"
            )
    
    print("\n" + "="*70)
    if success:
        print("✅ SUCCESS! Repository pushed to GitHub")
        print(f"   URL: {GITHUB_URL}")
    else:
        print("⚠️  Push may have failed or requires authentication")
        print(f"   URL: {GITHUB_URL}")
        print("\n   Manual steps:")
        print("   1. Create a GitHub Personal Access Token at: https://github.com/settings/tokens")
        print("   2. Run: git push -u origin master")
        print("   3. Enter username and PAT when prompted")
    print("="*70 + "\n")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
