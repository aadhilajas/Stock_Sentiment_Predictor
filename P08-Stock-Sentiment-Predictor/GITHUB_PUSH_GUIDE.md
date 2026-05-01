# 🚀 How to Push P08 Project to GitHub

This guide will walk you through pushing the P08 Stock Sentiment Predictor project to your GitHub repository.

## Prerequisites

1. **GitHub Account**: https://github.com/aadhilajas (already created ✓)
2. **Repository Created**: `Stock_Sentiment_Predictor` (already created ✓)
3. **Git Installed**: Download from https://git-scm.com/download/win
4. **Personal Access Token**: Create at https://github.com/settings/tokens

---

## Step-by-Step Guide

### Step 1: Install Git (if not already installed)

1. Visit: https://git-scm.com/download/win
2. Download and run the installer
3. Follow the default installation steps
4. **Restart your terminal/PowerShell** after installation
5. Verify installation:
   ```bash
   git --version
   ```
   Should output: `git version 2.x.x`

---

### Step 2: Create a GitHub Personal Access Token

1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Fill in:
   - **Note**: P08 Stock Sentiment Predictor
   - **Expiration**: 90 days (or custom)
4. Select scopes:
   - ✅ `repo` (Full control of private repositories)
   - ✅ `workflow` (Update GitHub Actions workflows)
5. Click **"Generate token"**
6. **Copy the token** (you won't see it again!)
   - Save it somewhere safe temporarily

---

### Step 3: Configure Git Locally

Open PowerShell and run:

```bash
# Set your GitHub username
git config --global user.name "aadhilajas"

# Set your email
git config --global user.email "aadhilajasakr@gmail.com"

# Verify configuration
git config --global --list
```

---

### Step 4: Navigate to Your Project

```bash
cd d:/AI_ML_NLP/P08-Stock-Sentiment-Predictor
```

---

### Step 5: Initialize Git Repository

```bash
# Initialize git
git init

# Check status
git status
```

Expected output:
```
On branch master

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        .gitignore
        README.md
        app/
        ...
```

---

### Step 6: Add Files to Git

```bash
# Add all files
git add .

# Check what will be committed
git status
```

---

### Step 7: Create Initial Commit

```bash
git commit -m "Initial commit: P08 Stock Sentiment Predictor - LSTM + FinBERT fusion"
```

---

### Step 8: Create Main Branch

```bash
# Rename master to main (GitHub default)
git branch -M main

# Verify
git branch
```

---

### Step 9: Add GitHub Remote

```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/aadhilajas/Stock_Sentiment_Predictor.git

# Verify
git remote -v
```

Expected output:
```
origin  https://github.com/aadhilajas/Stock_Sentiment_Predictor.git (fetch)
origin  https://github.com/aadhilajas/Stock_Sentiment_Predictor.git (push)
```

---

### Step 10: Push to GitHub

```bash
# Push to GitHub
git push -u origin main
```

**When prompted**:
- **Username**: `aadhilajas`
- **Password**: Paste your Personal Access Token (not your actual password)

---

## ✅ Verification

Once the push completes, verify on GitHub:

1. Go to: https://github.com/aadhilajas/Stock_Sentiment_Predictor
2. You should see:
   - ✅ All files uploaded
   - ✅ `.gitignore` file (excludes cache, models, etc.)
   - ✅ `README.md` with full documentation
   - ✅ All source code in `src/`
   - ✅ Notebooks in `notebooks/`

---

## 🔄 Future Updates

After the initial push, for any future updates:

```bash
# 1. Make changes to files
# 2. Check status
git status

# 3. Stage changes
git add .

# 4. Commit
git commit -m "Description of changes"

# 5. Push
git push origin main
```

---

## 🆘 Troubleshooting

### ❌ "fatal: not a git repository"
```bash
# Make sure you're in the project directory
cd d:/AI_ML_NLP/P08-Stock-Sentiment-Predictor

# Re-initialize git
git init
```

### ❌ "'git' is not recognized"
- Git is not installed or not in PATH
- Download and install from: https://git-scm.com/download/win
- **Restart PowerShell after installation**

### ❌ "Permission denied (publickey)"
- You're using SSH instead of HTTPS
- Use HTTPS URL: `https://github.com/aadhilajas/Stock_Sentiment_Predictor.git`
- Use your PAT as password

### ❌ "fatal: remote origin already exists"
```bash
# Remove existing remote
git remote remove origin

# Add the correct remote
git remote add origin https://github.com/aadhilajas/Stock_Sentiment_Predictor.git
```

### ❌ "Authentication failed"
- Your PAT may have expired
- Create a new token at: https://github.com/settings/tokens
- Use the new token when prompted

---

## 📊 What Gets Pushed

### ✅ Included Files
- All Python source code (`src/`)
- Jupyter notebooks (`notebooks/`)
- README and documentation
- Configuration files
- Requirements.txt

### ❌ Excluded Files (via `.gitignore`)
- PyTorch models (`*.pth`) — too large
- NumPy arrays (`*.npy`) — too large
- Trained FinBERT weights
- Python cache (`__pycache__/`)
- IDE settings (`.vscode/`, `.idea/`)
- Virtual environment (`venv/`)

### 💾 Charts & Results (Small, Included)
- `experiment_results.csv` — metrics table
- `*.png` — plots and charts
- Markdown files — documentation

---

## 📝 Recommended Workflow

1. **Local Development**
   ```bash
   # Make changes
   # Test locally
   
   # Push when ready
   git add .
   git commit -m "Your message"
   git push origin main
   ```

2. **Collaborate**
   - Share repository link with teammates
   - Pull latest changes: `git pull origin main`

3. **Backup & Archive**
   - GitHub now serves as your cloud backup
   - Repository is version-controlled and auditable

---

## 🎯 Next Steps

After successful push:

1. ✅ Verify at: https://github.com/aadhilajas/Stock_Sentiment_Predictor
2. Add a **GitHub Pages** site (optional): https://pages.github.com
3. Set up **GitHub Actions** for CI/CD (optional)
4. Share with instructors: Send them the GitHub link

---

## 📞 Support

If you encounter issues:

1. Check **Troubleshooting** section above
2. View GitHub Help: https://docs.github.com
3. Git Documentation: https://git-scm.com/doc

---

**You're all set! Push your project now.** 🚀
