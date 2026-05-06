# Publication-Ready Code Checklist

## ✅ What Has Been Done

### Code Structure
- ✅ Modular package structure (`safe_metric/`)
- ✅ Separate CLI tool (`scripts/`)
- ✅ Unit tests (`tests/`)
- ✅ Example data and notebooks
- ✅ Proper Python package setup (`setup.py`)

### Documentation
- ✅ Comprehensive README with examples
- ✅ Docstrings in all modules and functions
- ✅ Contributing guidelines
- ✅ Usage examples in notebooks

### Configuration
- ✅ `requirements.txt` for dependencies
- ✅ `.gitignore` for clean repository
- ✅ Configurable parameters (tau, model, columns)

### Code Quality
- ✅ No hardcoded paths (all configurable)
- ✅ Error handling in CLI
- ✅ Type hints in function signatures
- ✅ Clear variable names
- ✅ Separated concerns (core logic vs. CLI vs. evaluation)

---

## 📋 Before Pushing to GitHub

### 1. Test the Code Locally

```bash
cd /home/claude/safe-quad-absa

# Install in development mode
pip install -e .

# Run unit tests
pytest tests/ -v

# Test CLI on sample data
python scripts/evaluate_dataset.py \
    examples/sample_input.csv \
    /tmp/test_output.csv \
    --verbose

# Verify output looks correct
head -20 /tmp/test_output.csv
```

### 2. Anonymize for Double-Blind Review

**Files to check:**
- [ ] `README.md` — Remove author names in citation (use `[Authors]`)
- [ ] `setup.py` — Keep generic "SAFE Research Team"
- [ ] `safe_metric/__init__.py` — Generic author
- [ ] No institutional emails in comments
- [ ] No file paths with personal usernames

**Search for identifying info:**
```bash
grep -r "Marwah" .
grep -r "Victoria University" .
grep -r "@victoria.edu" .
grep -r "/home/marwah" .
```

### 3. Update URLs in Files

Replace `YOUR-USERNAME` with your actual GitHub username OR leave as placeholder:
- `README.md` (GitHub URL)
- `setup.py` (url field)

### 4. Test Installation from Scratch

```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate

# Install from local directory
pip install /home/claude/safe-quad-absa

# Try importing
python -c "from safe_metric import SAFEMetric; print('✓ Import works')"

# Try CLI
safe-evaluate --help
```

---

## 🚀 Publishing Steps

### Step 1: Initialize Git Repository

```bash
cd /home/claude/safe-quad-absa
git init
git add .
git commit -m "Initial SAFE metric implementation"
```

### Step 2: Push to GitHub

Option A - Your Real Account (for post-acceptance):
```bash
git remote add origin https://github.com/YOUR-USERNAME/safe-quad-absa.git
git branch -M main
git push -u origin main
```

Option B - Anonymous Account (for submission):
```bash
# Create anonymous GitHub account first
# Then:
git remote add origin https://github.com/safe-eval-2026/safe-quad-absa.git
git branch -M main
git push -u origin main
```

### Step 3: Anonymize with Anonymous GitHub

1. Go to: https://anonymous.4open.science/
2. Paste: `https://github.com/YOUR-USERNAME/safe-quad-absa`
3. Get anonymized URL
4. Use in NeurIPS paper

---

## 📝 In Your Paper

### Data and Code Availability Section

```latex
\section*{Data and Code Availability}
The human evaluation dataset, review corpus, and evaluation materials are 
publicly available at Harvard Dataverse: \url{https://doi.org/10.7910/DVN/HISKDV}. 

The SAFE metric implementation, analysis scripts, and reproduction code are 
available at: \url{https://anonymous.4open.science/r/safe-quad-absa-XXXX/}.
```

### In Methods Section

Reference the code when describing SAFE:

```latex
Our implementation uses the \texttt{sentence-transformers} library with the 
\texttt{all-MiniLM-L6-v2} model. Complete implementation details and source 
code are provided in the supplementary materials and code repository.
```

---

## 🔍 Pre-Submission Verification

### Functionality Checks
- [ ] Code runs without errors on sample data
- [ ] Unit tests pass
- [ ] CLI produces expected output
- [ ] README examples are accurate

### Anonymization Checks
- [ ] No author names in code/docs
- [ ] No institutional affiliations
- [ ] No personal file paths
- [ ] Generic email addresses only

### Documentation Checks
- [ ] README installation instructions work
- [ ] Example usage is clear
- [ ] Citation placeholder is present
- [ ] License file included (MIT)

### Repository Checks
- [ ] `.gitignore` excludes unnecessary files
- [ ] No large data files committed
- [ ] Examples directory has sample data
- [ ] requirements.txt is complete

---

## 🎯 After NeurIPS Acceptance

### Update for Camera-Ready

1. **De-anonymize the code:**
   - Add real author names to README citation
   - Add institutional affiliations
   - Update author in `setup.py`

2. **Update paper:**
   - Replace anonymous GitHub URL with real URL
   - Add complete citation with authors

3. **Add NeurIPS badge to README:**
   ```markdown
   [![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue)]
   ```

4. **Release on PyPI (optional):**
   ```bash
   python setup.py sdist bdist_wheel
   twine upload dist/*
   ```

---

## 📧 Support

Questions about publication process:
- GitHub Issues: for code-related questions
- Email: safe.eval.2026@gmail.com
- NeurIPS E&D Track Chairs: evaluationsdatasets@neurips.cc

---

## Summary

**What you have now:**
✅ Professional, modular Python package
✅ Publication-ready code structure
✅ Comprehensive documentation
✅ Reproducible examples
✅ Unit tests for verification
✅ Ready for NeurIPS E&D submission

**Next steps:**
1. Test locally
2. Anonymize
3. Push to GitHub
4. Get anonymous URL
5. Include in paper submission
