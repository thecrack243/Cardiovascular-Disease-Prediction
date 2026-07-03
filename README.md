<div style="display: flex; justify-content: center; gap: 10px;">
    <div style="text-align: center;">
        <img src="../notebooks/cover2.png" style="width:900px; height:450px;"><br>
    </div>
</div>

# Cardiovascular Disease Prediction

Machine Learning Course Final Project

Binary classification pipeline for cardiovascular disease detection using the **UCI Cleveland Heart Disease Dataset** (297 patients, 5 clinical features).

## Models Implemented

| # | Model | Test Accuracy | AUC-ROC | Key Feature |
|---|-------|--------------|---------|-------------|
| 1 | Logistic Regression | 75.56% | **0.8442** | Strong linear baseline |
| 2 | SVM (GridSearch) | 73.33% | 0.8279 | 20 configs, RBF kernel |
| 3 | SVM (Optuna) | 72.22% | 0.7976 | Bayesian optimization, 20 trials |
| 4 | **PyTorch MLP** | **77.78%** | 0.8299 | **Best model** — 833 params |
| 5 | CBAM MLP | 72.22% | 0.8318 | Channel attention (+1.9% params) |

**Key Finding:** On 297 samples, simpler models generalize better. CBAM attention hurt accuracy by 5.56% due to insufficient data.

## Dataset

- **Source:** [UCI Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Samples:** 297 (after removing 6 rows with missing values)
- **Features:** `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
- **Split:** 70/30 stratified, `random_state=90`
- **Preprocessing:** Median imputation + Z-score standardization (fit on train only)

## Project Structure
├── data/
│   ├── heart.csv                  # Raw dataset
│   └── sample_test.csv            # Sample inference data
├── notebooks/
│   └── notebook.ipynb             # Full pipeline + interactive inference
├── .gitignore
└── requirements.txt


## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Open notebook and run the last cell for interactive inference
jupyter notebook notebooks/notebook.ipynb
