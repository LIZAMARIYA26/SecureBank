# Fraudulent Detection in Banking System using Random Forest Classifier

Machine Learning project :
 *Fraudulent Detection in Banking System using Random Forest Classifier Algorithm*.

The system trains a Random Forest on the **BankSim** synthetic bank-payment dataset and serves a banking-style Flask console that scores live payments, batch CSVs and simulated transactions.

## Project overview

Traditional rule reviews are too slow for digital payments. This project follows the paper’s pipeline:

1. Load and inspect BankSim.
2. Clean, encode and scale features.
3. Engineer behaviour features (frequency, merchant risk, velocity, weekend, spending band).
4. Train a Random Forest (70:30 split).
5. Tune with GridSearchCV and compare Logistic Regression, Decision Tree and KNN.
6. Explain decisions with SHAP plus a plain-language sentence.
7. Expose the model through a SecureBank AI web desk.

Hold-out metrics (accuracy, precision, recall, F1, ROC-AUC) and report charts are written to `outputs/`.

## Objectives

- Detect fraudulent BankSim payments with a Random Forest classifier.
- Show a complete, commented ML pipeline students can defend in a viva.
- Provide a professional web UI for single, batch and simulated scoring.
- Keep preprocessing, feature engineering and plots in separate modules.

## Technologies used

| Layer | Stack |
| --- | --- |
| Language | Python 3.10+ |
| Web | Flask, HTML, CSS, JavaScript, Chart.js |
| ML | scikit-learn Random Forest, GridSearchCV |
| Data | pandas, NumPy |
| Plots | matplotlib, seaborn |
| Persistence | joblib |
| Explainability | SHAP |

## Dataset

**BankSim** (`dataset/banksim.csv`) is a publicly studied simulator of Spanish bank payments (Lopez-Rojas et al.). Typical columns:

`step, customer, age, gender, zipcodeOri, merchant, zipMerchant, category, amount, fraud`

Source used in this repo: the BankSim CSV published with common fraud-detection tutorials (originally Kaggle *Synthetic data from a financial payment system*).

A tiny demo file for batch scoring lives at `dataset/sample_transactions.csv`.

## Folder structure

```
fraud-detection-randomforest/
  app.py
  train_model.py
  config.py
  requirements.txt
  README.md
  dataset/banksim.csv
  models/random_forest_model.pkl
  static/style.css
  static/script.js
  templates/index.html
  templates/prediction.html
  templates/dashboard.html
  utils/preprocessing.py
  utils/feature_engineering.py
  utils/visualization.py
  notebooks/EDA.ipynb
  outputs/
```

## Installation

```bash
cd fraud-detection-randomforest
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

From this folder. Train (creates `models/` and `outputs/` charts), then start the web desk:

```bash
python3 train_model.py
python3 app.py
```

(`python` works if that alias exists on your machine.)

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

| Route | Purpose |
| --- | --- |
| `/` | Home / control centre |
| `/predict` | Score one payment (POST) |
| `/dashboard` | Counts, charts, recent table |
| `/generate` | Random realistic transaction + instant score |

## Web features

- Banking blue/white theme, sidebar, cards, dark-mode toggle
- Fraud (red) / genuine (green) result cards with probability meter
- CSV upload and CSV download of batch scores
- Loading spinner, input checks, flash alerts
- Prediction history on the dashboard
- SHAP plots from training plus a sentence explaining each live score

## Screenshots

Add viva / report captures here after you run the app:

1. Home page — `static/images/screenshot-home.png`
2. Prediction (fraud card) — `static/images/screenshot-fraud.png`
3. Prediction (genuine card) — `static/images/screenshot-genuine.png`
4. Dashboard — `static/images/screenshot-dashboard.png`

Training also saves report figures in `outputs/` (confusion matrix, ROC, feature importance, heatmap, accuracy comparison).

## Modules (mapped to code)

| Module | Where |
| --- | --- |
| 1 Dataset loading | `utils/preprocessing.py`, `train_model.py` |
| 2 Preprocessing | `utils/preprocessing.py` |
| 3 EDA | `utils/visualization.py`, `notebooks/EDA.ipynb` |
| 4 Feature engineering | `utils/feature_engineering.py` |
| 5 Feature selection | RF importances in `train_model.py` |
| 6 Model building | `train_model.py` |
| 7 GridSearchCV | `train_model.py` |
| 8 Evaluation | `train_model.py` + `outputs/` |
| 9 Web app | `app.py` + `templates/` |
| 10 Explainability | SHAP plots + `explain_in_words()` |
| 11 Simulation | `/generate` |
| 14 Report outputs | `outputs/*.png` |

## Future scope

- SMOTE for class imbalance
- XGBoost and CatBoost challengers
- Isolation Forest for unseen fraud patterns
- Streaming detection on the payment switch
- Continuous retraining on confirmed cases
- API integration with core banking

## Academic note

