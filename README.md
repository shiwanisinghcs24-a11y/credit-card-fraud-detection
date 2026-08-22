# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions in a highly imbalanced dataset, using Logistic Regression and Random Forest.

## Problem Statement

This project builds a machine learning model to detect fraudulent credit card transactions in a highly imbalanced dataset, where only 0.17% of transactions are fraud. This rarity makes naive approaches — like simply optimizing for accuracy — fundamentally misleading: a model that predicts "not fraud" for every transaction would score ~99.8% accuracy while catching zero fraud cases. This project instead uses specialized techniques to handle class imbalance effectively and evaluates performance using metrics that actually reflect real-world usefulness.

## Dataset

The dataset was sourced from Kaggle (the ULB Credit Card Fraud Detection dataset), containing 284,807 transactions, of which only 492 (0.17%) were fraudulent — a severe class imbalance. The dataset includes 31 columns: `Time`, `Amount`, `Class` (the target: 0 = normal, 1 = fraud), and 28 anonymized features (`V1`–`V28`) transformed via PCA for privacy reasons, meaning their real-world meaning cannot be directly interpreted.

## Approach / Methodology

1. **Data Cleaning** — Removed 1,081 duplicate transactions.
2. **Exploratory Data Analysis (EDA)** — Analyzed transaction amounts by class using grouping and boxplot visualizations.
3. **Train-Test Split** — 80/20 split with stratified sampling to preserve the original fraud/normal ratio in both sets.
4. **Handling Class Imbalance (SMOTE)** — Applied SMOTE to generate synthetic, realistic fraud examples in the training data only, keeping the test set untouched for honest evaluation.
5. **Feature Scaling** — Standardized all features using StandardScaler, since `Amount` had a vastly different range than the anonymized `V1`-`V28` features.
6. **Model Training & Evaluation** — Trained and evaluated Logistic Regression and Random Forest, using Precision, Recall, F1-score, and ROC-AUC rather than accuracy alone.
7. **Feature Importance** — Compared which features most influenced predictions across both models.
8. **Risk Classification** — Built a 3-tier risk system (Low/Medium/High) based on predicted fraud probabilities, for a more practical decision framework.

## Results

Two models were trained and evaluated on the untouched, imbalanced test set:

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Precision (Fraud) | 14% | **89%** |
| Recall (Fraud) | 85% | 77% |
| F1-score (Fraud) | 0.24 | **0.82** |
| ROC-AUC | 0.963 | 0.961 |

Random Forest was selected as the superior model overall. While Logistic Regression achieved slightly higher Recall, it suffered from extremely poor Precision (14%), generating 488 false positives — meaning most of its fraud alerts were actually normal transactions wrongly flagged. Random Forest achieved a much better balance, with only 9 false positives and a significantly higher F1-score (0.82 vs 0.24), making it far more practical for real-world deployment despite catching slightly fewer fraud cases (73 vs 81).

## Key Insights

1. **Amount is deceptive** — While `Amount` was a meaningful positive predictor for Logistic Regression (weight = 2.35), it was almost irrelevant to Random Forest (importance = 0.004), suggesting Random Forest relies more heavily on the anonymized PCA features to detect fraud patterns that transaction amount alone doesn't capture.

2. **High ROC-AUC doesn't guarantee good real-world performance** — Both models scored nearly identical ROC-AUC (0.963 vs 0.961), meaning both were similarly good at generally sensing which transactions seemed more suspicious overall. However, at the actual 0.5 cutoff used to make real decisions, Random Forest was dramatically more precise (89% vs 14%) — showing that having good overall judgment and making good final decisions aren't always the same thing.

3. **Feature agreement across models strengthens confidence** — Both models independently identified V14, V10, and V12 as top predictors, despite using fundamentally different algorithms — reinforcing that these features carry genuine fraud signal.

4. **Even strong models have blind spots** — In the tiered risk classification system, 20 out of 95 real fraud cases were confidently but wrongly classified as "Low Risk," highlighting that no model is perfect, and a real-world system still needs additional safeguards or human oversight.

## How to Run

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `Fraud_Detection.ipynb` in Jupyter/Colab
4. Run all cells sequentially

## Tech Stack

Python, Pandas, scikit-learn, imbalanced-learn (SMOTE), Matplotlib
