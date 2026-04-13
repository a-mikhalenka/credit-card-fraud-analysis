# Credit Card Fraud Detection

## Overview
Fraudulent transactions result in direct financial losses and damage customer trust. This project builds and evaluates machine learning models to automatically detect fraudulent credit card transactions.
The goals of this analysis are to:
- Identify the key factors associated with fraudulent transactions
- Build a machine learning model that predicts whether a transaction is fraudulent
- Minimise financial losses by maximising detection of fraud
 
## Data
The dataset is available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains 284,807 transaction records (283,726 after removing 1,081 duplicates) with 31 features (28 features are PCA-transformed and uncorrelated with each other).

## Key findings

### EDA
- 14 strong predictors identified: V17, V14, V12, V10, V16, V3, V7, V11, V4, V18, V1, V9, V5, V2
- Amount and Time show weak correlation with fraud
- Fraud rate elevated during low volume periods - hours 2-5 and 26-29 spike to 1.57% vs 0.167% average
- Most strong predictors show fraud clustering below 0; V11, V4, V2 show fraud above 0

### Feature Engineering
- `Amount_log = np.log1p(Amount)` - addresses right skew in transaction amounts
- `Hour = np.floor(Time / 3600) % 24` - captures intra-day fraud timing patterns

### Feature Importance (XGBoost)
V14 emerged as the dominant predictor at 72% importance, with V12, V4 and V17 forming a distant second tier. Hour and Amount_log ranked near the bottom, suggesting the timing pattern and amount distribution did not translate into strong predictive signal for the model.

## Modelling
SMOTE was applied to the training set only (after train/test split) to address class imbalance. Three models were trained and compared.

### Results

| Model | Precision | Recall | F1 | ROC-AUC | Train-Test Avg Gap |
|-------|:---------:|:------:|:--:|:-------:|:-----------------:|
| Logistic Regression | 0.13 | 0.85 | 0.23 | 0.9634 | 0.02 |
| Random Forest | 0.82 | 0.80 | 0.81 | 0.9684 | 0.12 |
| **XGBoost** | **0.88** | **0.80** | **0.84** | **0.9675** | 0.16 |

**Primary metric: Recall** — in fraud detection, missing a real fraud case (false negative) is more costly than a false positive.

### Final Model: XGBoost
- `learning_rate=0.1`, `max_depth=7`, `n_estimators=200`, `subsample=0.8`
- Highest precision (0.88) and F1 (0.84) across all models
- Correctly identified 76 out of 95 fraud cases in the test set
- Only 10 legitimate transactions incorrectly flagged

## Recommendations
- Final model performed well on the test data, classifying **86** out of **56746** transactions as fraudulent. This volume makes it possible for employees to **manually check** these transactions in a short period of time.
- **Threshold** for fraud detection **can be adjusted** depending on the bank's priorities. Lowering the classification threshold will increase recall at the cost of higher false positives.
- Given that fraud evolves over time, the model **should be retrained regularly** on new transaction data to stay effective.
- `V14` feature, the **primary predictor**, should be **investigated more closely**. This was impossible to do after PCA, but the bank should be able to conduct deeper analysis.
- **Hours with lower transaction volume should be monitored** more extensively, as the amount of fraud increases during hours with lower overall activity.

## Limitations
The **interpretability** of the final results is **limited** because of data transformation using PCA. Future work could include real-time deployment and adjustments depending on the model performance.

## Technologies used
Python, pandas, numpy, matplotlib, seaborn, scikit-learn (Logistic Regression, Random Forest, GridSearchCV, SMOTE), XGBoost

