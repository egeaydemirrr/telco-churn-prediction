# Telco Customer Churn Prediction

Predicting which telecom customers will leave, using the IBM Telco Customer Churn dataset (7,043 customers, about 26% churn). Built as the final project of a machine-learning programming course.

The focus is on a clean, leak-free model-selection process rather than on a single model:

- **Exploration and cleaning:** missing values (`TotalCharges`), target imbalance, numeric vs. categorical features.
- **Preprocessing in a pipeline:** imputation, scaling, one-hot encoding (`drop='first'`), all inside `sklearn` / `imblearn` pipelines so nothing leaks from the test set.
- **Imbalance handling:** SMOTE vs. RandomOverSampler, compared inside cross-validation.
- **Model search:** Logistic Regression, KNN, Random Forest and discriminant analysis, with optional PCA / feature selection, searched with `RandomizedSearchCV` and **nested 5-fold cross-validation**, scored on F1 because accuracy is misleading with imbalanced classes.
- **Refinement and diagnostics:** a finer search around the winner, learning curve and validation curve for the regularisation strength `C`.

## Result

| | F1 |
|---|---|
| Nested CV estimate | 0.64 |
| Held-out test set  | 0.62 |

The winner in every outer fold was **Logistic Regression with L1 penalty + RandomOverSampler**. It catches **73% of the customers who churn**. The close CV and test scores and the learning curve show the model is not overfitting.

## Run it

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
```

Data: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (IBM sample dataset, also on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).

## License

MIT
