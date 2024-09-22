
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Data Loading and Preprocessing - Add your dataset here
# Assuming X_final and y are prepared
# X_final = [Your feature matrix]
# y = [Your target variable]

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42)

# Logistic Regression Model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
y_prob_logreg = logreg.predict_proba(X_test)[:, 1]
classification_rep_logreg = classification_report(y_test, y_pred_logreg)
auc_roc_logreg = roc_auc_score(y_test, y_prob_logreg)

print("Logistic Regression Classification Report:")
print(classification_rep_logreg)
print(f"Logistic Regression AUC-ROC: {auc_roc_logreg:.4f}")

# XGBoost Model
xgb_clf = xgb.XGBClassifier(random_state=42)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]
classification_rep_xgb = classification_report(y_test, y_pred_xgb)
auc_roc_xgb = roc_auc_score(y_test, y_prob_xgb)

print("XGBoost Classification Report:")
print(classification_rep_xgb)
print(f"XGBoost AUC-ROC: {auc_roc_xgb:.4f}")

# Hyperparameter Tuning for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}

grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_xgb_clf = grid_search.best_estimator_

y_pred_best_xgb = best_xgb_clf.predict(X_test)
y_prob_best_xgb = best_xgb_clf.predict_proba(X_test)[:, 1]
classification_rep_best_xgb = classification_report(y_test, y_pred_best_xgb)
auc_roc_best_xgb = roc_auc_score(y_test, y_prob_best_xgb)

print("Tuned XGBoost Classification Report:")
print(classification_rep_best_xgb)
print(f"Tuned XGBoost AUC-ROC: {auc_roc_best_xgb:.4f}")

# Feature Importance for XGBoost
xgb.plot_importance(best_xgb_clf, importance_type='gain', max_num_features=10)
plt.title('Top 10 Feature Importance (Gain)')
plt.show()
