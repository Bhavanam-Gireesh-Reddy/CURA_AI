import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from machine_learning.train import X_train, X_test, y_train, y_test
from sklearn.metrics import accuracy_score, classification_report

rf = joblib.load("machine_learning/artifacts/random_forest.pkl")
lr = joblib.load("machine_learning/artifacts/logistic_regression.pkl")
xgb = joblib.load("machine_learning/artifacts/xgboost.pkl")

##################################
# Parameters
##################################

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf_params = {
    "n_estimators": [200, 300, 400],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

lr_params = {
    "C": [0.001, 0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
    "max_iter": [500, 1000]
}

xgb_params = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2],
    "min_child_weight": [1, 3, 5]
}

##################################
# Gird Search CV for Finetuning Models
##################################
grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=rf_params,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)

grid_rf.fit(X_train, y_train)

print(grid_rf.best_params_)
print(grid_rf.best_score_)

best_rf = grid_rf.best_estimator_

joblib.dump(best_rf, "machine_learning/artifacts/random_forest_tuned.pkl")

# Gird Search CV for Logistic Regression

grid_lr = GridSearchCV(
    estimator=lr,
    param_grid=lr_params,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)

grid_lr.fit(X_train, y_train)

print(grid_lr.best_params_)
print(grid_lr.best_score_)

best_lr = grid_lr.best_estimator_

joblib.dump(best_lr, "machine_learning/artifacts/logistic_regression_tuned.pkl")

# Gird Search CV for XGBoost

grid_XG = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_params,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)

grid_XG.fit(X_train, y_train)

print(grid_XG.best_params_)
print(grid_XG.best_score_)

best_xgb = grid_XG.best_estimator_

joblib.dump(best_xgb, "machine_learning/artifacts/xgboost_tuned.pkl")

#######################################
# Prection And Accuracy
#######################################

pred_rf = best_rf.predict(X_test)

print("Gird Search CV RF Accuracy:", accuracy_score(y_test, pred_rf))
print(classification_report(y_test, pred_rf))

pred_lr = best_lr.predict(X_test)

print("Gird Search CV LR Accuracy:", accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

pred_xgb = best_xgb.predict(X_test)

print("Gird Search CV XGBoost Accuracy:", accuracy_score(y_test, pred_xgb))
print(classification_report(y_test, pred_xgb))

##############################################
# Random Search CV for Fine Tuning
##############################################
random_rf = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_params,
    n_iter=30,
    cv=cv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

random_rf.fit(X_train, y_train)

print(random_rf.best_params_)
print(random_rf.best_score_)

best_rrf = random_rf.best_estimator_

joblib.dump(best_rrf, "machine_learning/artifacts/randomforest_rtuned.pkl")

# Random Search CV for Logistic Regression

random_lr = RandomizedSearchCV(
    estimator=lr,
    param_distributions=lr_params,
    n_iter=30,
    cv=cv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

random_lr.fit(X_train, y_train)

print(random_lr.best_params_)
print(random_lr.best_score_)

best_rlr = random_lr.best_estimator_

joblib.dump(best_rlr, "machine_learning/artifacts/LogisticRegression_rtuned.pkl")

# Random Search CV for XGBoost

random_xgb = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=xgb_params,
    n_iter=30,
    cv=cv,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)

random_xgb.fit(X_train, y_train)

print(random_xgb.best_params_)
print(random_xgb.best_score_)

best_rxgb = random_xgb.best_estimator_

joblib.dump(best_rxgb, "machine_learning/artifacts/xgb_rtuned.pkl")

#######################################
# Prection And Accuracy
#######################################

pred_rrf = best_rrf.predict(X_test)

print("Random Search CV RF Accuracy:", accuracy_score(y_test, pred_rrf))
print(classification_report(y_test, pred_rrf))

pred_rlr = best_rlr.predict(X_test)

print("Random Search CV LR Accuracy:", accuracy_score(y_test, pred_rlr))
print(classification_report(y_test, pred_rlr))

pred_rxgb = best_rxgb.predict(X_test)

print("Random Search CV XGBoost Accuracy:", accuracy_score(y_test, pred_rxgb))
print(classification_report(y_test, pred_rxgb))