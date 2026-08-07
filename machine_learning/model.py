from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_model(model_name):

    if model_name == "random_forest":

        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

    elif model_name == "logistic_regression":

        return LogisticRegression(
            max_iter=1000,
            random_state=42
        )

    elif model_name == "xgboost":

        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )

    else:

        raise ValueError("Invalid model selected")