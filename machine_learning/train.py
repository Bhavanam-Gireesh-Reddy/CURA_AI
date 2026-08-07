import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "indian_liver_patient.csv"
)


ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)


os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)



# Load Data

df = pd.read_csv(DATA_PATH)


print(
    "Dataset Shape:",
    df.shape
)



# Cleaning

df["Gender"] = df["Gender"].map(
    {
        "Male":1,
        "Female":0
    }
)


df["Dataset"] = df["Dataset"].map(
    {
        1:1,
        2:0
    }
)


df = df.drop_duplicates()


df = df.fillna(
    df.median(numeric_only=True)
)



# Features and Target

X = df.drop(
    "Dataset",
    axis=1
)


y = df["Dataset"]



# Scaling

scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)


joblib.dump(
    scaler,
    os.path.join(
        ARTIFACT_DIR,
        "scaler.pkl"
    )
)



# Split Data

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)



models = {


"random_forest":

RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
),



"logistic_regression":

LogisticRegression(
    max_iter=1000
),



"xgboost":

XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)


}



for name, model in models.items():

    print(
        f"Training {name}"
    )


    model.fit(
        X_train,
        y_train
    )


    accuracy = model.score(
        X_test,
        y_test
    )


    print(
        "Accuracy:",
        accuracy
    )


    joblib.dump(
        model,
        os.path.join(
            ARTIFACT_DIR,
            f"{name}.pkl"
        )
    )


print(
    "All models saved successfully"
)
print(X.columns)
print(len(X.columns))