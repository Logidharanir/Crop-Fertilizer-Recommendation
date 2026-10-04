
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
REPORTS = BASE / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

RANDOM_STATE = 42

def train_crop_model():
    df = pd.read_csv(DATA / "Crop_recommendation.csv")
    X = df.drop(columns=["label"])
    y = df["label"].str.title()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    joblib.dump(model, MODELS / "crop_model.joblib")

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    result = {
        "project": "Crop Recommendation",
        "algorithm": "StandardScaler + RandomForestClassifier",
        "samples": int(len(df)),
        "features": list(X.columns),
        "classes": sorted(y.unique().tolist()),
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "accuracy": float(accuracy),
        "classification_report": report
    }
    (REPORTS / "crop_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result

def train_fertilizer_model():
    df = pd.read_csv(DATA / "Fertilizer_Prediction.csv")
    df.columns = [c.strip() for c in df.columns]

    X = df.drop(columns=["Fertilizer Name"])
    y = df["Fertilizer Name"]

    categorical = ["Soil Type", "Crop Type"]
    numeric = [
        "Temparature", "Humidity", "Moisture",
        "Nitrogen", "Potassium", "Phosphorous"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    joblib.dump(model, MODELS / "fertilizer_model.joblib")

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    result = {
        "project": "Fertilizer Recommendation",
        "algorithm": "ColumnTransformer + StandardScaler/OneHotEncoder + RandomForestClassifier",
        "samples": int(len(df)),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "classes": sorted(y.unique().tolist()),
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "accuracy": float(accuracy),
        "classification_report": report
    }
    (REPORTS / "fertilizer_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result

if __name__ == "__main__":
    crop = train_crop_model()
    fert = train_fertilizer_model()
    print(f"Crop accuracy: {crop['accuracy']:.4f}")
    print(f"Fertilizer accuracy: {fert['accuracy']:.4f}")
    print("Models saved in models/")
