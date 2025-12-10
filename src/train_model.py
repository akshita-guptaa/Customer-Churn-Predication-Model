
import os

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

try:
    from .preprocess import (
        load_telco_data,
        prepare_features_and_target,
        split_feature_types,
        build_preprocessor,
    )
except ImportError:
    from preprocess import (
        load_telco_data,
        prepare_features_and_target,
        split_feature_types,
        build_preprocessor,
    )

DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "telco_churn_model.pkl")
PLOTS_DIR = "plots"


def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def build_pipeline(X_sample):
    
    cat_features, num_features = split_feature_types(X_sample)
    preprocessor = build_preprocessor(cat_features, num_features)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_and_evaluate():
    ensure_dirs()

   
    print(f"Loading data from: {DATA_PATH}")
    df = load_telco_data(DATA_PATH)
    X, y = prepare_features_and_target(df)

   
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    
    clf = build_pipeline(X_train)

    
    print("Training model...")
    clf.fit(X_train, y_train)

    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n=== Evaluation on test set ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    
    plt.figure(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve")
    roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve plot to: {roc_path}")

    
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved trained model pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
