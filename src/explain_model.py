
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Import helpers from preprocess
try:
    from .preprocess import load_telco_data, prepare_features_and_target
except ImportError:
    from preprocess import load_telco_data, prepare_features_and_target



THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "telco_churn_model.pkl")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}. "
            "Run src/train_model.py first from the project root."
        )
    clf = joblib.load(MODEL_PATH)
    return clf


def get_feature_names_from_preprocessor(preprocessor):
    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue

        if hasattr(transformer, "named_steps"):
            last_step = list(transformer.named_steps.values())[-1]

            if hasattr(last_step, "get_feature_names_out"):
                fn = last_step.get_feature_names_out(cols)
                feature_names.extend(fn)
            else:
                feature_names.extend(cols)
        else:
            feature_names.extend(cols)

    return feature_names


def generate_feature_importance_plot(top_n: int = 20):
    ensure_plots_dir()

    clf = load_model()
    df = load_telco_data(DATA_PATH)
    X, y = prepare_features_and_target(df)

    preprocessor = clf.named_steps["preprocessor"]
    model = clf.named_steps["model"]

    feature_names = get_feature_names_from_preprocessor(preprocessor)
    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        print(
            f"[WARN] Number of feature names ({len(feature_names)}) "
            f"does not match number of importances ({len(importances)})."
        )

    idx_sorted = np.argsort(importances)[::-1]  
    top_n = min(top_n, len(importances))
    idx_top = idx_sorted[:top_n]

    top_features = [feature_names[i] for i in idx_top]
    top_importances = importances[idx_top]

    
    plt.figure(figsize=(8, max(4, 0.3 * top_n)))
    plt.barh(range(top_n), top_importances[::-1])
    plt.yticks(range(top_n), top_features[::-1])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Most Important Features (RandomForest)")
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "shap_summary_bar.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    generate_feature_importance_plot(top_n=20)
