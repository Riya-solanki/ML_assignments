# stacking_pipeline_lime_inline.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# External libs
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lime.lime_text import LimeTextExplainer
from IPython.display import display

# ---------------- Load Dataset ----------------
def load_df(filename, text_col="student"):
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(filename)
    elif filename.endswith(".csv"):
        df = pd.read_csv(filename)
    else:
        raise ValueError("Unsupported file format")

    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found. Available columns: {df.columns.tolist()}")
    return df

# ---------------- Main ----------------
def main(filename,
         text_col="student",
         target_col=None,
         min_class_count=2,
         test_size=0.3,
         random_state=42):

    df = load_df(filename, text_col=text_col)

    if target_col is None:
        target_col = df.columns[-1]

    # Clean target column
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    typo_map = {
        "hgh": "high", "hgih": "high", "hig": "high",
        "loww": "low", "lo": "low",
        "mediu": "medium", "med": "medium"
    }
    df[target_col] = df[target_col].replace(typo_map)
    df[target_col] = df[target_col].replace(["na", "nan", "none", "n/a"], np.nan)
    df = df.dropna(subset=[target_col, text_col])

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].astype(str))

    # Remove rare classes
    counts = Counter(y)
    allowed = {cls for cls, cnt in counts.items() if cnt >= min_class_count}
    mask = [yi in allowed for yi in y]
    df = df.loc[mask].reset_index(drop=True)
    y = y[np.array(mask)]

    if len(df) == 0:
        raise ValueError("No data left after filtering rare classes.")

    X_text = df[text_col].astype(str).values

    # Train-Test Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y)) > 1 else None
    )

    # ---------------- Base Estimators ----------------
    base_estimators = [
        ("dt", DecisionTreeClassifier(random_state=random_state)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ("nb", MultinomialNB()),
        ("mlp", MLPClassifier(max_iter=500, random_state=random_state)),
        ("ada", AdaBoostClassifier(random_state=random_state)),
        ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=random_state)),
        ("cat", CatBoostClassifier(verbose=0, random_state=random_state))
    ]

    # ---------------- Meta Estimators ----------------
    meta_estimators = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=random_state),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=random_state)
    }

    results = []

    for meta_name, meta_est in meta_estimators.items():
        print(f"\nTraining StackingClassifier with meta estimator: {meta_name}")

        stacking = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_est,
            cv=5,
            n_jobs=-1
        )

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500)),
            ("clf", stacking)
        ])

        pipeline.fit(X_train_text, y_train)
        y_pred = pipeline.predict(X_test_text)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy ({meta_name}): {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        results.append((meta_name, acc))

    # ---------------- Results Table ----------------
    results_df = pd.DataFrame(results, columns=["Meta Estimator", "Test Accuracy"]).sort_values("Test Accuracy", ascending=False)
    print("\nComparison of Meta-Estimators:\n", results_df)

    plt.figure(figsize=(8,5))
    sns.barplot(data=results_df, x="Meta Estimator", y="Test Accuracy")
    plt.title("Stacking Meta-Estimator Comparison")
    plt.ylim(0,1)
    plt.show()

    # ---------------- Best Pipeline ----------------
    best_meta = results_df.iloc[0]["Meta Estimator"]
    print(f"\nBest meta estimator chosen: {best_meta}")
    best_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500)),
        ("clf", StackingClassifier(estimators=base_estimators, final_estimator=meta_estimators[best_meta], cv=5, n_jobs=-1))
    ])
    best_pipeline.fit(X_train_text, y_train)

    # ---------------- LIME Explanations ----------------
    explainer = LimeTextExplainer(class_names=list(le.classes_))

    print("\nLIME Explanations:")
    sample_idx = np.random.choice(range(len(X_test_text)), size=min(3, len(X_test_text)), replace=False)
    for idx in sample_idx:
        text_instance = X_test_text[idx]
        pred_label = best_pipeline.predict([text_instance])[0]
        print(f"\nExample index: {idx}")
        print(f"Text: {text_instance[:200]}...")
        print(f"Predicted Label: {le.inverse_transform([pred_label])[0]}")

        exp = explainer.explain_instance(text_instance, best_pipeline.predict_proba, num_features=10, labels=[pred_label])
        exp.show_in_notebook(text=True)   # ✅ shows inline

    return results_df, best_pipeline


# ---------------- Run ----------------
if __name__ == "__main__":
    filename = r"C:\Users\hp\OneDrive - Amrita vishwa vidyapeetham\Documents\SEM V\MACHINE_LEARNING\ML_assignments\lab_3\original (1).xlsx"
    results, pipeline = main(filename)
    print("\nDone.")
