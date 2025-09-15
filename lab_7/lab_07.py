import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Regressors (for A4 template)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Clustering (for A5 template)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


def main():
    # ---------------- Load Dataset ----------------
    filename = "C:/Users/hp/OneDrive - Amrita vishwa vidyapeetham/Documents/ML_assignments/lab_3/original (1).xlsx"
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(filename)
    elif filename.endswith(".csv"):
        df = pd.read_csv(filename)
    else:
        raise ValueError("Unsupported file format")

    # Columns
    text_col = "student"
    target_col = df.columns[-1]

    # ---------------- Preprocessing ----------------
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    typo_map = {
        "hgh": "high", "hgih": "high", "hig": "high",
        "loww": "low", "lo": "low",
        "mediu": "medium", "med": "medium"
    }
    df[target_col] = df[target_col].replace(typo_map)
    df[target_col] = df[target_col].replace(["na", "nan", "none", "n/a"], np.nan)
    df = df.dropna(subset=[target_col])

    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target_col].astype(str))

    # Remove rare classes
    counts = Counter(y)
    keep_mask = np.isin(y, [cls for cls, cnt in counts.items() if cnt >= 2])
    df = df[keep_mask]
    y = y[keep_mask]

    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=50)
    X_tfidf = vectorizer.fit_transform(df[text_col].astype(str))

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

    # ==========================================================
    # A2: RandomizedSearchCV for Decision Tree
    # ==========================================================
    print("\nA2: Hyperparameter Tuning (Decision Tree with RandomizedSearchCV)")
    param_dist = {"max_depth": [2, 4, 6, 8, None],
                  "min_samples_split": [2, 5, 10]}
    rs = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=5, cv=3, random_state=42, n_jobs=-1
    )
    rs.fit(X_train, y_train)
    best_dt = rs.best_estimator_
    print("Best Parameters:", rs.best_params_)
    print("Best Cross-Validation Score:", rs.best_score_)

    # ==========================================================
    # A3: Multiple Classifiers
    # ==========================================================
    print("\nA3: Multiple Classifiers Comparison")
    models = {
        "DecisionTree": best_dt,
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "NaiveBayes": MultinomialNB(),
        "MLP": MLPClassifier(max_iter=500, random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }

    results = []
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append([name, acc])
        print(f"\n{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=le_target.classes_))

    results_df = pd.DataFrame(results, columns=["Model", "Test Accuracy"])
    print("\nComparison Table:\n", results_df)

    # Plot comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Model", y="Test Accuracy", data=results_df)
    plt.xticks(rotation=45)
    plt.title("Model Comparison (Test Accuracy)")
    plt.show()

    # ==========================================================
    # A4: Regression (Template Example)
    # ==========================================================
    print("\nA4: Regression Models (Template Example with Random Data)")
    X_reg = np.random.rand(100, 5)
    y_reg = np.random.rand(100)
    regressors = {
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "XGBRegressor": XGBRegressor(random_state=42),
        "SVR": SVR(),
        "MLPRegressor": MLPRegressor(max_iter=500, random_state=42),
        "CatBoostRegressor": CatBoostRegressor(verbose=0, random_state=42)
    }
    for name, reg in regressors.items():
        reg.fit(X_reg, y_reg)
        score = reg.score(X_reg, y_reg)
        print(f"{name} R^2 Score: {score:.4f}")

    # ==========================================================
    # A5: Clustering (Template Example)
    # ==========================================================
    print("\nA5: Clustering Algorithms (Template Example with Random Data)")
    X_cluster = np.random.rand(50, 2)
    clustering_models = {
        "KMeans": KMeans(n_clusters=3, random_state=42),
        "Agglomerative": AgglomerativeClustering(n_clusters=3),
        "DBSCAN": DBSCAN(eps=0.2, min_samples=3)
    }
    for name, cluster in clustering_models.items():
        labels = cluster.fit_predict(X_cluster)
        print(f"{name} cluster labels:", np.unique(labels))


if __name__ == "__main__":
    main()
