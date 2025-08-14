# -*- coding: utf-8 -*-
"""
Decision Tree with TF-IDF (student column only) and Clean Target Labels
Subject: 23CSE301
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# -------------------- A1: Entropy --------------------
def calculate_entropy(y):
    if len(y) == 0:
        return 0
    value_counts = Counter(y)
    total_samples = len(y)
    entropy = 0
    for count in value_counts.values():
        probability = count / total_samples
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy

# -------------------- A2: Gini Index --------------------
def calculate_gini_index(y):
    if len(y) == 0:
        return 0
    value_counts = Counter(y)
    total_samples = len(y)
    gini = 1.0
    for count in value_counts.values():
        probability = count / total_samples
        gini -= probability ** 2
    return gini

# -------------------- A3: Information Gain --------------------
def calculate_information_gain(X, y, feature_index):
    parent_entropy = calculate_entropy(y)
    feature_values = np.unique(X[:, feature_index])
    weighted_entropy = 0
    total_samples = len(y)
    for value in feature_values:
        mask = X[:, feature_index] == value
        subset_y = y[mask]
        if len(subset_y) > 0:
            subset_entropy = calculate_entropy(subset_y)
            weight = len(subset_y) / total_samples
            weighted_entropy += weight * subset_entropy
    return parent_entropy - weighted_entropy

def rank_information_gain(X_dense, y, feature_names, top_n=10):
    gains = []
    for i in range(X_dense.shape[1]):
        gain = calculate_information_gain(X_dense, y, i)
        gains.append((feature_names[i], gain))
    gains_sorted = sorted(gains, key=lambda x: x[1], reverse=True)
    print(f"\nTop {top_n} features by Information Gain:")
    for feature, gain in gains_sorted[:top_n]:
        print(f"{feature:20} {gain:.4f}")
    return gains_sorted

# -------------------- A5: Decision Tree Build --------------------
def build_decision_tree(X, y, max_depth=4):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    return clf

# -------------------- A6: Decision Tree Plot --------------------
def plot_decision_tree_sklearn(clf, feature_names, class_names):
    plt.figure(figsize=(20, 12))
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=8)
    plt.title("Decision Tree Visualization")
    plt.show()

# -------------------- A7: PCA-based Decision Boundary --------------------
def plot_decision_boundary_pca(X_tfidf, y, class_names):
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_tfidf.toarray())
    clf_2d = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    clf_2d.fit(X_2d, y)
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=[class_names[i] for i in y],
                    palette="deep", edgecolor='k', s=80)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Decision Boundary (PCA-reduced TF-IDF space)")
    plt.show()

# -------------------- MAIN --------------------
def main():
    # Step 1: Load dataset directly from local path
    filename = "C:/Users/hp/OneDrive - Amrita vishwa vidyapeetham/Documents/ML_assignments/lab_3/original (1).xlsx"

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(filename)
    elif filename.endswith(".csv"):
        df = pd.read_csv(filename)
    else:
        raise ValueError("Unsupported file format. Please use .xlsx, .xls, or .csv")

    # Step 2: Identify columns
    text_col = "student"  # fixed as per requirement
    target_col = df.columns[-1]  # assumes last column is target

    # Step 3: Clean target labels
    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    typo_map = {
        "hgh": "high", "hgih": "high", "hig": "high",
        "loww": "low", "lo": "low",
        "mediu": "medium", "med": "medium"
    }
    df[target_col] = df[target_col].replace(typo_map)
    df[target_col] = df[target_col].replace(["na", "nan", "none", "n/a"], np.nan)
    df = df.dropna(subset=[target_col])

    # Step 4: Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target_col].astype(str))

    # Step 5: Remove rare classes (<2 samples)
    counts = Counter(y)
    keep_mask = np.isin(y, [cls for cls, cnt in counts.items() if cnt >= 2])
    df = df[keep_mask]
    y = y[keep_mask]

    # Step 6: TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=50)
    X_tfidf = vectorizer.fit_transform(df[text_col].astype(str))
    feature_names = vectorizer.get_feature_names_out()
    X_dense = X_tfidf.toarray()

    # ---------------- A1 ----------------
    print("\nA1: Entropy")
    print(f"Entropy of target: {calculate_entropy(y):.4f}")

    # ---------------- A2 ----------------
    print("\nA2: Gini Index")
    print(f"Gini index of target: {calculate_gini_index(y):.4f}")

    # ---------------- A3 ----------------
    print("\nA3: Information Gain Ranking")
    _ = rank_information_gain(X_dense, y, feature_names, top_n=10)

    # ---------------- A5 ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.3, random_state=42
    )
    clf = build_decision_tree(X_train, y_train, max_depth=4)

    y_pred = clf.predict(X_test)
    present_classes = np.unique(y_test)
    present_class_names = le_target.inverse_transform(present_classes)

    print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=present_classes,
        target_names=present_class_names
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_class_names,
                yticklabels=present_class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # ---------------- A6 ----------------
    print("\nA6: Decision Tree Visualization")
    plot_decision_tree_sklearn(clf, feature_names, le_target.classes_)

    # ---------------- A7 ----------------
    print("\nA7: Decision Boundary in PCA-reduced space")
    plot_decision_boundary_pca(X_tfidf, y, le_target.classes_)

# Run locally
if __name__ == "__main__":
    main()
