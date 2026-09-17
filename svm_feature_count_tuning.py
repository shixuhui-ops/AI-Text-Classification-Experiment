import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


SEED = 42
FEATURE_COUNTS = [5000, 10000, 20000, 40000, None]

data = pd.read_csv("train_data.csv")

X_train, X_valid, y_train, y_valid = train_test_split(
    data["text"],
    data["target"],
    test_size=0.2,
    random_state=SEED,
    stratify=data["target"],
)

results = []

for max_features in FEATURE_COUNTS:
    display_name = (
        "unlimited"
        if max_features is None
        else str(max_features)
    )

    print(
        f"\n正在训练 max_features={display_name}"
    )

    # 本轮只改变 max_features。
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
    )

    start = time.perf_counter()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)

    model = LinearSVC(
        C=1.0,
        dual=True,
        max_iter=5000,
        random_state=SEED,
    )

    model.fit(X_train_tfidf, y_train)

    train_predictions = model.predict(X_train_tfidf)
    valid_predictions = model.predict(X_valid_tfidf)

    elapsed = time.perf_counter() - start

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )
    valid_accuracy = accuracy_score(
        y_valid,
        valid_predictions,
    )

    train_macro_f1 = f1_score(
        y_train,
        train_predictions,
        average="macro",
    )
    valid_macro_f1 = f1_score(
        y_valid,
        valid_predictions,
        average="macro",
    )

    result = {
        "max_features": display_name,
        "actual_features": X_train_tfidf.shape[1],
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
        "train_macro_f1": train_macro_f1,
        "valid_macro_f1": valid_macro_f1,
        "generalization_gap": (
            train_macro_f1 - valid_macro_f1
        ),
        "time_seconds": elapsed,
    }

    results.append(result)

    print(
        "实际特征数:",
        X_train_tfidf.shape[1],
    )
    print(
        f"训练 Accuracy: {train_accuracy:.4f}"
    )
    print(
        f"验证 Accuracy: {valid_accuracy:.4f}"
    )
    print(
        f"训练 Macro-F1: {train_macro_f1:.4f}"
    )
    print(
        f"验证 Macro-F1: {valid_macro_f1:.4f}"
    )
    print(
        "泛化差距: "
        f"{train_macro_f1 - valid_macro_f1:.4f}"
    )
    print(f"运行时间: {elapsed:.2f} 秒")


results_df = pd.DataFrame(results)

results_df.to_csv(
    "svm_feature_count_results.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n=== 特征数量实验结果 ===")
print(results_df.to_string(index=False))


# 使用实际特征数作为横坐标
plot_df = results_df.sort_values(
    "actual_features"
)

plt.figure(figsize=(7, 4.5))

plt.plot(
    plot_df["actual_features"],
    plot_df["train_macro_f1"],
    marker="s",
    linewidth=2,
    label="Train Macro-F1",
)

plt.plot(
    plot_df["actual_features"],
    plot_df["valid_macro_f1"],
    marker="o",
    linewidth=2,
    label="Validation Macro-F1",
)

plt.xlabel("Number of TF-IDF Features")
plt.ylabel("Macro-F1")
plt.ylim(0.85, 1.01)
plt.title("Effect of TF-IDF Vocabulary Size")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    "svm_feature_count_curve.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()