import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


SEED = 42

data = pd.read_csv("train_data.csv")

X_train, X_valid, y_train, y_valid = train_test_split(
    data["text"],
    data["target"],
    test_size=0.2,
    random_state=SEED,
    stratify=data["target"],
)

feature_settings = {
    "unigram": (1, 1),
    "unigram_bigram": (1, 2),
}

results = []

for feature_name, ngram_range in feature_settings.items():
    print(f"\n正在训练特征设置：{feature_name}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        max_features=20000,
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

    train_f1 = f1_score(
        y_train,
        train_predictions,
        average="macro",
    )
    valid_f1 = f1_score(
        y_valid,
        valid_predictions,
        average="macro",
    )

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )
    valid_accuracy = accuracy_score(
        y_valid,
        valid_predictions,
    )

    result = {
        "feature": feature_name,
        "ngram_range": str(ngram_range),
        "feature_count": X_train_tfidf.shape[1],
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
        "train_macro_f1": train_f1,
        "valid_macro_f1": valid_f1,
        "generalization_gap": train_f1 - valid_f1,
        "time_seconds": elapsed,
    }

    results.append(result)

    print("特征数:", X_train_tfidf.shape[1])
    print(f"训练 Accuracy: {train_accuracy:.4f}")
    print(f"验证 Accuracy: {valid_accuracy:.4f}")
    print(f"训练 Macro-F1: {train_f1:.4f}")
    print(f"验证 Macro-F1: {valid_f1:.4f}")
    print(f"泛化差距: {train_f1 - valid_f1:.4f}")
    print(f"运行时间: {elapsed:.2f} 秒")


results_df = pd.DataFrame(results)

results_df.to_csv(
    "svm_feature_compare_results.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n=== 特征表示对比 ===")
print(results_df.to_string(index=False))

# 绘制对比图
x = range(len(results_df))

plt.figure(figsize=(7, 4.5))

plt.bar(
    [i - 0.18 for i in x],
    results_df["valid_accuracy"],
    width=0.36,
    label="Validation Accuracy",
)

plt.bar(
    [i + 0.18 for i in x],
    results_df["valid_macro_f1"],
    width=0.36,
    label="Validation Macro-F1",
)

plt.xticks(list(x), results_df["feature"])
plt.ylim(0.7, 1.0)
plt.ylabel("Score")
plt.title("Effect of N-gram Features")
plt.grid(axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    "svm_feature_compare.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()