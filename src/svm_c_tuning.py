import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


SEED = 42
C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]


# 1. 读取数据
data = pd.read_csv("train_data.csv")

# 2. 保持与之前完全相同的分层划分
X_train, X_valid, y_train, y_valid = train_test_split(
    data["text"],
    data["target"],
    test_size=0.2,
    random_state=SEED,
    stratify=data["target"],
)

print("训练集大小:", len(X_train))
print("验证集大小:", len(X_valid))

# 3. 保持与之前完全相同的 TF-IDF 设置
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    max_features=5000,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)

print("TF-IDF 特征数:", X_train_tfidf.shape[1])

results = []

# 4. 只改变 C
for c_value in C_VALUES:
    print(f"\n正在训练 LinearSVC(C={c_value})")

    start = time.perf_counter()

    model = LinearSVC(
        C=c_value,
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

    results.append(
        {
            "C": c_value,
            "train_accuracy": train_accuracy,
            "valid_accuracy": valid_accuracy,
            "train_macro_f1": train_macro_f1,
            "valid_macro_f1": valid_macro_f1,
            "generalization_gap": (
                train_macro_f1 - valid_macro_f1
            ),
            "iterations": model.n_iter_,
            "time_seconds": elapsed,
        }
    )

    print(f"训练 Accuracy: {train_accuracy:.4f}")
    print(f"验证 Accuracy: {valid_accuracy:.4f}")
    print(f"训练 Macro-F1: {train_macro_f1:.4f}")
    print(f"验证 Macro-F1: {valid_macro_f1:.4f}")
    print(
        f"泛化差距: "
        f"{train_macro_f1 - valid_macro_f1:.4f}"
    )
    print(f"迭代次数: {model.n_iter_}")
    print(f"运行时间: {elapsed:.2f} 秒")


# 5. 保存结果
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="valid_macro_f1",
    ascending=False,
).reset_index(drop=True)

results_df.to_csv(
    "svm_c_tuning_results.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n=== C 参数实验排名 ===")
print(results_df.to_string(index=False))


# 6. 绘制 C 与验证 Macro-F1 的关系
plot_df = results_df.sort_values("C")

plt.figure(figsize=(7, 4.5))

plt.plot(
    plot_df["C"],
    plot_df["valid_macro_f1"],
    marker="o",
    linewidth=2,
    label="Validation Macro-F1",
)

plt.plot(
    plot_df["C"],
    plot_df["train_macro_f1"],
    marker="s",
    linewidth=2,
    label="Train Macro-F1",
)

plt.xscale("log")
plt.xlabel("C, logarithmic scale")
plt.ylabel("Macro-F1")
plt.ylim(0.7, 1.02)
plt.title("Effect of SVM C on Macro-F1")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    "svm_c_tuning_curve.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()