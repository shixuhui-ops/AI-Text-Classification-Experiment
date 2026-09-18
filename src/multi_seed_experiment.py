from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


# =========================
# 路径和固定配置
# =========================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"

TRAIN_FILE = DATA_DIR / "train_data.csv"
RESULT_FILE = RESULT_DIR / "multi_seed_results.csv"

SEEDS = [42, 123, 2025, 3407, 2026]
VALID_SIZE = 0.2


# =========================
# 读取数据
# =========================

train_data = pd.read_csv(TRAIN_FILE)

train_data["text"] = (
    train_data["text"]
    .fillna("")
    .astype(str)
)

texts = train_data["text"]
labels = train_data["target"]


# =========================
# 多随机种子重复实验
# =========================

results = []

for seed in SEEDS:
    print("=" * 60)
    print(f"正在运行 random_state={seed}")

    x_train, x_valid, y_train, y_valid = (
        train_test_split(
            texts,
            labels,
            test_size=VALID_SIZE,
            random_state=seed,
            stratify=labels,
        )
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2,
        sublinear_tf=True,
    )

    # 只在当前随机种子的训练子集上拟合
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_valid_tfidf = vectorizer.transform(x_valid)

    model = LinearSVC(
        C=1.0,
        dual=True,
        max_iter=5000,
        random_state=seed,
    )

    model.fit(x_train_tfidf, y_train)

    train_prediction = model.predict(x_train_tfidf)
    valid_prediction = model.predict(x_valid_tfidf)

    train_accuracy = accuracy_score(
        y_train,
        train_prediction,
    )
    valid_accuracy = accuracy_score(
        y_valid,
        valid_prediction,
    )

    train_macro_f1 = f1_score(
        y_train,
        train_prediction,
        average="macro",
    )
    valid_macro_f1 = f1_score(
        y_valid,
        valid_prediction,
        average="macro",
    )

    result = {
        "seed": seed,
        "train_samples": len(x_train),
        "valid_samples": len(x_valid),
        "feature_count": x_train_tfidf.shape[1],
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
        "train_macro_f1": train_macro_f1,
        "valid_macro_f1": valid_macro_f1,
        "generalization_gap": (
            train_macro_f1 - valid_macro_f1
        ),
    }

    results.append(result)

    print(f"特征数: {x_train_tfidf.shape[1]}")
    print(f"训练 Accuracy: {train_accuracy:.4f}")
    print(f"验证 Accuracy: {valid_accuracy:.4f}")
    print(f"训练 Macro-F1: {train_macro_f1:.4f}")
    print(f"验证 Macro-F1: {valid_macro_f1:.4f}")
    print(
        "泛化差距: "
        f"{train_macro_f1 - valid_macro_f1:.4f}"
    )


# =========================
# 保存单次结果
# =========================

results_df = pd.DataFrame(results)

RESULT_DIR.mkdir(exist_ok=True)

results_df.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 输出均值和标准差
# =========================

metric_columns = [
    "valid_accuracy",
    "valid_macro_f1",
    "generalization_gap",
]

summary_rows = []

for metric in metric_columns:
    summary_rows.append({
        "metric": metric,
        "mean": results_df[metric].mean(),
        "std": results_df[metric].std(ddof=1),
        "min": results_df[metric].min(),
        "max": results_df[metric].max(),
    })

summary_df = pd.DataFrame(summary_rows)

summary_file = RESULT_DIR / "multi_seed_summary.csv"

summary_df.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 控制台输出
# =========================

print("\n" + "=" * 60)
print("多随机种子实验结果")
print("=" * 60)

print(
    results_df[
        [
            "seed",
            "valid_accuracy",
            "valid_macro_f1",
            "generalization_gap",
        ]
    ].to_string(index=False)
)

print("\n均值、标准差、最小值和最大值：")
print(summary_df.to_string(index=False))

valid_f1_mean = results_df["valid_macro_f1"].mean()
valid_f1_std = results_df["valid_macro_f1"].std(ddof=1)

valid_acc_mean = results_df["valid_accuracy"].mean()
valid_acc_std = results_df["valid_accuracy"].std(ddof=1)

print("\n报告可使用的汇总结果：")
print(
    f"Accuracy = "
    f"{valid_acc_mean:.4f} ± {valid_acc_std:.4f}"
)
print(
    f"Macro-F1 = "
    f"{valid_f1_mean:.4f} ± {valid_f1_std:.4f}"
)

print("\n结果文件：")
print(RESULT_FILE)
print(summary_file)