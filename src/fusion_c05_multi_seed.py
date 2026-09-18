from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"

TRAIN_FILE = DATA_DIR / "train_data.csv"
RESULT_FILE = RESULT_DIR / "fusion_c05_multi_seed_results.csv"
SUMMARY_FILE = RESULT_DIR / "fusion_c05_multi_seed_summary.csv"

SEEDS = [42, 123, 2025, 3407, 2026]
VALID_SIZE = 0.2
C_VALUE = 0.5


data = pd.read_csv(TRAIN_FILE)
data["text"] = data["text"].fillna("").astype(str)

results = []

for seed in SEEDS:
    print("=" * 60)
    print(f"正在运行 C={C_VALUE}，random_state={seed}")

    x_train, x_valid, y_train, y_valid = train_test_split(
        data["text"],
        data["target"],
        test_size=VALID_SIZE,
        random_state=seed,
        stratify=data["target"],
    )

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 1),
        min_df=2,
        sublinear_tf=True,
    )

    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=80000,
        sublinear_tf=True,
    )

    word_train = word_vectorizer.fit_transform(x_train)
    word_valid = word_vectorizer.transform(x_valid)

    char_train = char_vectorizer.fit_transform(x_train)
    char_valid = char_vectorizer.transform(x_valid)

    x_train_fusion = hstack(
        [word_train, char_train],
        format="csr",
    )

    x_valid_fusion = hstack(
        [word_valid, char_valid],
        format="csr",
    )

    model = LinearSVC(
        C=C_VALUE,
        dual=True,
        max_iter=5000,
        random_state=seed,
    )

    model.fit(x_train_fusion, y_train)

    train_prediction = model.predict(x_train_fusion)
    valid_prediction = model.predict(x_valid_fusion)

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
        "C": C_VALUE,
        "word_features": word_train.shape[1],
        "char_features": char_train.shape[1],
        "total_features": x_train_fusion.shape[1],
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
        "train_macro_f1": train_macro_f1,
        "valid_macro_f1": valid_macro_f1,
        "generalization_gap": (
            train_macro_f1 - valid_macro_f1
        ),
    }

    results.append(result)

    print(
        f"特征数: {x_train_fusion.shape[1]}"
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
        f"泛化差距: "
        f"{train_macro_f1 - valid_macro_f1:.4f}"
    )


results_df = pd.DataFrame(results)

RESULT_DIR.mkdir(exist_ok=True)

results_df.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8-sig",
)

summary_metrics = [
    "valid_accuracy",
    "valid_macro_f1",
    "generalization_gap",
]

summary_rows = []

for metric in summary_metrics:
    summary_rows.append({
        "metric": metric,
        "mean": results_df[metric].mean(),
        "std": results_df[metric].std(ddof=1),
        "min": results_df[metric].min(),
        "max": results_df[metric].max(),
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)


print("\n" + "=" * 60)
print("C=0.5 融合模型多随机种子结果")
print("=" * 60)

print(
    results_df[
        [
            "seed",
            "C",
            "valid_accuracy",
            "valid_macro_f1",
            "generalization_gap",
        ]
    ].to_string(index=False)
)

print("\n均值、标准差、最小值和最大值：")
print(summary_df.to_string(index=False))

accuracy_mean = results_df["valid_accuracy"].mean()
accuracy_std = results_df["valid_accuracy"].std(ddof=1)

f1_mean = results_df["valid_macro_f1"].mean()
f1_std = results_df["valid_macro_f1"].std(ddof=1)

print("\n报告可使用的汇总结果：")
print(
    f"Accuracy = "
    f"{accuracy_mean:.4f} ± {accuracy_std:.4f}"
)
print(
    f"Macro-F1 = "
    f"{f1_mean:.4f} ± {f1_std:.4f}"
)

print("\n结果文件：")
print(RESULT_FILE)
print(SUMMARY_FILE)