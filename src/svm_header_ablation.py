import re
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


SEED = 42


def remove_email_header(text):
    """
    去除邮件头，只保留第一个空行之后的正文。

    如果文本没有明显空行，则保留原文本。
    """
    text = str(text)

    # 兼容 Windows 和 Unix 换行符
    match = re.search(r"\r?\n\r?\n", text)

    if match is None:
        return text

    body = text[match.end():].strip()

    # 防止异常情况下正文为空
    if body == "":
        return text

    return body


data = pd.read_csv("train_data.csv")

# 构造清洗后的文本。
data["body_text"] = data["text"].apply(remove_email_header)

X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
    data["text"],
    data["target"],
    test_size=0.2,
    random_state=SEED,
    stratify=data["target"],
)

# 使用完全相同的索引，确保两种文本输入对应同一批样本。
train_indices = X_train_raw.index
valid_indices = X_valid_raw.index

text_settings = {
    "raw_text": data.loc[train_indices, "text"],
    "body_only": data.loc[train_indices, "body_text"],
}

valid_text_settings = {
    "raw_text": data.loc[valid_indices, "text"],
    "body_only": data.loc[valid_indices, "body_text"],
}

results = []

for text_type in ["raw_text", "body_only"]:
    print(f"\n正在训练文本设置：{text_type}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2,
        sublinear_tf=True,
    )

    start = time.perf_counter()

    X_train_tfidf = vectorizer.fit_transform(
        text_settings[text_type]
    )
    X_valid_tfidf = vectorizer.transform(
        valid_text_settings[text_type]
    )

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
        "text_type": text_type,
        "feature_count": X_train_tfidf.shape[1],
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
        f"实际特征数: {X_train_tfidf.shape[1]}"
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
    "svm_header_ablation_results.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n=== 邮件头消融实验结果 ===")
print(results_df.to_string(index=False))


# 绘图
plt.figure(figsize=(7, 4.5))

x_positions = range(len(results_df))

plt.bar(
    [x - 0.18 for x in x_positions],
    results_df["valid_accuracy"],
    width=0.36,
    label="Validation Accuracy",
)

plt.bar(
    [x + 0.18 for x in x_positions],
    results_df["valid_macro_f1"],
    width=0.36,
    label="Validation Macro-F1",
)

plt.xticks(
    list(x_positions),
    results_df["text_type"],
)

plt.ylim(0.85, 1.0)
plt.ylabel("Score")
plt.title("Effect of Removing Email Headers")
plt.grid(axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    "svm_header_ablation.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()