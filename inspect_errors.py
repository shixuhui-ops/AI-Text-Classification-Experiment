import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


SEED = 42
TOP_TERMS = 12
EXAMPLES_PER_PAIR = 3

data = pd.read_csv("train_data.csv")

X_train, X_valid, y_train, y_valid = train_test_split(
    data["text"],
    data["target"],
    test_size=0.2,
    random_state=SEED,
    stratify=data["target"],
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    min_df=2,
    sublinear_tf=True,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_valid_vec = vectorizer.transform(X_valid)

model = LinearSVC(
    C=1.0,
    dual=True,
    max_iter=5000,
    random_state=SEED,
)

model.fit(X_train_vec, y_train)
predictions = model.predict(X_valid_vec)
decision_scores = model.decision_function(X_valid_vec)

feature_names = vectorizer.get_feature_names_out()


# 1. 查看每个类别权重最高的代表词
print("=== 各类别最具代表性的词 ===")

top_terms_rows = []

for class_position, class_label in enumerate(model.classes_):
    top_indices = np.argsort(
        model.coef_[class_position]
    )[-TOP_TERMS:][::-1]

    terms = feature_names[top_indices]

    print(
        f"类别 {class_label}: "
        + ", ".join(terms)
    )

    for rank, feature_index in enumerate(
        top_indices,
        start=1,
    ):
        top_terms_rows.append({
            "class": int(class_label),
            "rank": rank,
            "term": feature_names[feature_index],
            "weight": model.coef_[
                class_position,
                feature_index,
            ],
        })

pd.DataFrame(top_terms_rows).to_csv(
    "svm_top_terms.csv",
    index=False,
    encoding="utf-8-sig",
)


# 2. 整理验证集预测结果
analysis = pd.DataFrame({
    "original_index": X_valid.index,
    "text": X_valid.values,
    "true_label": y_valid.values,
    "predicted_label": predictions,
})

# 决策分数最高和第二高之间的差距
sorted_scores = np.sort(decision_scores, axis=1)

analysis["confidence_margin"] = (
    sorted_scores[:, -1] - sorted_scores[:, -2]
)

analysis["correct"] = (
    analysis["true_label"]
    == analysis["predicted_label"]
)

analysis.to_csv(
    "svm_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)


# 3. 输出最常见混淆方向的文本案例
confusion_pairs = [
    (7, 2),
    (2, 1),
    (2, 7),
]

output_lines = []

for true_label, predicted_label in confusion_pairs:
    subset = analysis[
        (analysis["true_label"] == true_label)
        & (
            analysis["predicted_label"]
            == predicted_label
        )
    ].sort_values(
        "confidence_margin",
        ascending=False,
    )

    heading = (
        f"\n{'=' * 70}\n"
        f"真实类别 {true_label}，"
        f"预测为 {predicted_label}，"
        f"共 {len(subset)} 条\n"
        f"{'=' * 70}"
    )

    print(heading)
    output_lines.append(heading)

    for number, (_, row) in enumerate(
        subset.head(EXAMPLES_PER_PAIR).iterrows(),
        start=1,
    ):
        # 合并空白字符，避免打印内容过长
        cleaned_text = " ".join(
            str(row["text"]).split()
        )

        excerpt = cleaned_text[:1200]

        block = (
            f"\n案例 {number}\n"
            f"原始索引: {row['original_index']}\n"
            f"置信差距: "
            f"{row['confidence_margin']:.4f}\n"
            f"文本节选:\n{excerpt}\n"
        )

        print(block)
        output_lines.append(block)

with open(
    "svm_error_examples.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write("\n".join(output_lines))

print("\n已生成：")
print("svm_top_terms.csv")
print("svm_validation_predictions.csv")
print("svm_error_examples.txt")