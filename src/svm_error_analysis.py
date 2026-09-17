import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

SEED = 42

# 只读取有标签数据；不读取测试集
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

print("验证 Macro-F1:", f1_score(
    y_valid, predictions, average="macro"
))
print("\n分类报告：")
print(classification_report(
    y_valid, predictions, digits=4, zero_division=0
))

labels = sorted(y_train.unique())
matrix = confusion_matrix(y_valid, predictions, labels=labels)

# 排除正确分类，只列出最常见的错误方向
mistakes = []
for row, true_label in enumerate(labels):
    for col, predicted_label in enumerate(labels):
        if true_label != predicted_label and matrix[row, col] > 0:
            mistakes.append({
                "true_label": true_label,
                "predicted_label": predicted_label,
                "count": int(matrix[row, col]),
            })

mistakes_df = pd.DataFrame(mistakes).sort_values(
    "count", ascending=False
)
mistakes_df.to_csv("svm_confusion_pairs.csv", index=False)

print("\n最常见的错误方向：")
print(mistakes_df.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels,
).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title("Linear SVM: Validation Confusion Matrix")
fig.tight_layout()
fig.savefig("svm_confusion_matrix.png", dpi=220)
plt.show()