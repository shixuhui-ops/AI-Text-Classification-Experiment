import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

SEED = 42

# 1. 读取有标签数据
data = pd.read_csv("train_data.csv")

# 2. 使用与前两次实验完全相同的分层划分
X_train, X_valid, y_train, y_valid = train_test_split(
    data["text"],
    data["target"],
    test_size=0.2,
    random_state=SEED,
    stratify=data["target"],
)

print("训练集大小:", len(X_train))
print("验证集大小:", len(X_valid))

# 3. 使用与前两次实验完全相同的 TF-IDF 设置
# 向量器只在训练集上拟合，以避免数据泄露
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    max_features=5000,
)

start = time.perf_counter()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)

# 4. 本轮唯一主要变量：将分类器改为线性 SVM
model = LinearSVC(
    C=1.0,
    dual=True,
    max_iter=5000,
    random_state=SEED,
)

model.fit(X_train_tfidf, y_train)

# 5. 分别预测训练集和验证集
train_predictions = model.predict(X_train_tfidf)
valid_predictions = model.predict(X_valid_tfidf)

elapsed = time.perf_counter() - start

# 6. 计算评估指标
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

generalization_gap = train_macro_f1 - valid_macro_f1

# 7. 输出实验配置和结果
print("\n=== 线性 SVM 基线配置 ===")
print("随机种子:", SEED)
print("验证集比例: 0.2")
print("TF-IDF: unigram, max_features=5000")
print("模型: LinearSVC(C=1.0)")

print("\n=== 主要结果 ===")
print(f"训练 Accuracy: {train_accuracy:.4f}")
print(f"验证 Accuracy: {valid_accuracy:.4f}")
print(f"训练 Macro-F1: {train_macro_f1:.4f}")
print(f"验证 Macro-F1: {valid_macro_f1:.4f}")
print(f"泛化差距: {generalization_gap:.4f}")
print(f"迭代次数: {model.n_iter_}")
print(f"运行时间: {elapsed:.2f} 秒")

print("\n=== 验证集分类报告 ===")
print(
    classification_report(
        y_valid,
        valid_predictions,
        digits=4,
        zero_division=0,
    )
)