import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

SEED = 42

data = pd.read_csv("train_data.csv")

X_train, X_valid, y_train, y_valid = train_test_split(
    data["text"],
    data["target"],
    test_size=0.2,
    random_state=SEED,
    stratify=data["target"],
)

# 与朴素贝叶斯基线完全一致
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    max_features=5000,
)

start = time.perf_counter()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)

# 本轮唯一主要变量：将分类器改为逻辑回归
model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=SEED,
)

model.fit(X_train_tfidf, y_train)

train_predictions = model.predict(X_train_tfidf)
valid_predictions = model.predict(X_valid_tfidf)

elapsed = time.perf_counter() - start

train_accuracy = accuracy_score(y_train, train_predictions)
valid_accuracy = accuracy_score(y_valid, valid_predictions)
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

print("=== 逻辑回归基线配置 ===")
print("随机种子:", SEED)
print("验证集比例: 0.2")
print("TF-IDF: unigram, max_features=5000")
print("模型: LogisticRegression(C=1.0)")

print("\n=== 主要结果 ===")
print(f"训练 Accuracy: {train_accuracy:.4f}")
print(f"验证 Accuracy: {valid_accuracy:.4f}")
print(f"训练 Macro-F1: {train_macro_f1:.4f}")
print(f"验证 Macro-F1: {valid_macro_f1:.4f}")
print(f"泛化差距: {train_macro_f1 - valid_macro_f1:.4f}")
print(f"迭代次数: {model.n_iter_.tolist()}")
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