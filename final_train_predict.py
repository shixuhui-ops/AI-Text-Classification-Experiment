import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


SEED = 42

TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data_unlabeled.csv"
PREDICTION_FILE = "predictions.csv"
CONFIG_FILE = "final_model_config.json"


# =========================
# 1. 读取数据
# =========================

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

if "text" not in train_data.columns:
    raise ValueError("训练集缺少 text 列")

if "target" not in train_data.columns:
    raise ValueError("训练集缺少 target 列")

if "text" not in test_data.columns:
    raise ValueError("测试集缺少 text 列")

train_data["text"] = (
    train_data["text"]
    .fillna("")
    .astype(str)
)

test_data["text"] = (
    test_data["text"]
    .fillna("")
    .astype(str)
)


# =========================
# 2. 锁定验证阶段选择的配置
# =========================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    min_df=2,
    sublinear_tf=True,
)

model = LinearSVC(
    C=1.0,
    dual=True,
    max_iter=5000,
    random_state=SEED,
)


# =========================
# 3. 使用全部有标签数据拟合
# =========================

print("正在使用全部训练数据拟合 TF-IDF……")

X_train_tfidf = vectorizer.fit_transform(
    train_data["text"]
)

print(
    "训练样本数:",
    X_train_tfidf.shape[0],
)

print(
    "TF-IDF 特征数:",
    X_train_tfidf.shape[1],
)

print("正在训练最终 LinearSVC……")

model.fit(
    X_train_tfidf,
    train_data["target"],
)


# =========================
# 4. 转换测试集并预测
# =========================

print("正在转换测试集……")

X_test_tfidf = vectorizer.transform(
    test_data["text"]
)

predictions = model.predict(
    X_test_tfidf
)


# =========================
# 5. 保存预测文件
# =========================

# 老师示例要求：无索引、无表头，每行一个类别标签。
prediction_df = pd.DataFrame(predictions)

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False,
    header=False,
)

print("\n预测文件已生成:", PREDICTION_FILE)
print("测试样本数:", len(predictions))
print("预测标签数:", len(predictions))
print(
    "预测类别分布:"
)
print(
    prediction_df[0]
    .value_counts()
    .sort_index()
)


# =========================
# 6. 保存最终配置记录
# =========================

config = {
    "seed": SEED,
    "train_file": TRAIN_FILE,
    "test_file": TEST_FILE,
    "model": "LinearSVC",
    "C": 1.0,
    "max_iter": 5000,
    "feature_method": "TF-IDF",
    "lowercase": True,
    "ngram_range": [1, 1],
    "min_df": 2,
    "sublinear_tf": True,
    "text_cleaning": "raw_text",
    "train_samples": int(len(train_data)),
    "test_samples": int(len(test_data)),
    "feature_count": int(X_train_tfidf.shape[1]),
    "validation_accuracy": 0.943012,
    "validation_macro_f1": 0.943059,
}

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config,
        file,
        ensure_ascii=False,
        indent=2,
    )

print("配置文件已生成:", CONFIG_FILE)