from pathlib import Path
import json

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"

TRAIN_FILE = DATA_DIR / "train_data.csv"
TEST_FILE = DATA_DIR / "test_data_unlabeled.csv"

PREDICTIONS_FILE = ROOT / "predictions.csv"
PREDICTION_FILE = ROOT / "prediction.csv"
CONFIG_FILE = RESULT_DIR / "final_fusion_model_config.json"


SEED = 42


# =========================
# 1. 读取数据
# =========================

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

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
# 2. 构造两种 TF-IDF 特征
# =========================

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

print("正在拟合 word TF-IDF……")

word_train = word_vectorizer.fit_transform(
    train_data["text"]
)
word_test = word_vectorizer.transform(
    test_data["text"]
)

print("word 特征数:", word_train.shape[1])

print("正在拟合 char_wb TF-IDF……")

char_train = char_vectorizer.fit_transform(
    train_data["text"]
)
char_test = char_vectorizer.transform(
    test_data["text"]
)

print("char_wb 特征数:", char_train.shape[1])


# =========================
# 3. 拼接 word 和 char 特征
# =========================

x_train = hstack(
    [word_train, char_train],
    format="csr",
)

x_test = hstack(
    [word_test, char_test],
    format="csr",
)

print("融合特征数:", x_train.shape[1])


# =========================
# 4. 训练最终 LinearSVC
# =========================

model = LinearSVC(
    C=1.0,
    dual=True,
    max_iter=5000,
    random_state=SEED,
)

print("正在训练最终融合模型……")

model.fit(
    x_train,
    train_data["target"],
)


# =========================
# 5. 生成预测
# =========================

predictions = model.predict(x_test)
prediction_frame = pd.DataFrame(predictions)

# 仓库中的预测文件
prediction_frame.to_csv(
    PREDICTIONS_FILE,
    index=False,
    header=False,
)

# 课程提交文件
prediction_frame.to_csv(
    PREDICTION_FILE,
    index=False,
    header=False,
)


# =========================
# 6. 保存配置
# =========================

config = {
    "seed": SEED,
    "model": "LinearSVC",
    "C": 1.0,
    "text_processing": "raw_text",
    "feature_combination": (
        "word_tfidf_unigram + "
        "char_wb_tfidf_3_5gram"
    ),
    "word_min_df": 2,
    "word_sublinear_tf": True,
    "char_ngram_range": [3, 5],
    "char_min_df": 2,
    "char_max_features": 80000,
    "word_features": int(word_train.shape[1]),
    "char_features": int(char_train.shape[1]),
    "total_features": int(x_train.shape[1]),
    "train_samples": int(len(train_data)),
    "test_samples": int(len(test_data)),
    "validation_accuracy_mean": 0.945862,
    "validation_accuracy_std": 0.003820,
    "validation_macro_f1_mean": 0.946090,
    "validation_macro_f1_std": 0.003868,
    "prediction_files": [
        "predictions.csv",
        "prediction.csv",
    ],
}

CONFIG_FILE.write_text(
    json.dumps(
        config,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# =========================
# 7. 输出检查
# =========================

print("\n最终融合模型训练完成")
print("预测样本数:", len(predictions))
print("预测类别数:", prediction_frame[0].nunique())
print("predictions.csv:", PREDICTIONS_FILE)
print("prediction.csv:", PREDICTION_FILE)
print("配置文件:", CONFIG_FILE)

assert len(predictions) == len(test_data)
assert prediction_frame[0].nunique() == 10