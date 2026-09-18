from pathlib import Path
import time

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
RESULT_FILE = RESULT_DIR / "word_char_fusion_results.csv"

SEED = 42

data = pd.read_csv(TRAIN_FILE)
data["text"] = data["text"].fillna("").astype(str)

x_train, x_valid, y_train, y_valid = train_test_split(
    data["text"],
    data["target"],
    test_size=0.2,
    random_state=SEED,
    stratify=data["target"],
)

start = time.perf_counter()

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

print("word 特征数:", word_train.shape[1])
print("char_wb 特征数:", char_train.shape[1])

# 拼接两种稀疏特征
x_train_fusion = hstack(
    [word_train, char_train],
    format="csr",
)

x_valid_fusion = hstack(
    [word_valid, char_valid],
    format="csr",
)

print("融合后特征数:", x_train_fusion.shape[1])

model = LinearSVC(
    C=1.0,
    dual=True,
    max_iter=5000,
    random_state=SEED,
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

train_f1 = f1_score(
    y_train,
    train_prediction,
    average="macro",
)
valid_f1 = f1_score(
    y_valid,
    valid_prediction,
    average="macro",
)

elapsed = time.perf_counter() - start

result = {
    "feature_type": "word_unigram + char_wb",
    "word_features": word_train.shape[1],
    "char_features": char_train.shape[1],
    "total_features": x_train_fusion.shape[1],
    "train_accuracy": train_accuracy,
    "valid_accuracy": valid_accuracy,
    "train_macro_f1": train_f1,
    "valid_macro_f1": valid_f1,
    "generalization_gap": train_f1 - valid_f1,
    "time_seconds": elapsed,
}

result_df = pd.DataFrame([result])

RESULT_DIR.mkdir(exist_ok=True)

result_df.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("\n=== Word + Character 特征融合结果 ===")
print(result_df.to_string(index=False))
print(f"\n结果已保存：{RESULT_FILE}")