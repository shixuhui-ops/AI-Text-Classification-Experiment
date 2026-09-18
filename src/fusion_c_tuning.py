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
RESULT_FILE = RESULT_DIR / "fusion_c_tuning_results.csv"

SEED = 42
C_VALUES = [0.5, 1.0, 2.0]


data = pd.read_csv(TRAIN_FILE)
data["text"] = data["text"].fillna("").astype(str)

x_train, x_valid, y_train, y_valid = train_test_split(
    data["text"],
    data["target"],
    test_size=0.2,
    random_state=SEED,
    stratify=data["target"],
)

print("正在构建 word TF-IDF...")

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 1),
    min_df=2,
    sublinear_tf=True,
)

word_train = word_vectorizer.fit_transform(x_train)
word_valid = word_vectorizer.transform(x_valid)

print("word 特征数:", word_train.shape[1])
print("正在构建 char_wb TF-IDF...")

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=80000,
    sublinear_tf=True,
)

char_train = char_vectorizer.fit_transform(x_train)
char_valid = char_vectorizer.transform(x_valid)

print("char_wb 特征数:", char_train.shape[1])

x_train_fusion = hstack(
    [word_train, char_train],
    format="csr",
)

x_valid_fusion = hstack(
    [word_valid, char_valid],
    format="csr",
)

print("融合特征数:", x_train_fusion.shape[1])

results = []

for c_value in C_VALUES:
    print(f"\n正在训练 C={c_value}")

    start = time.perf_counter()

    model = LinearSVC(
        C=c_value,
        dual=True,
        max_iter=5000,
        random_state=SEED,
    )

    model.fit(x_train_fusion, y_train)

    train_prediction = model.predict(x_train_fusion)
    valid_prediction = model.predict(x_valid_fusion)

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

    train_accuracy = accuracy_score(
        y_train,
        train_prediction,
    )

    valid_accuracy = accuracy_score(
        y_valid,
        valid_prediction,
    )

    result = {
        "C": c_value,
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
        "train_macro_f1": train_f1,
        "valid_macro_f1": valid_f1,
        "generalization_gap": train_f1 - valid_f1,
        "time_seconds": time.perf_counter() - start,
    }

    results.append(result)

    print(f"训练 Accuracy: {train_accuracy:.4f}")
    print(f"验证 Accuracy: {valid_accuracy:.4f}")
    print(f"训练 Macro-F1: {train_f1:.4f}")
    print(f"验证 Macro-F1: {valid_f1:.4f}")
    print(f"泛化差距: {train_f1 - valid_f1:.4f}")

results_df = pd.DataFrame(results).sort_values(
    "valid_macro_f1",
    ascending=False,
)

RESULT_DIR.mkdir(exist_ok=True)

results_df.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("\n=== 融合模型 C 参数实验结果 ===")
print(results_df.to_string(index=False))
print(f"\n结果已保存：{RESULT_FILE}")