from pathlib import Path
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"

TRAIN_FILE = DATA_DIR / "train_data.csv"
RESULT_FILE = RESULT_DIR / "char_tfidf_results.csv"

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

settings = {
    "word_unigram": {
        "analyzer": "word",
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_features": None,
    },
    "char": {
        "analyzer": "char",
        "ngram_range": (3, 5),
        "min_df": 2,
        "max_features": 80000,
    },
    "char_wb": {
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "min_df": 2,
        "max_features": 80000,
    },
}

results = []

for name, config in settings.items():
    print("=" * 60)
    print(f"正在运行：{name}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer=config["analyzer"],
        ngram_range=config["ngram_range"],
        min_df=config["min_df"],
        max_features=config["max_features"],
        sublinear_tf=True,
    )

    start = time.perf_counter()

    x_train_vec = vectorizer.fit_transform(x_train)
    x_valid_vec = vectorizer.transform(x_valid)

    model = LinearSVC(
        C=1.0,
        dual=True,
        max_iter=5000,
        random_state=SEED,
    )

    model.fit(x_train_vec, y_train)

    train_prediction = model.predict(x_train_vec)
    valid_prediction = model.predict(x_valid_vec)

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

    elapsed = time.perf_counter() - start

    result = {
        "feature_type": name,
        "feature_count": x_train_vec.shape[1],
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
        "train_macro_f1": train_f1,
        "valid_macro_f1": valid_f1,
        "generalization_gap": train_f1 - valid_f1,
        "time_seconds": elapsed,
    }

    results.append(result)

    print(f"特征数: {x_train_vec.shape[1]}")
    print(f"训练 Accuracy: {train_accuracy:.4f}")
    print(f"验证 Accuracy: {valid_accuracy:.4f}")
    print(f"训练 Macro-F1: {train_f1:.4f}")
    print(f"验证 Macro-F1: {valid_f1:.4f}")
    print(f"泛化差距: {train_f1 - valid_f1:.4f}")
    print(f"运行时间: {elapsed:.2f} 秒")


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

print("\n=== 字符与单词 TF-IDF 对比 ===")
print(results_df.to_string(index=False))
print(f"\n结果已保存：{RESULT_FILE}")