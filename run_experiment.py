from pathlib import Path
import json
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"

TRAIN_FILE = DATA_DIR / "train_data.csv"
TEST_FILE = DATA_DIR / "test_data_unlabeled.csv"
PREDICTION_FILE = ROOT / "predictions.csv"
RESULT_FILE = RESULT_DIR / "run_results.csv"
CONFIG_FILE = RESULT_DIR / "final_model_config.json"

SEED = 42
VALID_SIZE = 0.2


def load_data():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    train["text"] = train["text"].fillna("").astype(str)
    test["text"] = test["text"].fillna("").astype(str)

    return train, test


def evaluate_models(train):
    x_train, x_valid, y_train, y_valid = train_test_split(
        train["text"],
        train["target"],
        test_size=VALID_SIZE,
        random_state=SEED,
        stratify=train["target"],
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2,
        sublinear_tf=True,
    )

    x_train_vec = vectorizer.fit_transform(x_train)
    x_valid_vec = vectorizer.transform(x_valid)

    models = {
        "MultinomialNB": MultinomialNB(alpha=1.0),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=SEED,
        ),
        "LinearSVC": LinearSVC(
            C=1.0,
            dual=True,
            max_iter=5000,
            random_state=SEED,
        ),
    }

    rows = []

    for name, model in models.items():
        start = time.perf_counter()

        model.fit(x_train_vec, y_train)

        train_prediction = model.predict(x_train_vec)
        valid_prediction = model.predict(x_valid_vec)

        elapsed = time.perf_counter() - start

        rows.append({
            "model": name,
            "train_accuracy": accuracy_score(
                y_train,
                train_prediction,
            ),
            "valid_accuracy": accuracy_score(
                y_valid,
                valid_prediction,
            ),
            "train_macro_f1": f1_score(
                y_train,
                train_prediction,
                average="macro",
            ),
            "valid_macro_f1": f1_score(
                y_valid,
                valid_prediction,
                average="macro",
            ),
            "train_seconds": elapsed,
        })

    results = pd.DataFrame(rows).sort_values(
        "valid_macro_f1",
        ascending=False,
    )

    results.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return results


def train_final_model(train, test):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2,
        sublinear_tf=True,
    )

    x_train_vec = vectorizer.fit_transform(
        train["text"]
    )
    x_test_vec = vectorizer.transform(
        test["text"]
    )

    model = LinearSVC(
        C=1.0,
        dual=True,
        max_iter=5000,
        random_state=SEED,
    )

    model.fit(
        x_train_vec,
        train["target"],
    )

    prediction = model.predict(x_test_vec)

    pd.DataFrame(prediction).to_csv(
        PREDICTION_FILE,
        index=False,
        header=False,
    )

    config = {
        "seed": SEED,
        "validation_ratio": VALID_SIZE,
        "model": "LinearSVC",
        "C": 1.0,
        "feature": "TF-IDF unigram",
        "min_df": 2,
        "sublinear_tf": True,
        "text_processing": "raw_text",
        "train_samples": len(train),
        "test_samples": len(test),
        "feature_count": x_train_vec.shape[1],
        "prediction_file": str(
            PREDICTION_FILE.relative_to(ROOT)
        ),
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


def main():
    RESULT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("实验一：文本分类统一运行程序")
    print("=" * 60)

    print("\n读取数据……")
    train, test = load_data()

    print(f"训练集样本数：{len(train)}")
    print(f"测试集样本数：{len(test)}")
    print(f"类别数量：{train['target'].nunique()}")

    print("\n运行模型对比……")
    results = evaluate_models(train)

    print("\n验证集结果：")
    print(
        results[
            [
                "model",
                "valid_accuracy",
                "valid_macro_f1",
            ]
        ].to_string(index=False)
    )

    print("\n使用全部训练数据训练最终模型……")
    train_final_model(train, test)

    print("\n实验完成。")
    print(f"验证结果：{RESULT_FILE.relative_to(ROOT)}")
    print(f"最终预测：{PREDICTION_FILE.relative_to(ROOT)}")
    print(f"模型配置：{CONFIG_FILE.relative_to(ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()