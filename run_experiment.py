"""Unified entry point for Experiment 1."""

import argparse
import json
import runpy
import time
from pathlib import Path

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
SRC_DIR = ROOT / "src"

TRAIN_FILE = DATA_DIR / "train_data.csv"
TEST_FILE = DATA_DIR / "test_data_unlabeled.csv"

PREDICTIONS_FILE = ROOT / "predictions.csv"
SUBMISSION_PREDICTION_FILE = ROOT / "prediction.csv"

RESULT_FILE = RESULT_DIR / "run_results.csv"
CONFIG_FILE = RESULT_DIR / "final_model_config.json"

SEED = 42
VALID_SIZE = 0.2


def load_data():
    """Load and validate the labeled and unlabeled data."""
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    if not {"text", "target"}.issubset(train.columns):
        raise ValueError(
            "data/train_data.csv must contain text and target columns."
        )

    if "text" not in test.columns:
        raise ValueError(
            "data/test_data_unlabeled.csv must contain a text column."
        )

    train["text"] = train["text"].fillna("").astype(str)
    test["text"] = test["text"].fillna("").astype(str)

    return train, test


def build_vectorizer():
    """Create the TF-IDF configuration selected on validation data."""
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2,
        sublinear_tf=True,
    )


def evaluate_models(train):
    """Compare three classifiers under identical features and split."""
    x_train, x_valid, y_train, y_valid = train_test_split(
        train["text"],
        train["target"],
        test_size=VALID_SIZE,
        random_state=SEED,
        stratify=train["target"],
    )

    vectorizer = build_vectorizer()

    # Fit TF-IDF only on the training subset.
    x_train_vec = vectorizer.fit_transform(x_train)
    x_valid_vec = vectorizer.transform(x_valid)

    models = {
        "MultinomialNB": MultinomialNB(
            alpha=1.0,
        ),
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

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        start = time.perf_counter()
        model.fit(x_train_vec, y_train)

        train_prediction = model.predict(x_train_vec)
        valid_prediction = model.predict(x_valid_vec)

        rows.append({
            "model": model_name,
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
            "train_seconds": (
                time.perf_counter() - start
            ),
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
    """
    Retrain the selected model on all labeled data.

    The unlabeled test data is used only at this final stage.
    """
    vectorizer = build_vectorizer()

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

    predictions = model.predict(x_test_vec)
    prediction_frame = pd.DataFrame(predictions)

    # Repository output.
    prediction_frame.to_csv(
        PREDICTIONS_FILE,
        index=False,
        header=False,
    )

    # File name required in the submission archive.
    prediction_frame.to_csv(
        SUBMISSION_PREDICTION_FILE,
        index=False,
        header=False,
    )

    config = {
        "seed": SEED,
        "validation_ratio": VALID_SIZE,
        "model": "LinearSVC",
        "C": 1.0,
        "feature_method": "TF-IDF",
        "ngram_range": [1, 1],
        "min_df": 2,
        "sublinear_tf": True,
        "text_processing": "raw_text",
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "feature_count": int(x_train_vec.shape[1]),
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


def run_minibatch_loss_experiment():
    """Run the separate mini-batch Log Loss analysis."""
    script = SRC_DIR / "loss_minibatch_experiment.py"

    if not script.exists():
        raise FileNotFoundError(
            f"Missing loss experiment script: {script}"
        )

    runpy.run_path(
        str(script),
        run_name="__main__",
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the main text-classification workflow."
        )
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Also regenerate the mini-batch loss curve. "
            "Without this option, run only model comparison "
            "and final prediction."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    RESULT_DIR.mkdir(exist_ok=True)

    print("=" * 62)
    print("Experiment 1: Text Classification")
    print("=" * 62)

    print("\nLoading data...")
    train, test = load_data()

    print(f"Labeled samples: {len(train)}")
    print(f"Unlabeled test samples: {len(test)}")
    print(f"Number of classes: {train['target'].nunique()}")

    print("\nRunning fair model comparison...")
    results = evaluate_models(train)

    print("\nValidation results:")
    print(
        results[
            [
                "model",
                "valid_accuracy",
                "valid_macro_f1",
            ]
        ].to_string(index=False)
    )

    print("\nTraining the final model on all labeled data...")
    train_final_model(train, test)

    print("\nGenerated:")
    print(f"- {RESULT_FILE.relative_to(ROOT)}")
    print(f"- {CONFIG_FILE.relative_to(ROOT)}")
    print(f"- {PREDICTIONS_FILE.relative_to(ROOT)}")
    print(
        f"- {SUBMISSION_PREDICTION_FILE.relative_to(ROOT)}"
    )

    if args.full:
        print("\nRunning mini-batch loss analysis...")
        run_minibatch_loss_experiment()

    print("\nFinished successfully.")
    print("=" * 62)


if __name__ == "__main__":
    main()