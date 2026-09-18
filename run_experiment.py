"""Unified entry point for the final text-classification workflow."""

from pathlib import Path
import runpy
import time

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
SRC_DIR = ROOT / "src"

TRAIN_FILE = DATA_DIR / "train_data.csv"
TEST_FILE = DATA_DIR / "test_data_unlabeled.csv"

SEED = 42


def check_data():
    """Check that required input files exist and have valid columns."""
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    if not {"text", "target"}.issubset(train.columns):
        raise ValueError(
            "train_data.csv must contain text and target columns."
        )

    if "text" not in test.columns:
        raise ValueError(
            "test_data_unlabeled.csv must contain text column."
        )

    print("训练集样本数:", len(train))
    print("测试集样本数:", len(test))
    print("类别数量:", train["target"].nunique())

    return train, test


def run_script(filename):
    """Run a source script from the src directory."""
    script_path = SRC_DIR / filename

    if not script_path.exists():
        raise FileNotFoundError(script_path)

    print(f"\n运行 {filename} ...")
    runpy.run_path(
        str(script_path),
        run_name="__main__",
    )


def main():
    RESULT_DIR.mkdir(exist_ok=True)

    print("=" * 64)
    print("当代人工智能实验一：文本分类")
    print("=" * 64)

    print("\n检查数据...")
    check_data()

    # 统一运行器默认执行最终融合模型。
    # 该脚本会同时生成 predictions.csv 和 prediction.csv。
    start = time.perf_counter()

    run_script("final_fusion_predict.py")

    elapsed = time.perf_counter() - start

    print("\n最终模型配置：")
    print("word TF-IDF unigram")
    print("char_wb TF-IDF 3~5 gram")
    print("LinearSVC(C=1.0)")
    print("random_state=42")

    print(f"\n最终模型运行时间：{elapsed:.2f} 秒")
    print("已生成：predictions.csv")
    print("已生成：prediction.csv")
    print("=" * 64)


if __name__ == "__main__":
    main()