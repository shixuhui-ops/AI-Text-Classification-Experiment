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
C_VALUE = 0.5


def check_data():
    """Check required files and columns."""
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"找不到训练集：{TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"找不到测试集：{TEST_FILE}"
        )

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    if not {"text", "target"}.issubset(train.columns):
        raise ValueError(
            "train_data.csv 必须包含 text 和 target 列"
        )

    if "text" not in test.columns:
        raise ValueError(
            "test_data_unlabeled.csv 必须包含 text 列"
        )

    print("训练集样本数:", len(train))
    print("测试集样本数:", len(test))
    print("类别数量:", train["target"].nunique())

    return train, test


def run_final_fusion_prediction():
    """Run the final word + char_wb fusion prediction script."""
    script_path = SRC_DIR / "final_fusion_predict.py"

    if not script_path.exists():
        raise FileNotFoundError(
            f"找不到最终预测脚本：{script_path}"
        )

    # final_fusion_predict.py 中固定使用 C=0.5。
    runpy.run_path(
        str(script_path),
        run_name="__main__",
    )


def verify_prediction_files():
    """Verify both prediction files have the required format."""
    predictions_path = ROOT / "predictions.csv"
    prediction_path = ROOT / "prediction.csv"

    predictions = pd.read_csv(
        predictions_path,
        header=None,
    )
    prediction = pd.read_csv(
        prediction_path,
        header=None,
    )

    expected_shape = (2457, 1)

    if predictions.shape != expected_shape:
        raise RuntimeError(
            f"predictions.csv 形状错误：{predictions.shape}"
        )

    if prediction.shape != expected_shape:
        raise RuntimeError(
            f"prediction.csv 形状错误：{prediction.shape}"
        )

    if not predictions.equals(prediction):
        raise RuntimeError(
            "predictions.csv 和 prediction.csv 内容不一致"
        )

    if predictions[0].nunique() != 10:
        raise RuntimeError(
            "预测结果没有覆盖 10 个类别"
        )

    print("预测文件检查通过：")
    print("predictions.csv:", predictions.shape)
    print("prediction.csv:", prediction.shape)
    print("两个文件内容一致：True")


def main():
    RESULT_DIR.mkdir(exist_ok=True)

    print("=" * 64)
    print("当代人工智能实验一：文本分类")
    print("=" * 64)

    print("\n检查数据...")
    check_data()

    print("\n运行最终融合模型...")
    start = time.perf_counter()
    run_final_fusion_prediction()
    elapsed = time.perf_counter() - start

    verify_prediction_files()

    print("\n最终模型配置：")
    print("word TF-IDF unigram")
    print("char_wb TF-IDF 3~5 gram")
    print(f"LinearSVC(C={C_VALUE})")
    print(f"random_state={SEED}")

    print(f"\n运行时间：{elapsed:.2f} 秒")
    print("实验完成。")
    print("=" * 64)


if __name__ == "__main__":
    main()