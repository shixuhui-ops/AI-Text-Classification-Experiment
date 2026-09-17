from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# =========================
# 1. 路径和实验配置
# =========================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"

TRAIN_FILE = DATA_DIR / "train_data.csv"

SEED = 42
EPOCHS = 20
BATCH_SIZE = 256

RESULT_DIR.mkdir(exist_ok=True)


# =========================
# 2. 读取并划分数据
# =========================

data = pd.read_csv(TRAIN_FILE)

data["text"] = (
    data["text"]
    .fillna("")
    .astype(str)
)

X_train_text, X_valid_text, y_train, y_valid = (
    train_test_split(
        data["text"],
        data["target"],
        test_size=0.2,
        random_state=SEED,
        stratify=data["target"],
    )
)


# =========================
# 3. TF-IDF 特征
# =========================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    min_df=2,
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(X_train_text)
X_valid = vectorizer.transform(X_valid_text)

y_train = np.asarray(y_train)
y_valid = np.asarray(y_valid)

classes = np.sort(np.unique(y_train))

print("训练集大小:", X_train.shape[0])
print("验证集大小:", X_valid.shape[0])
print("TF-IDF 特征数:", X_train.shape[1])
print("Batch size:", BATCH_SIZE)


# =========================
# 4. Mini-batch SGD 模型
# =========================

model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-4,
    learning_rate="constant",
    eta0=0.01,
    average=False,
    random_state=SEED,
)


# =========================
# 5. Mini-batch 训练
# =========================

history = []
update_step = 0
first_update = True

for epoch in range(1, EPOCHS + 1):
    X_epoch, y_epoch = shuffle(
        X_train,
        y_train,
        random_state=SEED + epoch,
    )

    for batch_start in range(
        0,
        X_epoch.shape[0],
        BATCH_SIZE,
    ):
        batch_end = min(
            batch_start + BATCH_SIZE,
            X_epoch.shape[0],
        )

        X_batch = X_epoch[batch_start:batch_end]
        y_batch = y_epoch[batch_start:batch_end]

        if first_update:
            model.partial_fit(
                X_batch,
                y_batch,
                classes=classes,
            )
            first_update = False
        else:
            model.partial_fit(
                X_batch,
                y_batch,
            )

        update_step += 1

        # 当前 mini-batch 的即时损失
        batch_probability = model.predict_proba(
            X_batch
        )

        batch_loss = log_loss(
            y_batch,
            batch_probability,
            labels=classes,
        )

        # 每次更新后在固定验证集上评估
        valid_probability = model.predict_proba(
            X_valid
        )

        valid_loss = log_loss(
            y_valid,
            valid_probability,
            labels=classes,
        )

        history.append({
            "update_step": update_step,
            "epoch": epoch,
            "batch_start": batch_start,
            "batch_size": len(y_batch),
            "batch_train_loss": batch_loss,
            "valid_loss": valid_loss,
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Step {update_step:03d} | "
            f"Batch Loss={batch_loss:.4f} | "
            f"Valid Loss={valid_loss:.4f}"
        )


# =========================
# 6. 保存训练记录
# =========================

history_df = pd.DataFrame(history)

history_path = (
    RESULT_DIR / "loss_minibatch_history.csv"
)

history_df.to_csv(
    history_path,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 7. 添加移动平均
# =========================

# 只用于帮助观察整体趋势，不替换原始损失。
window_size = 10

history_df["batch_loss_moving_average"] = (
    history_df["batch_train_loss"]
    .rolling(
        window=window_size,
        min_periods=1,
    )
    .mean()
)

history_df["valid_loss_moving_average"] = (
    history_df["valid_loss"]
    .rolling(
        window=window_size,
        min_periods=1,
    )
    .mean()
)


# =========================
# 8. 绘制曲线
# =========================

fig, ax = plt.subplots(figsize=(9, 5))

# 原始 mini-batch 训练损失：
# 使用浅色显示真实波动
ax.plot(
    history_df["update_step"],
    history_df["batch_train_loss"],
    color="#2563EB",
    alpha=0.28,
    linewidth=1,
    label="Mini-batch Train Loss",
)

# 训练损失移动平均：
# 用于展示整体趋势
ax.plot(
    history_df["update_step"],
    history_df["batch_loss_moving_average"],
    color="#1D4ED8",
    linewidth=2,
    label=(
        f"Train Loss Moving Average "
        f"(window={window_size})"
    ),
)

# 验证集损失
ax.plot(
    history_df["update_step"],
    history_df["valid_loss"],
    color="#F97316",
    alpha=0.45,
    linewidth=1,
    label="Validation Loss",
)

# 验证损失移动平均
ax.plot(
    history_df["update_step"],
    history_df["valid_loss_moving_average"],
    color="#C2410C",
    linewidth=2,
    label=(
        f"Validation Loss Moving Average "
        f"(window={window_size})"
    ),
)

ax.set_xlabel("Mini-batch Update Step")
ax.set_ylabel("Log Loss")
ax.set_title(
    "Mini-batch Training and Validation Log Loss"
)
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

plot_path = (
    RESULT_DIR / "loss_minibatch_curve.png"
)

fig.savefig(
    plot_path,
    dpi=220,
    bbox_inches="tight",
)

plt.show()


# =========================
# 9. 输出摘要
# =========================

best_index = history_df["valid_loss"].idxmin()

print("\n=== 实验摘要 ===")
print("总 Epoch:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("总更新次数:", update_step)
print(
    "最低验证损失:",
    f"{history_df.loc[best_index, 'valid_loss']:.6f}",
)
print(
    "最低验证损失所在更新次数:",
    int(history_df.loc[best_index, "update_step"]),
)
print(
    "对应 Epoch:",
    int(history_df.loc[best_index, "epoch"]),
)
print("\n结果文件:")
print(history_path)
print(plot_path)