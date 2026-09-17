import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


SEED = 42
EPOCHS = 100

# 使用新文件名前缀，避免覆盖之前的结果
OUTPUT_PREFIX = "loss_constant_100"


# =========================
# 1. 读取训练数据
# =========================

data = pd.read_csv("train_data.csv")

if "text" not in data.columns or "target" not in data.columns:
    raise ValueError("train_data.csv 必须包含 text 和 target 两列")

data["text"] = data["text"].fillna("").astype(str)

# 注意：这里只使用 train_data.csv
# test_data_unlabeled.csv 没有参与训练、验证或损失计算。


# =========================
# 2. 分层划分训练集和验证集
# =========================

X_train, X_valid, y_train, y_valid = train_test_split(
    data["text"],
    data["target"],
    test_size=0.2,
    random_state=SEED,
    stratify=data["target"],
)

print("训练集大小:", len(X_train))
print("验证集大小:", len(X_valid))


# =========================
# 3. TF-IDF 特征提取
# =========================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 1),
    min_df=2,
    sublinear_tf=True,
)

# 只在训练集上 fit
X_train_tfidf = vectorizer.fit_transform(X_train)

# 验证集只进行 transform
X_valid_tfidf = vectorizer.transform(X_valid)

print("TF-IDF 特征数:", X_train_tfidf.shape[1])


# =========================
# 4. 构造可逐轮训练的逻辑损失模型
# =========================

classes = np.sort(y_train.unique())
y_train_array = np.asarray(y_train)

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
# 5. 定义评估函数
# =========================

def evaluate_current_model(epoch):
    """
    计算当前轮次的训练/验证 Log Loss 和 Macro-F1。

    Log Loss 使用模型输出的类别概率计算。
    测试集不参与任何计算。
    """

    train_probability = model.predict_proba(
        X_train_tfidf
    )

    valid_probability = model.predict_proba(
        X_valid_tfidf
    )

    train_loss = log_loss(
        y_train,
        train_probability,
        labels=classes,
    )

    valid_loss = log_loss(
        y_valid,
        valid_probability,
        labels=classes,
    )

    train_prediction = model.predict(
        X_train_tfidf
    )

    valid_prediction = model.predict(
        X_valid_tfidf
    )

    train_macro_f1 = f1_score(
        y_train,
        train_prediction,
        average="macro",
    )

    valid_macro_f1 = f1_score(
        y_valid,
        valid_prediction,
        average="macro",
    )

    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_macro_f1": train_macro_f1,
        "valid_macro_f1": valid_macro_f1,
    }


# =========================
# 6. 逐轮训练并记录结果
# =========================

history = []

for epoch in range(1, EPOCHS + 1):
    # 每轮固定规则打乱训练样本
    shuffled_x, shuffled_y = shuffle(
        X_train_tfidf,
        y_train_array,
        random_state=SEED + epoch,
    )

    if epoch == 1:
        model.partial_fit(
            shuffled_x,
            shuffled_y,
            classes=classes,
        )
    else:
        model.partial_fit(
            shuffled_x,
            shuffled_y,
        )

    result = evaluate_current_model(epoch)
    history.append(result)

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss={result['train_loss']:.4f} | "
        f"Valid Loss={result['valid_loss']:.4f} | "
        f"Train Macro-F1={result['train_macro_f1']:.4f} | "
        f"Valid Macro-F1={result['valid_macro_f1']:.4f}"
    )


history_df = pd.DataFrame(history)

history_path = f"{OUTPUT_PREFIX}_history.csv"

history_df.to_csv(
    history_path,
    index=False,
    encoding="utf-8-sig",
)


# =========================
# 7. 查找最佳轮次
# =========================

best_loss_index = history_df["valid_loss"].idxmin()
best_loss_epoch = int(
    history_df.loc[best_loss_index, "epoch"]
)

best_f1_index = history_df["valid_macro_f1"].idxmax()
best_f1_epoch = int(
    history_df.loc[best_f1_index, "epoch"]
)


# =========================
# 8. 绘制 Log Loss 曲线
# =========================

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    history_df["epoch"],
    history_df["train_loss"],
    linewidth=2,
    label="Train Log Loss",
)

ax.plot(
    history_df["epoch"],
    history_df["valid_loss"],
    linewidth=2,
    label="Validation Log Loss",
)

ax.axvline(
    best_loss_epoch,
    linestyle="--",
    color="gray",
    alpha=0.8,
    label=(
        f"Minimum validation loss: "
        f"epoch {best_loss_epoch}"
    ),
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Log Loss")
ax.set_title("Training and Validation Log Loss")
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

loss_plot_path = f"{OUTPUT_PREFIX}_curve.png"

fig.savefig(
    loss_plot_path,
    dpi=220,
    bbox_inches="tight",
)

plt.show()


# =========================
# 9. 绘制 Macro-F1 曲线
# =========================

fig_f1, ax_f1 = plt.subplots(figsize=(8, 5))

ax_f1.plot(
    history_df["epoch"],
    history_df["train_macro_f1"],
    linewidth=2,
    label="Train Macro-F1",
)

ax_f1.plot(
    history_df["epoch"],
    history_df["valid_macro_f1"],
    linewidth=2,
    label="Validation Macro-F1",
)

ax_f1.axvline(
    best_f1_epoch,
    linestyle="--",
    color="gray",
    alpha=0.8,
    label=(
        f"Maximum validation Macro-F1: "
        f"epoch {best_f1_epoch}"
    ),
)

ax_f1.set_xlabel("Epoch")
ax_f1.set_ylabel("Macro-F1")
ax_f1.set_ylim(0, 1)
ax_f1.set_title("Training and Validation Macro-F1")
ax_f1.grid(alpha=0.25)
ax_f1.legend()

fig_f1.tight_layout()

f1_plot_path = f"{OUTPUT_PREFIX}_f1_curve.png"

fig_f1.savefig(
    f1_plot_path,
    dpi=220,
    bbox_inches="tight",
)

plt.show()


# =========================
# 10. 输出实验摘要
# =========================

print("\n=== 实验配置 ===")
print("训练数据文件: train_data.csv")
print("测试集是否参与训练或验证: 否")
print("随机种子:", SEED)
print("训练轮数:", EPOCHS)
print("learning_rate: constant")
print("eta0: 0.01")
print("alpha: 0.0001")
print("average: False")

print("\n=== 最佳结果 ===")
print("验证损失最低轮次:", best_loss_epoch)
print(
    "最低验证损失:",
    f"{history_df.loc[best_loss_index, 'valid_loss']:.6f}",
)
print(
    "该轮验证 Macro-F1:",
    f"{history_df.loc[best_loss_index, 'valid_macro_f1']:.6f}",
)

print("\n验证 Macro-F1 最高轮次:", best_f1_epoch)
print(
    "最高验证 Macro-F1:",
    f"{history_df.loc[best_f1_index, 'valid_macro_f1']:.6f}",
)

print("\n结果文件:")
print(history_path)
print(loss_plot_path)
print(f1_plot_path)