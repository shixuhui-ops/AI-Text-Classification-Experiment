# 当代人工智能实验一：文本分类

## 实验简介

本实验使用新闻文本十分类数据集，比较朴素贝叶斯、逻辑回归和线性支持向量机等经典机器学习方法。

实验内容包括：

- TF-IDF 文本特征提取
- 训练集与验证集分层划分
- 不同分类模型对比
- SVM 正则化参数调节
- TF-IDF 特征数量比较
- unigram 与 bigram 特征比较
- 邮件头信息消融实验
- 训练损失曲线
- 混淆矩阵与错误分析
- 测试集预测

## 数据集与划分

训练集为 `train_data.csv`，包含 7368 条带标签文本；测试集为 `test_data_unlabeled.csv`，包含 2457 条无标签文本。任务共有 10 个类别。

实验固定随机种子为 `42`，将有标签数据按类别分层划分为：

- 训练子集：80%，5894 条
- 验证子集：20%，1474 条

测试集不参与模型选择、超参数调整或验证评估。最终方案确定后，才使用全部有标签数据重新训练模型并生成测试集预测。

## 环境安装

建议使用 Python 3.10 或更高版本。

```bash
pip install numpy pandas scikit-learn matplotlib
```

本次实验实际运行环境：

- Python 3.13.12
- pandas 3.0.3
- scikit-learn 1.9.1
- matplotlib 3.10.9

## 最终模型

最终模型配置：

```text
文本处理：保留原始邮件文本
特征表示：TF-IDF unigram
min_df：2
sublinear_tf：True
分类器：LinearSVC
C：1.0
random_state：42
```

最佳验证集结果：

```text
Accuracy：0.9430
Macro-F1：0.9431
```

## 主要实验结果

| 实验设置 | 验证 Macro-F1 |
|---|---:|
| MultinomialNB，alpha=1.0 | 0.8867 |
| LogisticRegression，C=1.0 | 0.8985 |
| LinearSVC，5000 个 unigram 特征 | 0.9176 |
| LinearSVC，38178 个 unigram 特征 | 0.9431 |
| LinearSVC，unigram + bigram | 0.9319 |
| LinearSVC，仅保留正文 | 0.9248 |

线性 SVM 在当前高维稀疏 TF-IDF 特征上表现最好。增加 unigram 词表规模能够提升验证性能，但加入 bigram 后性能下降。删除邮件头后性能也下降，说明主题、组织机构和发件人等信息包含有效的类别线索。

## 运行最终预测

```bash
python final_train_predict.py
```

程序将使用全部有标签数据训练最终模型，并生成：

- `predictions.csv`：测试集预测结果
- `final_model_config.json`：最终模型配置

`predictions.csv` 无表头、无索引，每行一个预测标签，共 2457 行。

## 运行各项实验

运行三个基线模型：

```bash
python baseline.py
python logistic_baseline.py
python svm_baseline.py
```

运行 SVM 参数实验：

```bash
python svm_c_tuning.py
```

运行 TF-IDF 词表容量实验：

```bash
python svm_feature_count_tuning.py
```

运行 unigram 与 bigram 对比：

```bash
python svm_feature_compare.py
```

运行邮件头消融实验：

```bash
python svm_header_ablation.py
```

运行损失曲线实验：

```bash
python loss_curve_experiment.py
```

运行错误分析：

```bash
python svm_error_analysis.py
python inspect_errors.py
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `final_train_predict.py` | 最终模型训练和测试集预测 |
| `predictions.csv` | 最终测试集预测结果 |
| `final_model_config.json` | 最终模型配置 |
| `loss_constant_100_curve.png` | 训练集和验证集 Log Loss 曲线 |
| `loss_constant_100_f1_curve.png` | 训练集和验证集 Macro-F1 曲线 |
| `svm_c_tuning_curve.png` | SVM 参数 C 的对比图 |
| `svm_feature_count_curve.png` | TF-IDF 词表容量对比图 |
| `svm_feature_compare.png` | unigram 与 bigram 对比图 |
| `svm_header_ablation.png` | 邮件头消融实验图 |
| `svm_confusion_matrix.png` | 最佳模型验证集混淆矩阵 |
| `svm_error_examples.txt` | 典型错误案例 |
| `svm_top_terms.csv` | 各类别代表性词 |

## 损失曲线说明

最终分类性能最好的模型是 `LinearSVC`，但该模型不提供逐轮概率损失。

因此，损失曲线实验单独使用：

```text
SGDClassifier(loss="log_loss")
```

该实验用于观察迭代训练过程，不用于替代最终 SVM，也不将其损失曲线表述为 SVM 的训练损失。

在 100 轮训练中，训练损失从 `2.2450` 降至 `0.6843`，验证损失从 `2.2488` 降至 `0.8129`。验证损失仍保持下降，未观察到明显的后期反弹，因此不能断言该设置已经出现过拟合。

## 错误分析

最佳模型的主要混淆集中在类别 `0、1、2、7`。

其中：

- 类别 2 主要与计算机硬件有关
- 类别 7 主要与电子、电路和音频有关
- 最常见错误为类别 7 被预测为类别 2，共 9 条
- 类别 2 被预测为类别 1，共 7 条
- 类别 2 被预测为类别 7，共 6 条

错误案例表明，显示器、主板、硬盘接口和音频线等文本同时包含计算机硬件与电子电路词汇，类别之间存在真实的语义重叠。

## 可复现性说明

实验固定随机种子为 `42`，并记录了数据划分、TF-IDF 参数、模型参数和验证结果。

所有涉及词表及统计量的 TF-IDF 操作都只在训练子集上拟合。验证集和测试集只使用已经拟合的向量器进行转换，以避免数据泄露。