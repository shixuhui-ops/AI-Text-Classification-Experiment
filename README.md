# 当代人工智能实验一：文本分类

本项目使用 TF-IDF 和经典机器学习算法完成新闻文本十分类任务，包含模型对比、超参数实验、特征消融、训练过程可视化、错误分析以及测试集预测。

## 项目结构

```text
data-Project1/
├── data/
│   ├── train_data.csv
│   └── test_data_unlabeled.csv
├── src/
│   ├── baseline.py
│   ├── logistic_baseline.py
│   ├── svm_baseline.py
│   ├── svm_c_tuning.py
│   ├── svm_feature_compare.py
│   ├── svm_feature_count_tuning.py
│   ├── svm_header_ablation.py
│   ├── loss_curve_experiment.py
│   ├── svm_error_analysis.py
│   ├── inspect_errors.py
│   └── final_train_predict.py
├── results/
│   ├── run_results.csv
│   ├── final_model_config.json
│   ├── loss_constant_100_curve.png
│   ├── loss_constant_100_f1_curve.png
│   ├── svm_c_tuning_curve.png
│   ├── svm_feature_count_curve.png
│   ├── svm_feature_compare.png
│   ├── svm_header_ablation.png
│   ├── svm_confusion_matrix.png
│   └── 其他实验结果
├── run_experiment.py
├── predictions.csv
├── README.md
└── TUNING.md
```

## 数据集

有标签训练集包含 7368 条文本，无标签测试集包含 2457 条文本，任务共有 10 个类别。

验证阶段采用固定的分层划分：

- 训练子集：80%，5894 条
- 验证子集：20%，1474 条
- 随机种子：42

测试集不参与模型选择、超参数调整和验证评估。最终方案确定后，才使用全部有标签数据重新训练最终模型并预测测试集。

## 环境要求

建议使用 Python 3.10 或更高版本。

安装依赖：

```bash
pip install numpy pandas scikit-learn matplotlib
```

本实验实际运行环境：

```text
Python 3.13.12
pandas 3.0.3
scikit-learn 1.9.1
matplotlib 3.10.9
```

## 快速运行

在项目根目录执行：

```bash
python run_experiment.py
```

统一运行程序会自动完成：

1. 从 `data/` 读取训练集和测试集；
2. 对有标签数据进行固定的分层训练/验证划分；
3. 使用相同的 TF-IDF 特征比较朴素贝叶斯、逻辑回归和线性 SVM；
4. 保存验证结果到 `results/run_results.csv`；
5. 使用全部有标签数据重新训练最终模型；
6. 生成根目录下的 `predictions.csv`；
7. 保存最终模型配置到 `results/final_model_config.json`。

## 统一模型比较

三个模型使用相同的数据划分和 TF-IDF 特征设置，结果如下：

| 模型 | 验证 Accuracy | 验证 Macro-F1 |
|---|---:|---:|
| MultinomialNB | 0.9104 | 0.9109 |
| LogisticRegression | 0.9220 | 0.9224 |
| LinearSVC | **0.9430** | **0.9431** |

线性 SVM 在高维、稀疏的 TF-IDF 特征上取得了最佳验证性能。

## 最终模型

最终模型配置如下：

```text
文本处理：保留原始邮件文本
特征表示：TF-IDF unigram
min_df：2
sublinear_tf：True
分类器：LinearSVC
C：1.0
random_state：42
```

验证集结果：

```text
Accuracy：0.9430
Macro-F1：0.9431
```

最终使用全部 7368 条有标签文本重新拟合 TF-IDF 和 LinearSVC，并为 2457 条测试文本生成预测。

## 预测文件

最终预测文件为：

```text
predictions.csv
```

文件格式：

- 无表头
- 无索引
- 每行一个类别标签
- 共 2457 行
- 包含 10 个预测类别

## 实验设计

### SVM 正则化参数

保持数据划分和 TF-IDF 特征不变，只改变 `C`：

| C | 验证 Macro-F1 |
|---:|---:|
| 0.01 | 0.8569 |
| 0.1 | 0.9021 |
| 1 | **0.9176** |
| 10 | 0.9127 |
| 100 | 0.9102 |

较小的 `C` 导致欠拟合；当 `C` 大于 1 后，训练性能继续提升，但验证性能下降且泛化差距扩大。因此选择 `C=1.0`。

### TF-IDF 词表容量

保持 unigram 和 `C=1.0` 不变，只改变最大特征数：

| 实际特征数 | 验证 Macro-F1 |
|---:|---:|
| 5000 | 0.9207 |
| 10000 | 0.9311 |
| 20000 | 0.9417 |
| 38178 | **0.9431** |

扩大词表明显改善验证性能，但从 20000 增加到 38178 后边际收益已经较小。

### unigram 与 bigram

在相同的 20000 维特征预算下：

| 特征 | 验证 Macro-F1 |
|---|---:|
| unigram | **0.9417** |
| unigram + bigram | 0.9319 |

二元词组增加了稀疏性，并在固定词表容量下占用了部分有效单词特征空间，因此没有带来提升。

### 邮件头消融

| 文本输入 | 验证 Macro-F1 |
|---|---:|
| 原始邮件文本 | **0.9431** |
| 仅邮件正文 | 0.9248 |

删除邮件头使性能下降约 1.82 个百分点。实验说明 `Subject`、`Organization` 和发件人等邮件头信息包含有效的类别线索，因此最终方案保留原始文本。

## 损失曲线

最终模型 `LinearSVC` 不提供逐轮概率损失，因此训练过程可视化单独使用：

```text
SGDClassifier(loss="log_loss")
```

该实验仅用于观察 Log Loss 随训练轮次的变化，不将其表述为最终 SVM 的损失曲线。

在 100 轮训练中：

- 训练 Log Loss 从 2.2450 降至 0.6843；
- 验证 Log Loss 从 2.2488 降至 0.8129；
- 最高验证 Macro-F1 为 0.9184；
- 在 100 轮内未观察到验证损失反弹。

因此，在该学习率与正则化设置下模型仍处于缓慢收敛阶段，不能仅根据训练轮数断言出现过拟合。

## 错误分析

最佳 SVM 的错误主要集中在类别 0、1、2 和 7。

代表性主题词表明：

| 类别 | 主要主题 |
|---:|---|
| 0 | 计算机图形学 |
| 1 | Windows 软件 |
| 2 | 计算机硬件 |
| 7 | 电子、电路与音频 |

最常见混淆方向：

| 真实类别 | 预测类别 | 数量 |
|---:|---:|---:|
| 7 | 2 | 9 |
| 2 | 1 | 7 |
| 2 | 7 | 6 |

错误案例涉及主板、BIOS、硬盘接口、音频线和显示器信号等内容。这些文本同时包含计算机硬件和电子电路词汇，说明错误部分来自类别之间真实的语义重叠，而不只是随机误判。

## 详细实验脚本

`src/` 中保存了各阶段的独立实验脚本，用于展示完整探索过程：

| 脚本 | 功能 |
|---|---|
| `baseline.py` | 朴素贝叶斯基线 |
| `logistic_baseline.py` | 逻辑回归基线 |
| `svm_baseline.py` | 线性 SVM 基线 |
| `svm_c_tuning.py` | SVM 参数 C 对照实验 |
| `svm_feature_count_tuning.py` | TF-IDF 词表容量实验 |
| `svm_feature_compare.py` | unigram 与 bigram 对比 |
| `svm_header_ablation.py` | 邮件头消融实验 |
| `loss_curve_experiment.py` | Log Loss 和 Macro-F1 曲线 |
| `svm_error_analysis.py` | 分类报告与混淆矩阵 |
| `inspect_errors.py` | 代表词和典型错误案例 |
| `final_train_predict.py` | 最终训练与测试集预测 |

这些独立脚本记录实验过程。为了便于验收，推荐直接运行根目录中的 `run_experiment.py`。

## 结果文件

`results/` 中保存：

- 统一模型比较结果
- SVM 参数实验结果和曲线
- TF-IDF 词表容量实验结果和曲线
- unigram/bigram 对比结果
- 邮件头消融结果
- Log Loss 与 Macro-F1 曲线
- 混淆矩阵
- 类别代表词
- 典型错误案例
- 最终模型配置

## 可复现性

- 所有实验固定随机种子为 `42`；
- 验证集按标签分层划分；
- TF-IDF 只在训练子集上拟合；
- 验证集仅使用已拟合的向量器转换；
- 测试集不用于调参或模型选择；
- 最终配置确定后，才使用全部有标签数据重新训练；
- 模型参数、验证结果和预测文件均已保存。