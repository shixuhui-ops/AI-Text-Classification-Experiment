# 当代人工智能实验一：文本分类

本项目使用 TF-IDF 和经典机器学习算法完成新闻文本十分类任务，包含模型比较、超参数实验、特征消融、字符特征融合、损失曲线、错误分析和测试集预测。

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
│   ├── fusion_c_tuning.py
│   ├── fusion_c05_multi_seed.py
│   ├── svm_feature_compare.py
│   ├── svm_feature_count_tuning.py
│   ├── svm_header_ablation.py
│   ├── char_tfidf_experiment.py
│   ├── word_char_fusion_experiment.py
│   ├── multi_seed_experiment.py
│   ├── fusion_multi_seed_experiment.py
│   ├── loss_curve_experiment.py
│   ├── loss_minibatch_experiment.py
│   ├── svm_error_analysis.py
│   ├── inspect_errors.py
│   ├── final_train_predict.py
│   └── final_fusion_predict.py
├── results/
│   ├── final_fusion_model_config.json
│   ├── char_tfidf_results.csv
│   ├── word_char_fusion_results.csv
│   ├── fusion_c_tuning_results.csv
│   ├── fusion_c05_multi_seed_results.csv
│   ├── fusion_c05_multi_seed_summary.csv
│   ├── fusion_multi_seed_results.csv
│   ├── fusion_multi_seed_summary.csv
│   ├── loss_minibatch_curve.png
│   ├── loss_minibatch_history.csv
│   └── 其他实验结果
├── run_experiment.py
├── predictions.csv
├── prediction.csv
├── README.md
└── TUNING.md
```

## 数据集

训练集文件：

```text
data/train_data.csv
```

测试集文件：

```text
data/test_data_unlabeled.csv
```

训练集包含 7368 条带标签文本，测试集包含 2457 条无标签文本，共有 10 个类别。

有标签数据按照类别分层划分为：

- 训练子集：80%，5894 条；
- 验证子集：20%，1474 条；
- 随机种子：42。

测试集不参与模型选择、超参数调整或验证评估。最终方案确定后，才使用全部有标签数据重新训练模型并生成测试集预测。

## 环境要求

建议使用 Python 3.10 或更高版本。

安装依赖：

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

本实验实际运行环境：

```text
Python 3.13.12
pandas 3.0.3
scikit-learn 1.9.1
matplotlib 3.10.9
```

## 一键运行

在项目根目录执行：

```bash
python run_experiment.py
```

该命令会自动完成：

1. 从 `data/` 读取训练集和测试集；
2. 检查数据列和样本数量；
3. 使用最终融合特征训练 LinearSVC；
4. 使用全部有标签数据重新拟合模型；
5. 生成 `predictions.csv`；
6. 生成课程提交所需的 `prediction.csv`；
7. 保存最终模型配置到 `results/final_fusion_model_config.json`。

最终融合模型包括：

```text
word-level TF-IDF unigram
+
char_wb-level TF-IDF 3~5 gram
+
LinearSVC(C=0.5)
```

运行结束后检查预测文件：

```bash
python -c "import pandas as pd; a=pd.read_csv('predictions.csv',header=None); b=pd.read_csv('prediction.csv',header=None); print(a.shape,b.shape,a.equals(b))"
```

## 最终模型

最终模型配置如下：

```text
文本处理：保留原始邮件文本
word 特征：TF-IDF unigram
char 特征：char_wb TF-IDF 3~5 gram
word 特征 min_df：2
char 特征 min_df：2
char 特征最大数量：80000
分类器：LinearSVC
C：0.5
random_state：42
```

最终全量训练时：

```text
word 特征数：43277
char 特征数：80000
融合特征数：123277
```

## 模型比较结果

在 `seed=42` 的单次固定验证划分中：

| 模型或特征表示 | 验证 Accuracy | 验证 Macro-F1 |
|---|---:|---:|
| MultinomialNB | 0.9104 | 0.9109 |
| LogisticRegression | 0.9220 | 0.9224 |
| word TF-IDF + LinearSVC | 0.9430 | 0.9431 |
| char TF-IDF + LinearSVC | 0.9315 | 0.9319 |
| char_wb TF-IDF + LinearSVC | 0.9369 | 0.9373 |
| word + char_wb 融合，C=1.0 | 0.9484 | 0.9486 |
| word + char_wb 融合，C=0.5 | **0.9498** | **0.9500** |

字符特征单独使用时不如 word unigram，但与 word unigram 融合后性能提升，说明字符级特征能够补充技术型号、缩写、词形和特殊字符串信息。

## 多随机种子实验

为了检验数据划分带来的波动，使用以下 5 个随机种子重复实验：

```text
42, 123, 2025, 3407, 2026
```

word unigram 模型结果：

```text
Accuracy = 0.9427 ± 0.0026
Macro-F1 = 0.9430 ± 0.0026
```

word + char_wb 融合模型、`C=1.0`：

```text
Accuracy = 0.9459 ± 0.0038
Macro-F1 = 0.9461 ± 0.0039
```

word + char_wb 融合模型、`C=0.5`：

```text
Accuracy = 0.9478 ± 0.0046
Macro-F1 = 0.9480 ± 0.0046
```

`C=0.5` 的融合模型平均 Macro-F1 比 word unigram 提升约 0.50 个百分点。虽然其标准差略大，但平均性能更高，且单次验证结果也更好，因此最终选择 `C=0.5`。

## SVM 正则化参数实验

### word-only 初始实验

在 5000 维 unigram 特征下固定其他条件，只改变 SVM 的 `C`：

| C | 验证 Macro-F1 |
|---:|---:|
| 0.01 | 0.8569 |
| 0.1 | 0.9021 |
| 1 | **0.9176** |
| 10 | 0.9127 |
| 100 | 0.9102 |

较小的 `C` 导致欠拟合；当 `C` 大于 1 后，训练集性能继续提高，但验证性能下降、泛化差距扩大。随后以 `C=1.0` 作为融合模型的初始参数。

### 融合模型局部调参

在最终 word + char_wb 融合特征下比较 `C=0.5、1.0、2.0`：

| C | 验证 Macro-F1 | 泛化差距 |
|---:|---:|---:|
| 0.5 | **0.9500** | **0.0498** |
| 1.0 | 0.9486 | 0.0512 |
| 2.0 | 0.9493 | 0.0506 |

`C=0.5` 在验证集上取得最高 Macro-F1，同时泛化差距最小，因此被锁定为最终参数。

## TF-IDF 词表容量实验

保持 unigram 和 `C=1.0` 不变，只改变词表容量：

| 实际特征数 | 验证 Macro-F1 |
|---:|---:|
| 5000 | 0.9207 |
| 10000 | 0.9311 |
| 20000 | 0.9417 |
| 38178 | **0.9431** |

增加词表容量能够保留更多类别相关词。从 20000 增加到 38178 后，性能仅提升约 0.14 个百分点，边际收益已经较小。

## unigram 与 bigram 对比

在相同的 20000 维特征预算下：

| 特征表示 | 验证 Macro-F1 |
|---|---:|
| unigram | **0.9417** |
| unigram + bigram | 0.9319 |

加入 bigram 后性能下降，可能是因为二元词组增加了稀疏性，并占用了原本可以表示有效单词的特征空间。

## 邮件头消融实验

| 文本输入 | 验证 Macro-F1 |
|---|---:|
| 原始邮件文本 | **0.9431** |
| 仅保留正文 | 0.9248 |

删除邮件头后 Macro-F1 下降约 1.82 个百分点。这说明 `Subject`、`Organization` 和发件人等邮件头信息包含有效的类别线索。因此最终模型保留原始邮件文本。

## 损失曲线

最终模型 `LinearSVC` 不提供逐轮概率损失，因此单独使用：

```text
SGDClassifier(loss="log_loss")
```

观察优化过程。该损失曲线不代表最终 LinearSVC 的训练损失。

模型采用 batch size 为 256 的 mini-batch 增量训练，共训练 20 个 epoch、更新 480 次。每次参数更新后记录当前批次训练损失和固定验证集损失。

由于不同 mini-batch 的样本组成不同，训练损失存在真实的局部波动；移动平均线用于显示总体趋势。最低验证损失为 1.4334，出现在第 480 次更新。这说明在当前观察范围内模型仍处于收敛阶段，尚未出现验证损失反弹。

验证损失是在同一个完整验证集上计算的，因此比单个 mini-batch 的训练损失更平滑。曲线的局部波动来自随机批次训练，不是人为添加的噪声。

## 错误分析

最佳模型的错误主要集中在类别 0、1、2 和 7。

| 类别 | 依据高权重词推断的主题 |
|---:|---|
| 0 | 计算机图形学 |
| 1 | Windows 软件 |
| 2 | 计算机硬件 |
| 7 | 电子、电路与音频 |

最常见的混淆方向：

| 真实类别 | 预测类别 | 数量 |
|---:|---:|---:|
| 7 | 2 | 9 |
| 2 | 1 | 7 |
| 2 | 7 | 6 |

错误案例涉及主板、BIOS、硬盘接口、音频线和显示器信号等内容。这些文本同时包含计算机硬件和电子电路词汇，说明部分错误来自类别之间真实的语义重叠。

## 预测文件

程序会生成两个内容相同的预测文件：

```text
predictions.csv
prediction.csv
```

其中：

- `predictions.csv`：仓库中的预测结果；
- `prediction.csv`：按照课程要求放入提交压缩包的文件。

两个文件均满足：

- 无表头；
- 无索引；
- 每行一个预测标签；
- 共 2457 行；
- 包含 10 个类别。

## 详细实验脚本

| 脚本 | 功能 |
|---|---|
| `baseline.py` | 朴素贝叶斯基线 |
| `logistic_baseline.py` | 逻辑回归基线 |
| `svm_baseline.py` | 线性 SVM 基线 |
| `svm_c_tuning.py` | word-only SVM 参数实验 |
| `fusion_c_tuning.py` | 融合模型 C 参数实验 |
| `fusion_c05_multi_seed.py` | C=0.5 融合模型多随机种子实验 |
| `svm_feature_count_tuning.py` | TF-IDF 词表容量实验 |
| `svm_feature_compare.py` | unigram 与 bigram 对比 |
| `svm_header_ablation.py` | 邮件头消融实验 |
| `char_tfidf_experiment.py` | 字符级 TF-IDF 对照实验 |
| `word_char_fusion_experiment.py` | word 与 char_wb 特征融合 |
| `multi_seed_experiment.py` | word unigram 多随机种子实验 |
| `fusion_multi_seed_experiment.py` | C=1.0 融合模型多随机种子实验 |
| `loss_curve_experiment.py` | 按 epoch 记录损失的早期对照实验 |
| `loss_minibatch_experiment.py` | mini-batch Log Loss 曲线 |
| `svm_error_analysis.py` | 分类报告与混淆矩阵 |
| `inspect_errors.py` | 代表词和典型错误案例 |
| `final_fusion_predict.py` | 最终融合模型训练与预测 |

验收时推荐直接运行根目录中的：

```bash
python run_experiment.py
```

## 结果文件

`results/` 中保存：

- `final_fusion_model_config.json`：最终融合模型配置；
- `char_tfidf_results.csv`：字符级特征实验；
- `word_char_fusion_results.csv`：特征融合实验；
- `fusion_c_tuning_results.csv`：融合模型 C 参数实验；
- `fusion_c05_multi_seed_results.csv`：C=0.5 融合模型逐种子结果；
- `fusion_c05_multi_seed_summary.csv`：C=0.5 融合模型均值和标准差；
- `fusion_multi_seed_results.csv`：C=1.0 融合模型逐种子结果；
- `fusion_multi_seed_summary.csv`：C=1.0 融合模型均值和标准差；
- `multi_seed_results.csv`：word unigram 多种子结果；
- `multi_seed_summary.csv`：word unigram 均值和标准差；
- `loss_minibatch_curve.png`：mini-batch 损失曲线；
- `loss_minibatch_history.csv`：mini-batch 损失记录；
- SVM 参数实验结果；
- TF-IDF 词表容量实验结果；
- unigram/bigram 对比结果；
- 邮件头消融结果；
- 混淆矩阵；
- 类别代表词；
- 典型错误案例。

## 可复现性

- 所有实验固定随机种子；
- 验证集按标签分层划分；
- TF-IDF 只在训练子集上拟合；
- 验证集仅使用已拟合的向量器转换；
- 测试集不用于调参或模型选择；
- 最终方案确定后，才使用全部有标签数据重新训练；
- 模型参数、验证结果、损失记录和预测文件均已保存；
- 根目录 `run_experiment.py` 可直接生成最终 `C=0.5` 融合模型预测。