# 基于 RoBERTa 的中文仇恨言论检测
本项目实现了一个**仇恨言论检测**模型，采用了**RoBERTa**、**BiGRU**和**TextCNN**相结合的架构进行文本分类。该模型旨在基于预训练的RoBERTa模型，通过结合BiGRU和TextCNN模型，进行中文仇恨言论的识别。您可以在[这里](https://www.chatlwh.com/posts/20240925/article.html)看到更详细的说明。

## 项目结构

- `train.py`：用于训练模型的脚本。
- `test.py`：用于在测试集上评估训练好的模型，并输出评估指标。
- `inference.py`：用于对用户输入文本进行推理，判断其是否为仇恨言论。
- `Data/`：存放训练集、验证集和测试集CSV文件的文件夹。
- `chinese_roberta_wwm_ext/`: 存放从 Huggingface 下载的预训练模型（chinese-roberta-wwm-ext）。
- `best_model.pth`：训练过程中保存的最优模型。
- `README.md`：此文件。

## 环境要求

- Python 3.7+
- Pytorch 2.6.0
- transformers 4.48.3

## 数据集

数据集来源：[COLD](https://github.com/thu-coai/COLDataset)

## 训练模型

可以直接运行 `train.py` 文件或者使用下列命令：

```
python train.py --model_path ./chinese_roberta_wwm_ext --max_length 128 --batch_size 64 --train_path Data/train.csv --dev_path Data/dev.csv --test_path Data/test.csv --epochs 5
```

此命令将使用指定的路径训练模型，并保存最佳模型。

## 评估模型

训练完成后，使用以下命令在测试集上评估模型：

```
python test.py --model_path best_model.pth --test_path Data/test.csv --tokenizer_name ./chinese_roberta_wwm_ext --device cuda
```

此命令将加载最佳保存的模型，并在测试集上进行评估，输出准确率、精确率、召回率和F1值等指标。

## 推理

要对用户输入的单条文本进行推理，使用以下命令：

```
python inference.py --model_path best_model.pth --tokenizer_name ./chinese_roberta_wwm_ext --device cuda
```

你将被提示输入一句话，模型会将其分类为仇恨言论（1）或非仇恨言论（0）。

## 示例

以下是进行推理的示例：

```
Enter a sentence for prediction: 你们这些人真恶心！
This sentence is classified as Hate Speech.
```

## 结果

模型在测试集上取得了以下性能：

- **准确率（Accuracy）**: 82.85%
- **精确率（Precision）**: 74.15%
- **召回率（Recall）**: 87.00%
- **F1-Score**: 80.06%

## 许可证

本项目采用MIT许可证 - 详情请见 **LICENSE** 文件。

## 致谢

项目思路来源于[(Chinese Hate Speech detection method Based on RoBERTa-WWM)](https://aclanthology.org/2023.ccl-1.44/)，向其作者表示感谢。



