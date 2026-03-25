# 基于不确定性感知时序建模的航空发动机剩余寿命预测与维护预警方法研究

本项目基于 NASA CMAPSS 数据集实现一个可直接运行的小型 PyTorch 实验框架，先在 `FD001` 上完成航空发动机剩余寿命预测，再保留迁移到 `FD003` 的接口。框架同时支持：

- 点预测 TCN baseline
- 异方差高斯输出头的不确定性 TCN
- 训练、验证、测试评估
- 结果可视化
- 基于预测区间下界和不确定性的四级预警示意

## 目录结构

```text
project/
├── data/
├── configs/
│   ├── fd001_tcn_point.yaml
│   ├── fd001_tcn_uncertainty.yaml
│   └── fd003_tcn_uncertainty.yaml
├── datasets/
│   └── cmapss_dataset.py
├── models/
│   ├── tcn.py
│   ├── heads.py
│   └── tcn_rul_model.py
├── losses/
│   └── gaussian_nll.py
├── metrics/
│   ├── rmse.py
│   ├── phm_score.py
│   └── uncertainty_metrics.py
├── utils/
│   ├── seed.py
│   ├── scaler.py
│   ├── logger.py
│   ├── plotting.py
│   └── warning.py
├── scripts/
│   ├── preprocess_cmapss.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── tests/
│   └── test_phm_score.py
├── results/
│   ├── figures/
│   ├── checkpoints/
│   └── logs/
├── requirements.txt
└── README.md
```

## 环境安装

建议使用 Python 3.10 及以上版本。

```bash
pip install -r requirements.txt
```

## 数据放置方式

将 NASA CMAPSS 原始文件放到 `data/` 目录下，或直接在项目根目录保留 `CMAPSSData.zip`。脚本会优先读取：

- `data/train_FD001.txt`
- `data/test_FD001.txt`
- `data/RUL_FD001.txt`
- `data/train_FD003.txt`
- `data/test_FD003.txt`
- `data/RUL_FD003.txt`

如果 `data/` 中缺少目标子集文件，而根目录存在 `CMAPSSData.zip`，脚本会自动提取所需文件。

## 训练命令

点预测：

```bash
python scripts/train.py --config configs/fd001_tcn_point.yaml
```

不确定性模型：

```bash
python scripts/train.py --config configs/fd001_tcn_uncertainty.yaml
```

迁移到 FD003：

```bash
python scripts/train.py --config configs/fd003_tcn_uncertainty.yaml
```

## 评估命令

```bash
python scripts/evaluate.py --config configs/fd001_tcn_uncertainty.yaml
```

## 可视化命令

```bash
python scripts/visualize.py --config configs/fd001_tcn_uncertainty.yaml
```

## 数据预处理命令

```bash
python scripts/preprocess_cmapss.py --config configs/fd001_tcn_uncertainty.yaml
```

## 配置说明

所有核心参数均通过 YAML 配置管理。

`data`：

- `subset`：数据子集，当前支持 `FD001`、`FD003`
- `data_dir`：原始数据目录
- `rul_clip`：训练和测试标签的 RUL 截断阈值
- `window_size`：滑动窗口长度
- `val_ratio`：按 `unit_id` 划分验证集比例
- `include_op_settings`：是否将 `op1/op2/op3` 纳入输入
- `var_threshold`：近零方差筛选阈值，仅作用于 `s1~s21`
- `padding_mode`：测试和轨迹可视化时左侧补齐方式，支持 `repeat`、`zero`

`model`：

- `type`：`point` 或 `uncertainty`
- `num_channels`：TCN 每层通道数
- `kernel_size`：因果卷积核大小
- `dropout`：dropout 概率

`training`：

- `batch_size`、`epochs`、`lr`
- `weight_decay`
- `optimizer`：当前实现支持 `Adam`、`AdamW`，默认使用 `AdamW`
- `scheduler`：当前实现为 `ReduceLROnPlateau`
- `scheduler_patience`、`scheduler_factor`
- `early_stopping_monitor`：当前按 `val_loss` 早停
- `early_stopping_patience`
- `seed`

`warning`：

- `thresholds.normal`
- `thresholds.watch`
- `thresholds.alert`
- `sigma_escalation`
- `sigma_threshold`

`output`：

- `results_dir`
- `figures_dir`
- `checkpoint_dir`
- `logs_dir`

## 指标说明

- `RMSE`：均方根误差
- `PHM Score`：CMAPSS 常用累计惩罚分数。高估惩罚更重，定义如下：
  - `d = pred - true`
  - `d < 0` 时，`exp(-d / 13) - 1`
  - `d >= 0` 时，`exp(d / 10) - 1`
- `Gaussian NLL`：不确定性模型的训练和早停指标
- `PICP`：预测区间覆盖率
- `MPIW`：预测区间平均宽度

## 预警逻辑说明

不确定性模型输出 `mu` 与 `logvar`，其中：

- `logvar = log(sigma^2)`
- `sigma = exp(0.5 * logvar)`

预警模块使用 95% 置信下界：

- `lower = mu - 1.96 * sigma`

基础预警等级：

- `lower > 80`：正常
- `50 < lower <= 80`：关注
- `20 < lower <= 50`：预警
- `lower <= 20`：危险

若 `sigma > sigma_threshold` 且当前等级不是“危险”，则上调一级。

## 实现细节

- TCN 使用严格因果卷积，只在序列左侧 padding
- 训练/验证划分按 `unit_id`，同一发动机不会同时出现在 train 和 val
- 近零方差筛选和标准化仅使用训练集拟合，再应用到 val/test
- benchmark 测试集每台发动机仅保留一个最后窗口，标签直接来自 `RUL_FDxxx.txt`
- 训练完成后自动在测试集评估，并将结果追加到 `results/results_summary.csv`
