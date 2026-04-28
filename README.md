# DREAM

## 中文说明

本仓库为 DREAM 项目实现，基于 [Ultralytics](https://github.com/ultralytics/ultralytics) 构建。  
感谢 Ultralytics 团队的开源贡献。

### 仓库包含内容

- 模型配置文件：`ultralytics/cfg/models/dream/DREAM.yaml`
- 模块设计实现：`ultralytics/nn/modules/block_designer.py`

### 基本使用流程

1. 准备数据集，并按需配置数据集配置文件（如 `data.yaml`）。
2. 确认模型配置文件 `ultralytics/cfg/models/dream/DREAM.yaml`。
3. 运行训练：

```bash
python train.py
```

可根据实验需求进一步调整训练参数与数据配置。

### 致谢

本项目基于 Ultralytics 代码库开发。  
Ultralytics GitHub: https://github.com/ultralytics/ultralytics

---

## English

This repository contains the DREAM implementation built on top of [Ultralytics](https://github.com/ultralytics/ultralytics).  
We sincerely thank the Ultralytics team for their open-source contributions.

### Included in This Repository

- Model configuration file: `ultralytics/cfg/models/dream/DREAM.yaml`
- Module design implementation: `ultralytics/nn/modules/block_designer.py`

### Basic Workflow

1. Prepare your dataset and configure the dataset file (e.g., `data.yaml`).
2. Confirm the model configuration at `ultralytics/cfg/models/dream/DREAM.yaml`.
3. Run training:

```bash
python train.py
```

You can further adjust training arguments and dataset settings based on your experiment setup.

### Acknowledgement

This project is developed based on the Ultralytics codebase.  
Ultralytics GitHub: https://github.com/ultralytics/ultralytics
