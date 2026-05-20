# DREAM

## 中文说明

本仓库为 DREAM 项目实现，基于 [Ultralytics](https://github.com/ultralytics/ultralytics) 构建。  
感谢 Ultralytics 团队的开源贡献。

### 论文引用

我们的 DREAM 论文已被 *Expert Systems with Applications* 接收/发表。若本项目对您的研究有帮助，欢迎引用我们的文章：

Rong Zhu, Linfeng Shi, and Zheng He. "DREAM: Drone-based real-time efficient adaptive multidomain network." *Expert Systems with Applications*, 327, 132852, 2026. https://doi.org/10.1016/j.eswa.2026.132852

论文网页：https://www.sciencedirect.com/science/article/pii/S0957417426017653

```bibtex
@article{ZHU2026132852,
  title = {DREAM: Drone-based real-time efficient adaptive multidomain network},
  journal = {Expert Systems with Applications},
  volume = {327},
  pages = {132852},
  year = {2026},
  issn = {0957-4174},
  doi = {https://doi.org/10.1016/j.eswa.2026.132852},
  url = {https://www.sciencedirect.com/science/article/pii/S0957417426017653},
  author = {Rong Zhu and Linfeng Shi and Zheng He}
}
```

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

### Citation

Our DREAM paper has been accepted/published in *Expert Systems with Applications*. If this repository is helpful for your research, please consider citing our work:

Rong Zhu, Linfeng Shi, and Zheng He. "DREAM: Drone-based real-time efficient adaptive multidomain network." *Expert Systems with Applications*, 327, 132852, 2026. https://doi.org/10.1016/j.eswa.2026.132852

Article page: https://www.sciencedirect.com/science/article/pii/S0957417426017653

```bibtex
@article{ZHU2026132852,
  title = {DREAM: Drone-based real-time efficient adaptive multidomain network},
  journal = {Expert Systems with Applications},
  volume = {327},
  pages = {132852},
  year = {2026},
  issn = {0957-4174},
  doi = {https://doi.org/10.1016/j.eswa.2026.132852},
  url = {https://www.sciencedirect.com/science/article/pii/S0957417426017653},
  author = {Rong Zhu and Linfeng Shi and Zheng He}
}
```

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
