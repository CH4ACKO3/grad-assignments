# CSC6022 - Machine Learning

本文件夹包含CSC6022机器学习课程的作业文件和代码。

## 课程信息

- **课程代码**: CSC6022
- **课程名称**: Machine Learning
- **课程描述**: This course is to provide a comprehensive introduction to machine learning. Topics include Supervised Learning (Regression, Classification), Over-fitting, Regularization, Deep Double Descent, Bias Variance Trade-off, Curse of Dimensionality, Validation, Bayesian Classifiers, Nearest Neighbor, Logistic Regression, Decision trees, Support vector machines, MLE, MAP, Bayesian inference, EM and VAE algorithm, Multilayer Feedforward Networks, Deep Networks, Unsupervised Learning (clustering and PCA), and potentially self-supervised learning and Reinforcement Learning.

## 文件夹结构

```
CSC6022-Machine_Learning/
├── prompt/                 # 作业题目PDF文件
├── code/                   # 代码文件
│   ├── data/              # 数据文件
│   └── result/            # 结果文件（图表、输出等）
├── tex/                   # 编译生成的PDF和辅助文件
├── assignment_1.tex       # 作业1的LaTeX源文件
├── environment.yml        # Conda环境配置文件
└── README.md             # 本文件
```

## 环境配置

本课程使用独立的Conda环境，配置文件为 `environment.yml`。

### 创建环境

```bash
conda env create -f environment.yml
```

### 激活环境

```bash
conda activate csc6022_ml_env
```

### 更新环境

```bash
conda env update -f environment.yml --prune
```

## 使用说明

1. **作业题目**: 将作业PDF文件放在 `prompt/` 文件夹中
2. **代码开发**: 在 `code/` 文件夹中编写代码
3. **数据管理**: 将数据文件放在 `code/data/` 中
4. **结果输出**: 生成的图表和结果保存在 `code/result/` 中
5. **LaTeX编写**: 在根目录的 `.tex` 文件中编写作业报告
6. **PDF生成**: 编译后的PDF文件会保存在 `tex/` 文件夹中

## Git追踪

所有必要的文件夹都已配置为可被Git追踪（通过.gitkeep文件）。


