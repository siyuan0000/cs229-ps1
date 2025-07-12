# CS229 Problem Set 1 - Machine Learning

这是斯坦福大学CS229机器学习课程的作业1，包含以下内容：

## 项目结构

```
ps1-full/
├── src/                    # 源代码目录
│   ├── doubledescent/      # 双下降现象实验
│   ├── featuremaps/        # 特征映射实验
│   ├── gd_convergence/     # 梯度下降收敛实验
│   ├── implicitreg/        # 隐式正则化实验
│   └── lwr/               # 局部加权回归
├── tex/                    # LaTeX文档
│   ├── doubledescent/      # 双下降相关文档
│   ├── featuremaps/        # 特征映射相关文档
│   ├── gd_convergence/     # 梯度下降相关文档
│   ├── implicitreg/        # 隐式正则化相关文档
│   └── lwr/               # 局部加权回归相关文档
├── environment.yml         # Conda环境配置
└── README.md              # 项目说明
```

## 环境设置

使用Conda创建环境：

```bash
conda env create -f environment.yml
conda activate cs229-ps1
```

## 实验内容

1. **双下降现象 (Double Descent)**: 研究模型复杂度与泛化性能的关系
2. **特征映射 (Feature Maps)**: 探索不同特征映射对模型性能的影响
3. **梯度下降收敛 (GD Convergence)**: 分析梯度下降算法的收敛性质
4. **隐式正则化 (Implicit Regularization)**: 研究优化算法中的隐式正则化效应
5. **局部加权回归 (Locally Weighted Regression)**: 实现和优化局部加权回归算法

## 运行实验

每个实验目录都包含相应的Python脚本和数据集：

```bash
# 运行双下降实验
cd src/doubledescent
python doubledescent.py

# 运行特征映射实验
cd src/featuremaps
python featuremap.py

# 运行梯度下降收敛实验
cd src/gd_convergence
python experiment.py

# 运行隐式正则化实验
cd src/implicitreg
python linear.py
python qp.py

# 运行局部加权回归实验
cd src/lwr
python lwr.py
python tau.py
```

## 生成报告

使用LaTeX编译生成PDF报告：

```bash
cd tex
make
```

## 许可证

本项目仅用于学习目的，遵循斯坦福大学CS229课程的学术诚信政策。
