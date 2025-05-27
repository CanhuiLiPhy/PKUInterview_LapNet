# Package LapNets - 神经网络变分蒙特卡罗 (Neural Network Variational Monte Carlo)

这是一个基于PyTorch实现的神经网络变分蒙特卡罗（NNVMC）量子化学计算包，使用LapNet架构进行分子和原子系统的量子态计算。

## 项目概述

本项目实现了基于Transformer架构的LapNet神经网络波函数，用于量子蒙特卡罗计算。主要特性包括：

- **LapNet架构**: 使用Transformer层和稀疏注意力机制
- **多行列式波函数**: 支持多个Slater行列式的线性组合
- **Jastrow因子**: 处理电子相关性造成的尖角条件
- **包络函数**: 确保正确的边界条件
- **MCMC采样**: Metropolis-Hastings算法进行电子坐标采样
- **自适应步长**: 动态调整MCMC步长以优化接受率

## 测试结果

在 `test_results/` 目录中包含了以下系统的测试结果：

- **H2分子**: `H2_training_plots_20250524_194309.png` - 氢分子的训练过程可视化
- **氢原子**: `H_atom_training_plots_20250524_191150.png` - 氢原子的训练过程可视化

这些图表展示了训练过程中能量收敛、方差变化和接受率等关键指标。

本项目由于缺乏gpu，没有做针对cuda的测试，因此默认使用cpu训练。

## 依赖包安装

请确保安装以下Python包：

```bash
# 核心依赖
pip install torch>=1.12.0
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0
pip install tqdm>=4.64.0


或者创建requirements.txt文件并安装：

```bash
# requirements.txt内容
torch>=1.12.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
```

然后运行：
```bash
pip install -r requirements.txt
```

## 配置文件说明

配置文件位于 `configs/` 目录，使用JSON格式。以下是各个参数的详细说明：

### 系统配置 (system)

```json
{
  "system": {
    "n_electrons_up": 1,        // 自旋向上电子数量
    "n_electrons_down": 1,      // 自旋向下电子数量
    "atom_coords": [            // 原子坐标 [x, y, z] (单位: Bohr)
      [-1, 0.0, 0.0], 
      [1, 0.0, 0.0]
    ],
    "atom_types": [1, 1],       // 原子类型（暂未使用）
    "nuclear_charges": [1.0, 1.0], // 核电荷数
    "d_model": 128,             // Transformer隐藏层维度
    "n_layers": 4,              // Transformer层数
    "n_heads": 8,               // 注意力头数
    "n_determinants": 8,        // 行列式数量
    "dropout": 0.1,             // Dropout概率
    "use_layernorm": true,      // 是否使用层归一化
    "use_jastrow": true,        // 是否使用Jastrow因子
    "jastrow_init": 0.01,       // Jastrow因子初始化尺度
    "envelope_type": "isotropic", // 包络函数类型
    "use_cross_attention": true  // 是否使用交叉注意力
  }
}
```

### MCMC配置 (mcmc)

```json
{
  "mcmc": {
    "step_size": 0.05,          // 初始MCMC步长
    "n_steps": 10,              // 每个训练步骤的MCMC步数
    "init_width": 1.0,          // 电子初始位置的高斯宽度
    "adapt_step_size": true,    // 是否自适应调整步长
    "step_size_min": 0.001,     // 最小步长
    "step_size_max": 0.2,       // 最大步长
    "target_accept_rate": 0.5,  // 目标接受率
    "accept_rate_window": 10    // 接受率计算窗口大小
  }

}
```

### 哈密顿量配置 (hamiltonian)

```json
{
  "hamiltonian": {
    "method": "log",            // 计算方法 
    "grad_clip": 30.0          // 梯度裁剪阈值
  }
}
```

### 损失函数配置 (loss)

```json
{
  "loss": {
    "type": "vmc",              // 损失函数类型
    "params": {
      "clip_local_energy": 8.0,    // 局域能量裁剪阈值
      "remove_outliers": true,     // 是否移除异常值
      "outlier_width": 12.0        // 异常值检测宽度
    }
  }
}
```

### 优化器配置 (optimizer)

```json
{
  "optimizer": {
    "lr": 0.0003,               // 学习率
    "beta1": 0.9,               // Adam参数beta1
    "beta2": 0.999,             // Adam参数beta2
    "eps": 1e-8,                // Adam参数epsilon
    "weight_decay": 1e-6,       // 权重衰减
    "grad_clip": 0.5            // 梯度裁剪
  }
}
```

### 训练配置 (training)

```json
{
  "training": {
    "n_iterations": 500,        // 训练迭代次数
    "batch_size": 512,          // 批次大小
    "burn_in_steps": 20,        // 预热步数
    "save_frequency": 200,      // 保存检查点频率
    "plot_frequency": 25        // 绘图频率
  }
}
```

### 输出目录配置

```json
{
  "checkpoint_dir": "checkpoints/lapnet_h2_test",  // 检查点保存目录
  "plots_dir": "plots/lapnet_h2_test",            // 训练图表保存目录
  "results_dir": "results_json"                   // 结果JSON文件保存目录（可选，默认为results_json）
}
```

### 包络函数类型说明

- `"isotropic"`: 各向同性包络函数
- `"diagonal"`: 对角包络函数
- `"full"`: 完整包络函数
- `"sto"`: Slater型轨道包络函数
- `"null"`: 无包络函数

## 运行方法

### 基本运行命令

使用以下命令启动训练：

```bash
python train.py --config configs/lapnet_h2_test.json
```

### 可用的配置文件

1. **氢分子 (H2)**:
   ```bash
   python train.py --config configs/lapnet_h2_test.json
   ```

2. **氢原子 (H)**:
   ```bash
   python train.py --config configs/lapnet_h_test.json
   ```

3. **氢化锂 (LiH)**:
   ```bash
   python train.py --config configs/lapnet_lih_test.json
   ```

### 命令行参数

训练脚本支持以下命令行参数：

- `--config`: 配置文件路径（必需）
- `--device`: 计算设备（'cuda' 或 'cpu'，可选）
- `--checkpoint`: 从检查点恢复训练（可选）

示例：
```bash
python train.py --config configs/lapnet_h2_test.json
```

### 使用Checkpoint继续训练

项目支持从保存的checkpoint文件继续训练，这对于长时间训练或中断后恢复非常有用。

#### 1. 从checkpoint继续训练
```bash
# 从保存的checkpoint继续训练
python train.py --config configs/lapnet_h2_test.json --checkpoint checkpoints/lapnet_h2_test/checkpoint_200.pt
```

#### 2. 查看可用的checkpoint
```bash
# 列出所有保存的checkpoint文件
ls checkpoints/lapnet_h2_test/
# 输出示例：
# checkpoint_200.pt
# checkpoint_400.pt
# checkpoint_600.pt
```

#### 3. 检查checkpoint内容
使用提供的检查工具查看checkpoint的详细信息：
```bash
python inspect_checkpoint.py checkpoints/lapnet_h2_test/checkpoint_200.pt
```

这会显示：
- 训练进度（当前迭代次数）
- 系统配置信息
- 训练历史数据
- 神经网络参数统计
- 最新的能量和方差值

#### 4. 提取配置文件
从checkpoint中提取配置并保存为新的JSON文件：
```bash
python inspect_checkpoint.py checkpoints/lapnet_h2_test/checkpoint_200.pt --save-config extracted_config.json
```

#### Checkpoint工作原理

当使用`--checkpoint`参数时，程序会：
1. **加载神经网络权重** - 恢复到保存时的网络状态
2. **恢复优化器状态** - 包括Adam的动量信息，确保训练平滑继续
3. **载入训练历史** - 能量、方差、接受率等历史数据
4. **恢复MCMC状态** - 电子坐标位置，避免重新burn-in
5. **从正确的迭代开始** - 继续未完成的训练

**注意**: 程序会自动计算剩余的训练迭代次数。例如，如果配置文件设置总共训练1000次，而checkpoint保存在第200次，程序将继续训练800次。

## 输出文件

训练过程会生成以下文件：

1. **检查点文件**: 保存在配置文件指定的 `checkpoint_dir` 目录
2. **训练图表**: 保存在配置文件指定的 `plots_dir` 目录
3. **结果JSON文件**: 保存在配置文件指定的 `results_dir` 目录（默认为 `results_json/`），包含完整的训练历史和最终结果
4. **训练日志**: 输出到控制台，包含能量、方差、接受率等信息

## 项目结构

```
Package_LapNets/
├── configs/                    # 配置文件
│   ├── lapnet_h2_test.json    # H2分子配置
│   ├── lapnet_h_test.json     # 氢原子配置
│   └── lapnet_lih_test.json   # LiH分子配置
├── test_results/              # 测试结果
│   ├── H2_training_plots_*.png
│   └── H_atom_training_plots_*.png
├── results_json/              # 训练结果JSON文件
│   └── results_*.json
├── train.py                   # 主训练脚本
├── inspect_checkpoint.py      # Checkpoint检查工具
├── networks.py                # 神经网络波函数
├── transformer_blocks.py      # Transformer组件
├── envelopes.py              # 包络函数
├── hamiltonian.py            # 哈密顿量计算
├── mcmc.py                   # MCMC采样
├── loss.py                   # 损失函数
└── __init__.py              # 包初始化
```

## 常见问题

1. **GPU内存不足**: 减小 `batch_size` 或 `d_model`
2. **收敛慢**: 调整学习率 `lr` 或增加 `n_iterations`
3. **接受率过低/过高**: 调整 `step_size` 或启用 `adapt_step_size`
4. **梯度爆炸**: 调整 `grad_clip` 参数

## 理论背景

本项目基于变分蒙特卡罗方法，使用神经网络波函数来近似量子系统的基态。LapNet架构结合了：

- **Transformer注意力机制**: 捕获长程电子相关性
- **多行列式结构**: 提供更灵活的波函数表示
- **Jastrow因子**: 处理电子-电子相关性
- **包络函数**: 确保正确的渐近行为

这种方法在保持高精度的同时，能够扩展到更大的分子系统。

---

更多详细信息和理论背景，请参考相关论文和文档。 