# PPO 月球着陆器

在 LunarLander-v2 强化学习环境中实现近端策略优化（PPO）算法。

## 概述

本项目实现了 PPO 算法，用于训练智能体在 Gymnasium 的 [LunarLander-v2](https://gymnasium.farama.org/environments/box2d/lunar_lander/) 环境中成功着陆月球着陆器。实现包括：

- 可配置架构的 Actor-Critic 神经网络
- GAE（广义优势估计）用于优势计算
- TensorBoard 日志记录用于训练可视化
- 模型检查点和评估工具
- 训练和评估脚本

## 要求

- Python 3.12+
- UV 包管理器（推荐）或 pip

## 安装

### 使用 UV（推荐）

```bash
# 如果尚未安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
git clone https://github.com/wangyunqing7/ppo-lunar-lander.git
cd ppo-lunar-lander

# 安装依赖
uv sync
```

### 使用 pip

```bash
# 克隆仓库
git clone https://github.com/wangyunqing7/ppo-lunar-lander.git
cd ppo-lunar-lander

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows系统: .venv\Scripts\activate

# 安装依赖
pip install -e .
```

## 项目结构

```
ppo-lunar-lander/
├── src/
│   └── ppo_lunar_lander/
│       ├── agents/
│       │   ├── __init__.py
│       │   └── ppo.py           # PPO 智能体实现
│       ├── models/
│       │   ├── __init__.py
│       │   └── networks.py      # Actor-Critic 网络
│       └── __init__.py
├── train.py                     # 训练脚本
├── evaluate.py                  # 评估脚本
├── visualize.py                 # 可视化脚本
├── record_video.py              # 视频录制脚本
├── checkpoints/                 # 保存的模型检查点（训练时创建）
├── logs/                        # TensorBoard 日志（训练时创建）
├── pyproject.toml              # 项目配置
├── lunar_lander_best.gif        # 最佳模型演示视频
└── README.md
```

## 使用方法

### 训练

从头开始训练 PPO 智能体：

```bash
# 使用默认超参数进行基础训练
python train.py

# 使用自定义超参数训练
python train.py \
    --total-timesteps 1000000 \
    --learning-rate 3e-4 \
    --num-steps 2048 \
    --batch-size 64 \
    --num-epochs 10 \
    --hidden-dim 256 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-epsilon 0.2

# 在 GPU 上训练（如果可用）
python train.py --device cuda
```

**训练参数：**

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--env-id` | LunarLander-v2 | Gymnasium 环境 ID |
| `--seed` | 42 | 随机种子，用于可重复性 |
| `--total-timesteps` | 1,000,000 | 总训练步数 |
| `--learning-rate` | 3e-4 | 优化器学习率 |
| `--num-steps` | 2048 | 每次策略更新的步数 |
| `--batch-size` | 64 | 优化的小批量大小 |
| `--num-epochs` | 每次更新的优化轮数 | 10 |
| `--gamma` | 0.99 | 折扣因子 |
| `--gae-lambda` | 0.95 | GAE lambda 参数 |
| `--clip-epsilon` | 0.2 | PPO 裁剪参数 |
| `--c1` | 1.0 | 价值函数损失系数 |
| `--c2` | 0.01 | 熵奖励系数 |
| `--hidden-dim` | 256 | 隐藏层大小 |
| `--num-hidden` | 2 | 隐藏层数量 |
| `--save-dir` | checkpoints | 模型保存目录 |
| `--log-dir` | logs | TensorBoard 日志目录 |
| `--save-freq` | 100,000 | 模型保存频率（步数） |
| `--eval-freq` | 10,000 | 评估频率 |
| `--device` | auto | 设备（cpu/cuda/auto） |

### 监控训练

使用 TensorBoard 监控训练进度：

```bash
tensorboard --logdir logs
```

然后在浏览器中打开 `http://localhost:6006`

### 评估

评估训练好的模型：

```bash
# 不渲染进行评估
python evaluate.py --model-path checkpoints/ppo_final.pt --num-episodes 100

# 渲染进行评估（观看智能体）
python evaluate.py --model-path checkpoints/ppo_final.pt --num-episodes 10 --render
```

### 可视化

从 TensorBoard 日志生成训练图表：

```bash
python visualize.py --log-dir logs --output training_plot.png --smooth 10
```

### 视频录制

录制智能体表现的视频：

```bash
# 生成 GIF 格式
python record_video.py \
    --model-path checkpoints/ppo_best.pt \
    --output demo.gif \
    --num-episodes 3

# 生成 MP4 格式（需要 opencv-python）
python record_video.py \
    --model-path checkpoints/ppo_best.pt \
    --output demo.mp4 \
    --num-episodes 5 \
    --fps 30
```

## 算法详解

### PPO（近端策略优化）

PPO 是一种策略梯度方法，使用裁剪的代理目标来防止过大的策略更新。关键组件：

1. **Actor-Critic 架构**：
   - Actor：输出动作的概率分布
   - Critic：估计状态价值函数 V(s)

2. **GAE（广义优势估计）**：
   - 计算具有偏差-方差权衡的优势
   - λ 参数控制权衡（默认 0.95）

3. **裁剪代理目标**：
   - 防止破坏性的大策略更新
   - ε 参数控制裁剪范围（默认 0.2）

4. **熵奖励**：
   - 通过惩罚低熵策略来鼓励探索

### 网络架构

- 输入：状态观测（LunarLander-v2 为 8 维）
- 隐藏层：全连接，使用 Tanh 激活
- 输出：
  - Actor：动作 logits（4 个离散动作）
  - Critic：状态值（标量）

### 超参数

默认超参数针对 LunarLander-v2 进行了调优：

- 学习率：3e-4
- 折扣因子（γ）：0.99
- GAE lambda（λ）：0.95
- PPO 裁剪（ε）：0.2
- 价值损失系数：1.0
- 熵系数：0.01
- 最大梯度范数：0.5
- 批量大小：64
- 优化轮数：10

## 预期结果

使用默认超参数和 100 万步训练：

- **平均奖励**：应达到 200+（成功阈值）
- **成功率**：>80% 的 episode 奖励 >200
- **训练时间**：CPU 上约 30-60 分钟，GPU 上约 5-10 分钟

环境认为着陆成功的条件：
- 着陆器在两个旗帜之间接触地面
- 着陆器静止不动
- 着陆器保持直立（角度接近 0）
- 奖励 > 200

## 训练结果

本项目已完成 200 万步训练，达到以下性能：

- **最佳平均奖励**：273.33
- **最终平均奖励**：268.63 ± 13.37
- **成功率**：84%（100 轮测试中 84 轮成功）
- **最高奖励**：326.33
- **训练时长**：53 分钟（CPU）

**演示视频**：
- `lunar_lander_best.gif`（推荐）：5 个 episode，平均奖励 285.01
- `lunar_lander_demo.gif`：2 个 episode 快速演示

## 故障排除

**训练不稳定**：
- 降低学习率（尝试 1e-4）
- 增加优化轮数（尝试 20）
- 调整熵系数（尝试 0.005 或 0.02）

**智能体未学习**：
- 检查 TensorBoard 日志中是否有策略崩溃（熵 → 0）
- 增加熵系数以获得更多探索
- 使用 `--render` 验证环境是否正确渲染

**内存不足错误**：
- 减少批量大小
- 减少每次更新的步数
- 使用 CPU 代替 GPU

## 许可证

本项目是开源的，采用 MIT 许可证。

## 参考资料

- [Schulman et al., "Proximal Policy Optimization Algorithms" (2017)](https://arxiv.org/abs/1707.06347)
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [OpenAI 深度强化学习入门](https://spinningup.openai.com/en/latest/)

## 贡献

欢迎贡献！随时可以提出问题或提交拉取请求。

## 致谢

- OpenAI 提供 PPO 算法
- Gymnasium 团队提供 LunarLander 环境
- PyTorch 团队提供深度学习框架
