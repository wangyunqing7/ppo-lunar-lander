# PPO Lunar Lander - 可视化演示

本目录包含训练好的PPO agent在LunarLander-v3环境中的可视化演示。

## 📹 生成的视频文件

### 1. lunar_lander_best.gif（推荐）
- **描述**: 最佳性能演示 - 5个完整episode
- **分辨率**: 640x640
- **帧率**: 25 FPS
- **时长**: 42.6秒
- **文件大小**: 4.31 MB
- **平均奖励**: 285.01（优秀！）
- **Episode详情**:
  - Episode 1: 265.85
  - Episode 2: 281.08
  - Episode 3: 290.40
  - Episode 4: 286.05
  - Episode 5: 301.69

### 2. lunar_lander_demo.gif
- **描述**: 快速演示 - 2个episodes
- **分辨率**: 512x512
- **帧率**: 20 FPS
- **时长**: 15.4秒
- **文件大小**: 1.15 MB
- **平均奖励**: 164.93

## 🎬 如何查看

### 方法1: 直接打开
双击GIF文件，用浏览器或图片查看器打开

### 方法2: 命令行
```bash
# Linux
xdg-open lunar_lander_best.gif

# macOS
open lunar_lander_best.gif

# 或者用浏览器
firefox lunar_lander_best.gif
# 或
google-chrome lunar_lander_best.gif
```

## 📊 性能亮点

在`lunar_lander_best.gif`中，你可以看到：

✅ **稳定的着陆**：5个episodes全部成功着陆（奖励>200）
✅ **高分数**：平均285.01，最高301.69
✅ **软着陆**：飞船平稳降落到着陆垫
✅ **保持直立**：着陆时飞船保持正确角度
✅ **中心定位**：精准降落在两个旗帜之间

## 🔧 生成自己的视频

使用`record_video.py`脚本生成新的可视化：

```bash
# 基础用法
.venv/bin/python record_video.py \
    --model-path checkpoints/ppo_best.pt \
    --output my_demo.gif

# 自定义参数
.venv/bin/python record_video.py \
    --model-path checkpoints/ppo_best.pt \
    --env-id LunarLander-v3 \
    --num-episodes 3 \
    --output custom_demo.gif \
    --fps 30 \
    --resize 800

# 生成MP4（需要安装opencv-python）
pip install opencv-python
.venv/bin/python record_video.py \
    --model-path checkpoints/ppo_best.pt \
    --output demo.mp4
```

### 参数说明
- `--model-path`: 训练好的模型路径
- `--env-id`: 环境ID（默认LunarLander-v3）
- `--num-episodes`: 录制的episodes数量
- `--output`: 输出文件路径（.gif或.mp4）
- `--fps`: 帧率（默认30）
- `--resize`: 调整图像尺寸（默认512）
- `--device`: 设备（cpu/cuda/auto）

## 💡 提示

1. **GIF vs MP4**:
   - GIF: 兼容性好，无需额外软件，但文件较大
   - MP4: 文件小，质量高，需要视频播放器

2. **性能与质量**:
   - 降低fps可减小文件大小
   - 增大resize值可提高清晰度
   - 增加episodes会延长视频时长

3. **分享视频**:
   - GIF可直接在社交媒体分享
   - MP4适合上传到视频平台

## 📈 训练结果回顾

- **最佳平均奖励**: 273.33
- **成功率**: 84%
- **最高奖励**: 326.33
- **训练时长**: 53分钟（200万步）

模型已成功掌握LunarLander任务！🚀
