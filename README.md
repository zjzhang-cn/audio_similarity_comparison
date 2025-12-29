# Audio Similarity Comparison

基于 MFCC 特征和 DTW 对齐的音频相似度比较与匹配工具。

## 项目简介

本项目实现了一个强大的音频匹配系统，可以在长音频文件中查找并定位短音频片段。使用梅尔频率倒谱系数（MFCC）提取音频特征，并通过动态时间规整（DTW）算法进行精确匹配，能够处理时间变化和速度差异。

## 主要特性

✨ **13维 MFCC 特征提取**
- 每 100ms 计算一次 MFCC
- 50% 重叠，提供精细的时间分辨率
- 自适应采样率处理
- **自动移除静音部分**，提高匹配精度

🎯 **DTW 动态时间规整对齐**
- 处理音频速度变化
- 对时间扭曲具有鲁棒性
- 智能相似度计算（基于相对距离）

🔍 **滑动窗口搜索**
- 可调节的窗口跳跃比例
- 实时进度显示
- 多位置匹配检测

📊 **详细的匹配结果**
- 精确的时间位置（起始/结束时间）
- 相似度百分比
- DTW 距离统计
- 所有候选位置列表

## 安装

### 环境要求

- Python >= 3.13
- 推荐使用 `uv` 包管理器

### 安装步骤

```bash
# 克隆项目（如果适用）
cd audio-similarity-comparison

# 使用 uv 同步依赖
uv sync

# 或使用 pip 安装依赖
pip install -r requirements.txt
```

### 依赖包

- `librosa >= 0.10.0` - 音频处理和特征提取
- `numpy >= 1.24.0` - 数值计算
- `scipy >= 1.10.0` - 科学计算（余弦相似度）

## 使用方法

### 基本用法

1. 准备音频文件（或使用默认文件名）

2. 运行程序（使用默认参数）：

```bash
uv run python main.py
```

3. 使用命令行参数指定音频文件：

```bash
# 指定目标音频和源音频
uv run python main.py -t audio1.wav -s audio.wav

# 或使用完整参数名
uv run python main.py --target audio1.wav --source audio.wav
```

### 命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--target` | `-t` | audio1.wav | 要查找的目标音频片段 |
| `--source` | `-s` | audio.wav | 源音频文件（在其中搜索）|
| `--mfcc` | - | 13 | MFCC 特征维度数量 |
| `--threshold` | - | 0.7 | 相似度阈值（0-1）|
| `--hop-ratio` | - | 0.15 | 滑动窗口跳跃比例（0-1）|
| `--no-dtw` | - | False | 禁用DTW，使用余弦相似度 |
| `--no-trim-silence` | - | False | 禁用静音移除 |
| `--silence-threshold` | - | 30 | 静音阈值（dB），低于此值视为静音 |
| `--help` | `-h` | - | 显示帮助信息 |

### 使用示例

**示例 1：基本使用**
```bash
uv run python main.py -t my_clip.wav -s long_audio.wav
```

**示例 2：自定义MFCC维度和阈值**
```bash
uv run python main.py -t audio1.wav -s audio.wav --mfcc 20 --threshold 0.8
```

**示例 3：使用余弦相似度（更快）**
```bash
uv run python main.py -t audio1.wav -s audio.wav --no-dtw
```

**示例 4：精细搜索**
```bash
uv run python main.py -t audio1.wav -s audio.wav --hop-ratio 0.05 --threshold 0.6
```

**示例 5：快速搜索**
```bash
uv run python main.py -t audio1.wav -s audio.wav --hop-ratio 0.3 --mfcc 10
``示例 6：禁用静音移除**
```bash
uv run python main.py -t audio1.wav -s audio.wav --no-trim-silence
```

**示例 7：调整静音阈值（更严格）**
```bash
uv run python main.py -t audio1.wav -s audio.wav --silence-threshold 20
```

**`

**查看所有参数**
```bash
uv run python main.py --help
```

  移除静音: 是
  静音阈值: 30dB

正在加载音频文件...
已启用静音移除（阈值: 30dB）
处理目标音频: audio1.wav
  移除了 0.30秒 的静音
目标音频长度: 1.3
配置信息:
  目标音频: audio1.wav
  源音频: audio.wav
  MFCC维度: 13
  相似度阈值: 0.7
  跳跃比例: 0.15
  使用DTW: 是49 秒
  结束时间: 5.79 秒
  相似度: 1.0000 (100.00%)
  DTW距离: 112.69 (归一化后，越小越相似)

✓ audio1.wav 在 audio.wav 中存在
  位置: 4.49秒 - 5.79秒

【所有匹配位置】 (相似度 ≥ 70%):
  1. 2.54秒 - 3.84秒, 相似度: 0.7813 (78.13%)
  2. 2.73秒 - 4.03秒, 相似度: 0.7015 (70.15%)
  3. 4.49秒 - 5.79秒, 相似度: 1.0000 (100.00
【最佳匹配位置】
  算法: DTW（动态时间规整）对齐
  起始时间: 4.31 秒
  结束时间: 5.91 秒
  相似度: 1.0000 (100.00%)
  DTW距离: 140.94 (归一化后，越小越相似)

✓ audio1.wav 在 audio.wav 中存在
  位置: 4.31秒 - 5.91秒

【所有匹配位置】 (相似度 ≥ 70%):
  1. 1.20秒 - 2.79秒, 相似度: 0.8226 (82.26%)
  2. 2.40秒 - 3.99秒, 相似度: 0.7572 (75.72%)
  3. 4.31秒 - 5.91秒, 相似度: 1.0000 (100.00%)
  4. 10.30秒 - 11.90秒, 相似度: 0.7373 (73.73%)

【相似度统计】
  平均相似度: 0.3923
  最大相似度: 1.0000
  最小相似度: 0.0000

### 通过命令行配置

推荐使用命令行参数动态配置，无需修改代码：

```bash
uv run python main.py -t audio1.wav -s audio.wav --mfcc 20 --threshold 0.8
```

### 通过代码配置（高级）

如果需要在代码中调用，可以直接使用 `find_audio_in_audio()` 函
  平均DTW距离: 190.56
  最小DTW距离: 140.94
  最大DTW距离: 222.59
```

## 参数配置

在 `main()` 函数中可以调整以下参数：

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_mfcc` | 13 | MFCC 特征维度数量 |
| `threshold` | 0.7 | 相似度阈值（0-1），超过此值视为匹配 |
| `hop_ratio` | 0.15 | 滑动窗口跳跃比例，越小越精确但计算越慢 |
| `use_dtw` | True | 是否使用 DTW 对齐算法 |

### MFCC 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_ms` | 100 | MFCC 窗口长度（毫秒）|
| `hop_ms` | 50 | MFCC 跳跃长度（毫秒），50% 重叠 |

### 示例：自定义参数

```python
matches, best_match, positions, similarities, dtw_distances = find_audio_in_audio(
    target_path="my_target.wav",
    source_path="my_source.wav",
    n_mfcc=20,           # 使用20维MFCC
    threshold=0.8,       # 提高相似度阈值
    hop_ratio=0.1,       # 更精细的搜索
    use_dtw=True         # 使用DTW对齐
)
```静音移除

在比较音频前自动移除静音部分，提高匹配精度：

- **默认启用**：自动检测并移除开头和结尾的静音
- **静音阈值**：默认 30dB，低于此值的声音视为静音
- **仅处理目标音频**：源音频不移除静音，保持时间位置准确
- **可配置**：通过 `--silence-threshold` 调整灵敏度
- **可禁用**：使用 `--no-trim-silence` 参数

**静音阈值调整建议：**
- **30dB（默认）**：适合大多数场景
- **20dB（严格）**：移除更多低音量部分
- **40dB（宽松）**：只移除非常安静的部分

### 

## 技术细节

### MFCC（梅尔频率倒谱系数）

MFCC 是一种广泛用于语音和音频处理的特征表示方法：
移除目标音频的静音部分（可选）
3. 提取13维MFCC特征（100ms窗口，50%重叠）
4. 滑动窗口遍历源音频
5. 对每个窗口：
   a. 提取窗口的MFCC特征
   b. 使用DTW计算与目标的相似度
   c. 记录位置和相似度
6. 基于相对DTW距离重新计算相似度
7TW 算法通过找到最优时间对齐路径来比较两个时间序列：

1. **距离计算**：使用欧几里得距离计算帧间距离
2. **路径寻找**：动态规划找到最小累积距离路径
3. **相似度转换**：
   - 归一化：距离 ÷ 路径长度
   - 相对评分：最小距离 = 100%，最大距离 = 0%

### 算法流程

```
1. 加载目标音频和源音频
2. 提取13维MFCC特征（100ms窗口，50%重叠）
3. 滑动窗口遍历源音频
4. 对每个窗口：
   a. 提取窗口的MFCC特征
   b. 使用DTW计算与目标的相似度
   c. 记录位置和相似度
5. 基于相对DTW距离重新计算相似度
6. 输出最佳匹配和所有候选位置
```

## 性能优化建议

### 提高搜索速度

- 增大 `hop_ratio`（如 0.25 或 0.5）
- 减少 MFCC 维度（如 8 或 10）
- 使用余弦相似度代替 DTW（`use_dtw=False`）

### 提高匹配精度

- 减小 `hop_ratio`（如 0.05 或 0.1）
- 增加 MFCC 维度（如 20 或 26）
- 使用 DTW 对齐（`use_dtw=True`）
- 调整相似度阈值

## API 参考

### `extract_mfcc(audio_path, n_mfcc=13, window_ms=100, hop_ms=50)`

提取音频的 MFCC 特征。

**返回值：**
- `mfcc`: MFCC 特征矩阵 (n_mfcc, time_frames)
- `mfcc_mean`: MFCC 均值向量 (n_mfcc,)
- `y`: 音频时间序列
- `sr`: 采样率

### `compute_dtw_similarity(mfcc1, mfcc2)`

使用 DTW 计算两个 MFCC 特征的相似度。

**返回值：**
- `similarity`: 相似度分数 (0-1)
- `dtw_distance`: 归一化 DTW 距离
- `wp`: DTW 对齐路径

### `find_audio_in_audio(target_path, source_path, ...)`

在源音频中查找目标音频片段。

- 调整静音阈值（`--silence-threshold`）
- 尝试禁用静音移除（`--no-trim-silence`）
**返回值：**
- `matches`: 所有匹配位置列表
- `best_match`: 最佳匹配信息
- `positions`: 所有搜索位置
- `similarities`: 所有相似度分数
- `dtw_distances`: 所有 DTW 距离

## 常见问题

### Q: 为什么找不到明显存在的音频？


### Q: 什么时候应该禁用静音移除？

**A:**
- 当目标音频本身包含重要的静音段落时
- 当需要保持精确的时间长度匹配时
- 当音频质量很好且没有明显静音时
**A:** 尝试以下方法：
- 降低相似度阈值（如 0.5 或 0.6）
- 减小 `hop_ratio` 进行更密集搜索
- 检查音频文件质量和采样率是否一致

### Q: 搜索速度太慢怎么办？

**A:** 
- 增大 `hop_ratio` 到 0.3-0.5
- 使用余弦相似度（`use_dtw=False`）
- 减少 MFCC 维度

### Q: DTW 和余弦相似度有什么区别？

**A:**
- **DTW**: 适合处理时间变化，更鲁棒，但计算较慢
- **余弦相似度**: 计算快速，适合严格对齐的音频

## 支持的音频格式
- ✨ 命令行参数支持
- ✨ **自动移除静音功能**
- ✨ 可配置的静音阈值

通过 `librosa` 支持以下格式：
- WAV
- MP3
- FLAC
- OGG
- M4A

## 许可证

本项目仅供学习和研究使用。

## 作者

Audio Similarity Comparison Team

## 更新日志

### v0.1.0 (2025-12-29)
- ✨ 初始版本
- ✨ 实现13维 MFCC 特征提取
- ✨ 实现 DTW 动态时间规整对齐
- ✨ 实现滑动窗口搜索
- ✨ 支持余弦相似度和 DTW 两种匹配算法
- ✨ 详细的匹配结果输出
