# 🎵 Audio Similarity Comparison

基于 MFCC/Fbank 和 DTW 的音频相似度比较与匹配工具

## 📖 项目简介

一个高性能的音频片段匹配系统，可以在长音频文件中精确查找并定位短音频片段。采用模块化架构，集成了多种先进算法：

- *✨ 主要特性

### 🎯 核心功能
- **精确匹配**：在长音频中查找目标音频片段，精确定位时间位置
- **双算法验证**：DTW + 余弦相似度同时计算，降低误报率
- **多种特征提取**：支持 MFCC 和 Fbank（Filter Bank）两种特征
- **智能预处理**：自动静音移除、可选音频降噪
- **高性能缓存**：特征缓存系统，22x搜索加速（0.22s → 0.01s）

### 🔬 技术特性
- **MFCC特征**：13维（可配置），25ms帧长，10ms步长，高时间分辨率
- **Fbank特征**：40个滤波器（可配置），保留更多频谱信息
- **相对DTW归一化**：最小距离=100%，最大距离=0%
- **滑动窗口搜索**：可配置跳跃比例，精度与速度平衡
- **完全相同检测**：相似度≥99%时显示"完全相同"标记

### 📦 模块化架构
- `audio_processing.py` - 音频加载、降噪、静音移除
- `mfcc_extraction.py` - MFCC/Fbank特征提取
- `similarity_calculation.py` - DTW与余弦相似度计算
- `cache_manager.py` - 特征缓存管理
- `audio_matcher.py` - 核心匹配算法
- `main.py` - 命令行入口
📊 **详细的匹配结果**
- 精确的时间位置（起始/结束时间）
- 相似度百分比
- DTW 距离统计
- 所有候选位置列表

## 安装

###🚀 快速开始

### 环境要求
- Python >= 3.13
- 推荐使用 `uv` 包管理器

### 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install librosa numpy scipy noisereduce
```

### 核心依赖
| 包名 | 版本 | 用途 |
|------|------|------|
| librosa | >= 0.10.0 | 音频处理、MFCC、DTW |
| numpy | >= 1.24.0 | 数值计算 |
| scipy | >= 1.10.0 | 余弦相似度 |
| noisereduce | 3.0.3 | 音频降噪（可选）|

### 基本使用

```📋 命令行参数

### 完整参数列表

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--target` | `-t` | audio1.wav | 要查找的目标音频片段 |
| `--source` | `-s` | audio.wav | 源音频文件（在其中搜索）|
| `--feature-type` | - | mfcc | 特征类型：mfcc 或 fbank |
| `--mfcc` | - | 13 | MFCC 特征维度数量 |
| `--fbank` | - | 40 | Fbank 滤波器数量 |
| `--threshold` | - | 0.7 | 相似度阈值（0-1），超过视为匹配 |
| `--hop-ratio` | - | 0.15 | 滑动窗口跳跃比例（0-1）|
| `--no-trim-silence` | - | False | 禁用静音移除 |
| `--silence-threshold` | - | 30 | 静音阈值（dB）|
| `--reduce-noise` | - | False | 启用降噪功能 |
| `--help` | `-h` | - | 显示帮助信息 |

### 使用示例

```bash
# 基本使用（MFCC特征）
uv run python main.py -t audio1.wav -s audio.wav

# 使用Fbank特征
uv run python main.py -t audio1.wav -s audio.wav --feature-type fbank

# 自定义Fbank滤波器数量
uv run python main.py -t audio1.wav -s audio.wav --feature-type fbank --fbank 80

# 高精度匹配（阈值90%）
uv run python main.py -t audio1.wav -s audio.wav --threshold 0.9

# 启用降噪
uv run python main.py -t audio2.wav -s audio.wav --reduce-noise

# 自定义MFCC维度
uv run python main.py -t audio1.wav -s audio.wav --mfcc 20

# 精细搜索（小跳跃比例）
uv run python main.py -t audio1.wav -s audio.wav --hop-ratio 0.05

# 快速搜索（大跳跃比例）
uv run python main.py -t audio1.wav -s audio.wav --hop-ratio 0.3

# 禁用静音移除
uv run python main.py -t audio1.wav -s audio.wav --no-trim-silence

# 调整静音阈值
uv run python main.py -t audio1.wav -s audio.wav --silence-threshold 20
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
## 📊 输出示例

```bash
$ uv run python main.py -t audio1.wav -s audio.wav --threshold 0.9

配置信息:
  目标音频: audio1.wav
  源音频: audio.wav
  MFCC维度: 13
  相似度阈值: 0.9
  跳跃比例: 0.15
  计算方法: DTW + 余弦相似度（同时计算）
  降噪: 否
  移除静音: 是
  静音阈值: 30dB

正在加载音频文件...
已启用静音移除（阈值: 30dB）
处理音频: audio1.wav
  移除了 0.30秒 的静音
  从缓存加载: audio_mfcc_cache_13_25_10.pkl
音频加载完成，耗时: 1.39秒
目标音频长度: 1.30秒
源音频长度: 12.30秒

正在搜索匹配位置...
同时使用: DTW对齐 + 余弦相似度
搜索完成，耗时: 0.01秒

============================================================
搜索结果:
============================================================

【DTW匹配位置】 (相似度 ≥ 90%):
  1. 4.49秒 - 5.79秒, 相似度: 1.0000 (100.00%) 完全相同

【余弦相似度匹配位置】 (相似度 ≥ 90%):
  24. 4.49秒 - 5.79秒, 相似度: 0.9924 (99.24%) 完全相同

【相似度统计】
DTW算法:
  平均相似度: 0.2356
  最大相似度: 1.0000
  最小相似度: 0.0000

余弦相似度:
  平均相似度: 0.9664
  最大相似度: 0.9924
  最小相似度: 0.9544

【DTW距离统计】
  平均DTW距离: 399.07
  最小DTW距离: 202.60
  最大DTW距离: 459.64

【最佳匹配位置 - DTW算法】
  起始时间: 4.49 秒
  结束时间: 5.79 秒
  相似度: 1.0000 (100.00%)
  DTW距离: 202.60 (归一化后，越小越相似)

【最佳匹配位置 - 余弦相似度】
  起始时间: 4.49 秒
  结束时间: 5.79 秒
  相似度: 0.9924 (99.24%)

✓ audio1.wav 在 audio.wav 中存在
  DTW推荐位置: 4.49秒 - 5.79秒
  余弦相似度推荐位置: 4.49秒 - 5.79秒

【两种算法同时匹配的位置】 (DTW ≥ 90% 且 余弦 ≥ 90%):
  1. 4.49秒 - 5.79秒
     DTW相似度: 1.0000 (100.00%), 余弦相似度: 0.9924 (99.24%)
```

## 🔧 技术细节

### 音频特征提取

#### MFCC (Mel-Frequency Cepstral Coefficients)
MFCC是最常用的音频特征，通过模拟人耳对声音频率的非线性感知特性来提取特征。

**参数配置：**
- **维度**：13维（默认，可通过 --mfcc 调整）
- **帧长**：25ms
- **步长**：10ms（帧间重叠15ms）
- **特性**：对音色敏感，压缩频谱信息

**适用场景：**
- 语音识别
- 说话人识别
- 音乐风格分类
- 一般音频匹配

#### Fbank (Filter Bank Features)
Fbank是MFCC的前置步骤，保留了更多原始频谱信息（MFCC在Fbank基础上进行DCT变换）。

**参数配置：**
- **滤波器数量**：40个（默认，可通过 --fbank 调整）
- **帧长**：25ms
- **步长**：10ms
- **特性**：保留更多频谱细节，维度更高

**适用场景：**
- 需要更多频谱细节的场景
- 深度学习音频任务
- 环境声识别
- 音乐信息检索

#### MFCC vs Fbank 对比

| 特性 | MFCC | Fbank |
|------|------|-------|
| 维度 | 较低（默认13维）| 较高（默认40维）|
| 频谱信息 | 压缩（DCT变换）| 完整（无DCT）|
| 计算速度 | 快 | 稍慢 |
| 内存占用 | 小 | 稍大 |
| 匹配精度 | 适中 | 更高细节 |
| 推荐场景 | 语音、快速匹配 | 音乐、精确匹配 |

**选择建议：**
- **MFCC**：计算资源有限、需要快速响应、语音类音频
- **Fbank**：追求更高精度、音乐类音频、需要保留频谱细节

### DTW算法
1. **距离度量**：欧几里得距离
2. **路径寻找**：动态规划最优路径
3. **归一化**：距离/路径长度
4. **相对评分**：`1 - (d - min) / (max - min)`

### 余弦相似度
- 特征向量展平后计算
- `相似度 = 1 - 余弦距离`
- 快速计算，适合粗筛选

### 缓存机制
- **缓存内容**：源音频特征（MFCC或Fbank）+ 原始数据
- **缓存格式**：`{audio_name}_cache_{feature_type}_{n_features}_{window_ms}_{hop_ms}.pkl`
- **失效机制**：文件修改时间检测
- **性能提升**：22倍加速（0.22s → 0.01s）
- **智能识别**：自动向后兼容旧MFCC缓存格式

## 🎓 算法流程

```
1. 音频预处理
   ├── 加载音频文件
   ├── [可选] 降噪处理
   └── [可选] 静音移除（仅目标音频）

2. MFCC特征提取
   ├── 目标音频：直接计算
   └── 源音频：尝试加载缓存 → 无缓存则计算并保存

3. 滑动窗口搜索
   ├── 按hop_ratio移动窗口
   ├── 提取窗口MFCC（数组切片，无需重算）
   ├── 同时计算DTW相似度和余弦相似度
   └── 记录位置和相似度

4. 结果处理
   ├── DTW相似度归一化（相对距离）
   ├── 生成匹配列表（分别 + 同时）
   ├── 找出最佳匹配位置
   └── 输出详细结果
```

## ⚙️ 性能优化建议

### 提高搜索速度
| 方法 | 参数 | 效果 | 副作用 |
|------|------|------|--------|
| 增大跳跃比例 | `--hop-ratio 0.3` | 3-5倍加速 | 可能遗漏部分匹配 |
| 减少MFCC维度 | `--mfcc 8` | 1.5-2倍加速 | 特征表达能力下降 |
| 使用缓存 | 自动 | 22倍加速 | 需要磁盘空间 |

### 提高匹配精度
| 方法 | 参数 | 效果 | 副作用 |
|------|------|------|--------|
| 减小跳跃比例 | `--hop-ratio 0.05` | 更精细定位 | 计算时间增加 |
| 增加MFCC维度 | `--mfcc 20` | 特征更丰富 | 计算时间增加 |
| 启用降噪 | `--reduce-noise` | 提高干净度 | 处理时间增加 |
| 双算法验证 | 默认启用 | 减少误报 | 需同时满足两个阈值 |

## 📦 模块化架构

### 模块说明

| 模块 | 职责 | 主要函数 |
|------|------|----------|
| `audio_processing.py` | 音频加载与预处理 | `load_and_preprocess_audio()`, `reduce_noise()`, `trim_silence()` |
| `mfcc_extraction.py` | MFCC特征提取 | `extract_audio_mfcc()` |
| `similarity_calculation.py` | 相似度计算 | `compute_dtw_similarity()`, `compute_similarity()`, `normalize_dtw_similarities()` |
| `cache_manager.py` | MFCC缓存管理 | `load_source_mfcc_cache()`, `save_source_mfcc_cache()` |
| `audio_matcher.py` | 核心匹配算法 | `find_audio_in_audio()`, `compute_window_similarities()`, `generate_matches()` |
| `main.py` | 命令行入口 | `main()` |

### 代码调用示例

```python
from audio_matcher import find_audio_in_audio

# 调用核心匹配函数
matches_dtw, matches_cosine, matches_both, \
best_match_dtw, best_match_cosine, \
positions, similarities_dtw, similarities_cosine, dtw_distances = \
    find_audio_in_audio(
        target_path="audio1.wav",
        source_path="audio.wav",
        n_mfcc=13,
        threshold=0.9,
        hop_ratio=0.15,
        trim_silence_enabled=True,
        silence_threshold=30,
        reduce_noise_enabled=False
    )

# 打印最佳匹配
print(f"DTW最佳匹配: {best_match_dtw['start_time']:.2f}s - {best_match_dtw['end_time']:.2f}s")
print(f"DTW相似度: {best_match_dtw['similarity']*100:.2f}%")
```

## ❓ 常见问题

### Q1: 找不到明显存在的音频？
**A:** 尝试以下方法：
- 降低阈值：`--threshold 0.6`
- 减小跳跃比例：`--hop-ratio 0.05`
- 检查采样率是否一致
- 启用降噪：`--reduce-noise`

### Q2: 搜索速度太慢？
**A:** 优化方法：
- 增大跳跃比例：`--hop-ratio 0.3`
- 减少MFCC维度：`--mfcc 10`
- 确保缓存生效（第二次搜索会快很多）

### Q3: 什么时候应该禁用静音移除？
**A:** 以下场景：
- 目标音频包含重要静音段落
- 需要精确时间长度匹配
- 音频质量很好无明显静音

### Q4: DTW和余弦相似度选哪个？
**A:** 区别：
- **DTW**：对时间变化鲁棒，精度高，速度慢
- **余弦相似度**：速度快，适合严格对齐的音频
- **双算法**：同时使用（默认），准确率最高

### Q5: 降噪功能什么时候用？
**A:** 适用场景：
- 音频有明显背景噪声
- 录音质量较差
- 想提高静音检测准确性
- 注意：会增加处理时间

## 🎵 支持的音频格式

通过 `librosa` 支持以下格式：
- ✅ WAV
- ✅ MP3
- ✅ FLAC
- ✅ OGG
- ✅ M4A

## 📝 更新日志

### v0.2.0 (2025-12-29)
- ✨ 模块化架构重构
- ✨ 添加MFCC缓存系统（22x加速）
- ✨ 双算法验证机制（DTW + 余弦）
- ✨ 完全相同检测（≥99%）
- ✨ 性能计时显示
- ✨ 可选降噪功能

### v0.1.0 (2025-12-28)
- ✨ 初始版本
- ✨ 13维MFCC特征提取
- ✨ DTW动态时间规整
- ✨ 滑动窗口搜索
- ✨ 自动静音移除
- ✨ 命令行参数支持

## 📄 许可证

本项目仅供学习和研究使用。

---

**Audio Similarity Comparison** - 高性能音频匹配工具 🎵