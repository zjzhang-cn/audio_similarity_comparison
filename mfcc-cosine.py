import librosa
import numpy as np
from scipy.spatial.distance import cosine
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='音频MFCC特征提取和相似度计算')
parser.add_argument('-t', '--target', required=True, help='目标音频文件路径')
parser.add_argument('-s', '--source', required=True, help='查询音频文件路径')
args = parser.parse_args()

target_file = args.target
source_file = args.source

n_mfcc=40
hop_length=512

y, sr = librosa.load(target_file)
target_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

y, sr = librosa.load(source_file)
source_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

# 计算相似度（1 - 余弦距离）
# 将MFCC特征展平为一维向量
target_mfcc_flat = target_mfcc.flatten()
source_mfcc_flat = source_mfcc.flatten()

# 调整长度，使两个向量长度相同
min_length = min(len(target_mfcc_flat), len(source_mfcc_flat))
target_mfcc_flat = target_mfcc_flat[:min_length]
source_mfcc_flat = source_mfcc_flat[:min_length]

# 计算余弦相似度
cosine_distance = cosine(target_mfcc_flat, source_mfcc_flat)
similarity = 1 - cosine_distance

print(f"\n整体匹配：")
print(f"\t余弦距离: {cosine_distance:.4f}")
print(f"\t相似度 (1 - 余弦距离): {similarity:.4f}")

# 滑动窗口匹配
print(f"\n滑动窗口匹配：")
# 假设mfcc2是较短的音频（查询片段），在mfcc1中滑动查找最匹配的位置
# 使用帧数作为窗口单位
query_frames = source_mfcc.shape[1]  # 查询片段的帧数
target_frames = target_mfcc.shape[1]  # 目标音频的帧数

if query_frames > target_frames:
    print("警告：查询音频比目标音频长，交换角色进行匹配")
    target_mfcc, source_mfcc = source_mfcc, target_mfcc
    query_frames = source_mfcc.shape[1]
    target_frames = target_mfcc.shape[1]


print(f"\t查询片段长度: {query_frames} 帧")

print(f"\t目标音频长度: {target_frames} 帧")

best_similarity = -1
best_position = 0
similarities = []

# 对查询片段进行归一化和统计特征提取
query_normalized = (source_mfcc - np.mean(source_mfcc, axis=1, keepdims=True)) / (np.std(source_mfcc, axis=1, keepdims=True) + 1e-8)
query_flat = query_normalized.flatten()

# 提取查询片段的统计特征
query_stats = np.concatenate([
    np.mean(source_mfcc, axis=1),      # 均值
    np.std(source_mfcc, axis=1),       # 标准差
    np.max(source_mfcc, axis=1),       # 最大值
    np.min(source_mfcc, axis=1),       # 最小值
])

# 滑动窗口遍历
for i in range(target_frames - query_frames + 1):
    # 提取当前窗口的MFCC特征
    window = target_mfcc[:, i:i+query_frames]
    
    # 归一化窗口特征
    window_normalized = (window - np.mean(window, axis=1, keepdims=True)) / (np.std(window, axis=1, keepdims=True) + 1e-8)
    window_flat = window_normalized.flatten()
    
    # 提取窗口的统计特征
    window_stats = np.concatenate([
        np.mean(window, axis=1),      # 均值
        np.std(window, axis=1),       # 标准差
        np.max(window, axis=1),       # 最大值
        np.min(window, axis=1),       # 最小值
    ])
    
    # 计算余弦相似度（使用归一化特征）
    cos_dist_norm = cosine(window_flat, query_flat)
    sim_norm = 1 - cos_dist_norm
    
    # 计算统计特征的余弦相似度
    cos_dist_stats = cosine(window_stats, query_stats)
    sim_stats = 1 - cos_dist_stats
    
    # 综合相似度（加权平均）
    sim = 0.7 * sim_norm + 0.3 * sim_stats
    similarities.append(sim)
    
    # 更新最佳匹配
    if sim > best_similarity:
        best_similarity = sim
        best_position = i

# 计算最佳位置的时间（秒）
hop_length = 512
best_time = best_position * hop_length / sr
query_duration = query_frames * hop_length / sr

print(f"\t最佳匹配位置: 帧 {best_position} (时间: {best_time:.2f}秒)")
print(f"\t最佳匹配相似度: {best_similarity:.4f}")
print(f"\t匹配片段时间范围: {best_time:.2f}秒 - {best_time + query_duration:.2f}秒")

# 打印所有窗口位置的相似度列表
# print(f"\n所有窗口位置的余弦相似度列表:")
# print(f"总窗口数: {len(similarities)}")
# for idx, sim in enumerate(similarities):
#     time_pos = idx * hop_length / sr
#     print(f"位置 {idx} (时间: {time_pos:.2f}秒): {sim:.4f}")
