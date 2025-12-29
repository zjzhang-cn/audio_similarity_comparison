"""
相似度计算模块
包含余弦相似度和DTW相似度计算
"""

import numpy as np
import librosa
from scipy.spatial.distance import cosine


def compute_similarity(mfcc1, mfcc2):
    """
    计算两个MFCC特征矩阵的余弦相似度
    
    参数:
        mfcc1: 第一个MFCC特征矩阵
        mfcc2: 第二个MFCC特征矩阵
    
    返回:
        similarity: 相似度分数 (0-1之间，越高越相似)
    """
    # 使用余弦相似度
    # 将特征矩阵展平
    vec1 = mfcc1.flatten()
    vec2 = mfcc2.flatten()
    
    # 检查向量长度，如果不一致则强制调整为相同长度
    min_len = min(len(vec1), len(vec2))
    vec1 = vec1[:min_len]
    vec2 = vec2[:min_len]
    
    # 计算余弦相似度 (1 - 余弦距离)
    # 添加L2归一化以提高区分度
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
    similarity = np.dot(vec1_norm, vec2_norm)
    
    return max(0.0, min(1.0, similarity))  # 确保在[0,1]范围内


def compute_dtw_similarity(mfcc1, mfcc2):
    """
    使用DTW（动态时间规整）计算两个MFCC特征矩阵的相似度
    
    参数:
        mfcc1: 第一个MFCC特征矩阵，形状为 (n_mfcc, time_frames)
        mfcc2: 第二个MFCC特征矩阵，形状为 (n_mfcc, time_frames)
    
    返回:
        similarity: 相似度分数 (0-1之间，越高越相似)
        dtw_distance: 标准化DTW距离（无量纲，可跨配置比较）
                     = (原始累积距离 / 路径长度) / sqrt(特征维度)
                     表示每帧的平均维度归一化距离
        path: DTW对齐路径
    """
    # 使用librosa的DTW算法
    # 转置矩阵，使其形状为 (time_frames, n_mfcc)
    mfcc1_T = mfcc1.T
    mfcc2_T = mfcc2.T
    
    # 计算DTW距离和对齐路径
    D, wp = librosa.sequence.dtw(mfcc1_T, mfcc2_T, metric='euclidean')
    raw_dtw_distance = D[-1, -1]  # 原始累积距离
    
    # 多级归一化，使DTW距离标准化且可跨配置比较：
    # 1. 除以路径长度：消除音频长度影响
    path_length = len(wp)
    path_normalized_distance = raw_dtw_distance / path_length
    
    # 2. 除以特征维度的平方根：消除维度影响
    #    原理：欧氏距离随维度增长，sqrt(n_features)是理论缩放因子
    n_features = mfcc1.shape[0]
    standardized_dtw_distance = path_normalized_distance / np.sqrt(n_features)
    
    # 转换为相似度分数
    # 使用指数衰减函数，标准化后的DTW距离典型范围为100-200
    # 为了提高区分度，使用更敏感的衰减系数：
    # - 距离<140 → 相似度>0.9 (优秀匹配，真实片段)
    # - 距离~150 → 相似度~0.77 (良好匹配)
    # - 距离~160 → 相似度~0.58 (中等匹配)
    # - 距离>170 → 相似度<0.40 (较差匹配，可能非片段)
    # 衰减系数设为30.0，使真实片段与非片段有明显差异
    similarity = np.exp(-(standardized_dtw_distance - 130.0) / 30.0)
    similarity = max(0.0, min(1.0, similarity))  # 限制在[0,1]范围
    
    return similarity, standardized_dtw_distance, wp


def normalize_dtw_similarities(similarities_dtw, dtw_distances):
    """
    保持标准化DTW相似度不变（已通过指数衰减函数转换）
    
    注意：不再进行相对归一化（最小=1.0，最大=0.0），
    而是保持绝对相似度，以便区分真实匹配和非匹配片段。
    
    参数:
        similarities_dtw: 原始DTW相似度列表（已通过exp衰减转换）
        dtw_distances: 标准化的DTW距离列表（仅用于显示）
    
    返回:
        原始相似度列表（不做修改）
    """
    # 直接返回原始相似度，不进行相对归一化
    # 这样可以保持真实匹配和非匹配片段之间的绝对差异
    return similarities_dtw
