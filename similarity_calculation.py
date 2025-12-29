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
    
    # 计算余弦相似度 (1 - 余弦距离)
    similarity = 1 - cosine(vec1, vec2)
    return similarity


def compute_dtw_similarity(mfcc1, mfcc2):
    """
    使用DTW（动态时间规整）计算两个MFCC特征矩阵的相似度
    
    参数:
        mfcc1: 第一个MFCC特征矩阵，形状为 (n_mfcc, time_frames)
        mfcc2: 第二个MFCC特征矩阵，形状为 (n_mfcc, time_frames)
    
    返回:
        similarity: 相似度分数 (0-1之间，越高越相似)
        dtw_distance: 归一化后的DTW距离
        path: DTW对齐路径
    """
    # 使用librosa的DTW算法
    # 转置矩阵，使其形状为 (time_frames, n_mfcc)
    mfcc1_T = mfcc1.T
    mfcc2_T = mfcc2.T
    
    # 计算DTW距离和对齐路径
    D, wp = librosa.sequence.dtw(mfcc1_T, mfcc2_T, metric='euclidean')
    raw_dtw_distance = D[-1, -1]  # 原始累积距离
    
    # 归一化DTW距离：除以路径长度得到平均每步距离
    path_length = len(wp)
    normalized_dtw_distance = raw_dtw_distance / path_length
    
    # 转换为相似度分数
    # 使用指数衰减函数，针对MFCC的DTW距离范围调整：
    # 典型距离范围约为100-200，使用更大的衰减系数
    # - 距离<50 → 相似度>0.9 (非常相似)
    # - 距离~100 → 相似度~0.6 (中等相似)
    # - 距离>150 → 相似度<0.3 (不相似)
    similarity = np.exp(-normalized_dtw_distance / 50.0)
    
    return similarity, normalized_dtw_distance, wp


def normalize_dtw_similarities(similarities_dtw, dtw_distances):
    """
    将DTW距离归一化为相似度
    
    参数:
        similarities_dtw: 原始DTW相似度列表
        dtw_distances: DTW距离列表
    
    返回:
        归一化后的相似度列表
    """
    if len(dtw_distances) > 0:
        min_dist = np.min(dtw_distances)
        max_dist = np.max(dtw_distances)
        dist_range = max_dist - min_dist
        
        if dist_range > 0:
            # 重新计算DTW相似度：最小距离=1.0，最大距离=0.0
            similarities_dtw = [1 - (d - min_dist) / dist_range for d in dtw_distances]
    
    return similarities_dtw
