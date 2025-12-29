"""
音频匹配模块
核心匹配算法和结果处理
"""

import time
import numpy as np
import librosa

from audio_processing import load_and_preprocess_audio
from mfcc_extraction import extract_audio_mfcc
from similarity_calculation import compute_dtw_similarity, compute_similarity, normalize_dtw_similarities
from cache_manager import get_cache_path, load_source_mfcc_cache, save_source_mfcc_cache


def compute_window_similarities(target_mfcc, source_mfcc, target_y, source_y, source_sr, hop_ratio):
    """
    在滑动窗口中计算DTW和余弦相似度
    
    参数:
        target_mfcc: 目标音频MFCC特征
        source_mfcc: 源音频MFCC特征
        target_y: 目标音频时间序列
        source_y: 源音频时间序列
        source_sr: 源音频采样率
        hop_ratio: 滑动窗口跳跃比例
    
    返回:
        positions: 位置列表
        similarities_dtw: DTW相似度列表
        similarities_cosine: 余弦相似度列表
        dtw_distances: DTW距离列表
    """
    target_samples = len(target_y)
    hop_samples = int(target_samples * hop_ratio)
    target_frames = target_mfcc.shape[1]
    
    # 计算hop_length（从样本数和帧数的关系中推导）
    hop_length = target_samples // target_frames
    
    positions = []
    similarities_dtw = []
    similarities_cosine = []
    dtw_distances = []
    
    print(f"正在搜索匹配位置...")
    print(f"同时使用: DTW对齐 + 余弦相似度")
    search_start_time = time.time()
    
    # 滑动窗口搜索
    for start_sample in range(0, len(source_y) - target_samples + 1, hop_samples):
        # 计算MFCC帧索引
        start_frame = int(start_sample / hop_length)
        end_frame = start_frame + target_frames
        
        # 检查是否超出范围
        if end_frame > source_mfcc.shape[1]:
            break
        
        # 切片获取窗口MFCC
        window_mfcc = source_mfcc[:, start_frame:end_frame]
        
        # 1. DTW相似度
        similarity_dtw, dtw_dist, _ = compute_dtw_similarity(target_mfcc, window_mfcc)
        dtw_distances.append(dtw_dist)
        
        # 2. 余弦相似度
        min_frames = min(target_mfcc.shape[1], window_mfcc.shape[1])
        target_mfcc_trimmed = target_mfcc[:, :min_frames]
        window_mfcc_trimmed = window_mfcc[:, :min_frames]
        similarity_cosine = compute_similarity(target_mfcc_trimmed, window_mfcc_trimmed)
        
        # 记录时间位置
        time_position = start_sample / source_sr
        positions.append(time_position)
        similarities_dtw.append(similarity_dtw)
        similarities_cosine.append(similarity_cosine)
    
    search_end_time = time.time()
    search_elapsed = search_end_time - search_start_time
    print(f"搜索完成，耗时: {search_elapsed:.2f}秒")
    
    return positions, similarities_dtw, similarities_cosine, dtw_distances


def generate_matches(positions, similarities_dtw, similarities_cosine, target_duration, threshold):
    """
    生成匹配位置列表
    
    参数:
        positions: 位置列表
        similarities_dtw: DTW相似度列表
        similarities_cosine: 余弦相似度列表
        target_duration: 目标音频时长
        threshold: 相似度阈值
    
    返回:
        matches_dtw: DTW匹配列表
        matches_cosine: 余弦相似度匹配列表
        matches_both: 同时匹配列表
    """
    matches_dtw = []
    matches_cosine = []
    matches_both = []
    
    # 生成DTW匹配列表
    for i, similarity in enumerate(similarities_dtw):
        if similarity >= threshold:
            matches_dtw.append({
                'start_time': positions[i],
                'end_time': positions[i] + target_duration,
                'similarity': similarity
            })
    
    # 生成余弦相似度匹配列表
    for i, similarity in enumerate(similarities_cosine):
        if similarity >= threshold:
            matches_cosine.append({
                'start_time': positions[i],
                'end_time': positions[i] + target_duration,
                'similarity': similarity
            })
    
    # 生成同时超过阈值的匹配列表
    for i in range(len(similarities_dtw)):
        if similarities_dtw[i] >= threshold and similarities_cosine[i] >= threshold:
            matches_both.append({
                'start_time': positions[i],
                'end_time': positions[i] + target_duration,
                'similarity_dtw': similarities_dtw[i],
                'similarity_cosine': similarities_cosine[i]
            })
    
    return matches_dtw, matches_cosine, matches_both


def find_best_matches(positions, similarities_dtw, similarities_cosine, dtw_distances, target_duration):
    """
    找出最佳匹配位置
    
    参数:
        positions: 位置列表
        similarities_dtw: DTW相似度列表
        similarities_cosine: 余弦相似度列表
        dtw_distances: DTW距离列表
        target_duration: 目标音频时长
    
    返回:
        best_match_dtw: DTW最佳匹配
        best_match_cosine: 余弦相似度最佳匹配
    """
    # 找出DTW最高相似度的位置
    max_similarity_idx_dtw = np.argmax(similarities_dtw)
    best_match_dtw = {
        'start_time': positions[max_similarity_idx_dtw],
        'end_time': positions[max_similarity_idx_dtw] + target_duration,
        'similarity': similarities_dtw[max_similarity_idx_dtw],
        'dtw_distance': dtw_distances[max_similarity_idx_dtw]
    }
    
    # 找出余弦相似度最高的位置
    max_similarity_idx_cosine = np.argmax(similarities_cosine)
    best_match_cosine = {
        'start_time': positions[max_similarity_idx_cosine],
        'end_time': positions[max_similarity_idx_cosine] + target_duration,
        'similarity': similarities_cosine[max_similarity_idx_cosine]
    }
    
    return best_match_dtw, best_match_cosine


def find_audio_in_audio(target_path, source_path, n_mfcc=13, threshold=0.7, hop_ratio=0.5, trim_silence_enabled=True, silence_threshold=30, reduce_noise_enabled=False):
    """
    在源音频中查找目标音频片段（主函数）
    
    参数:
        target_path: 目标音频文件路径
        source_path: 源音频文件路径
        n_mfcc: MFCC维度
        threshold: 相似度阈值
        hop_ratio: 滑动窗口跳跃比例
        trim_silence_enabled: 是否移除静音
        silence_threshold: 静音阈值
        reduce_noise_enabled: 是否启用降噪
    
    返回:
        匹配结果、相似度等信息
    """
    print(f"正在加载音频文件...")
    load_start_time = time.time()
    
    if trim_silence_enabled:
        print(f"已启用静音移除（阈值: {silence_threshold}dB）")
    
    # MFCC参数
    window_ms = 25
    hop_ms = 10
    
    # 1. 加载和预处理目标音频
    target_y, target_sr, _ = load_and_preprocess_audio(
        target_path, 
        reduce_noise_enabled=reduce_noise_enabled,
        trim_silence_enabled=trim_silence_enabled,
        silence_threshold=silence_threshold
    )
    
    # 2. 提取目标音频MFCC
    target_mfcc, n_fft, hop_length = extract_audio_mfcc(target_y, target_sr, n_mfcc, window_ms, hop_ms)
    target_duration = len(target_y) / target_sr
    
    # 3. 加载源音频（尝试从缓存加载）
    cache_path = get_cache_path(source_path, n_mfcc, window_ms, hop_ms)
    cache_data = load_source_mfcc_cache(cache_path, source_path)
    
    if cache_data is not None:
        source_mfcc = cache_data['source_mfcc']
        source_y = cache_data['source_y']
        source_sr = cache_data['source_sr']
        source_duration = cache_data['source_duration']
    else:
        # 加载和预处理源音频
        source_y, source_sr, _ = load_and_preprocess_audio(
            source_path,
            reduce_noise_enabled=reduce_noise_enabled,
            trim_silence_enabled=False,  # 源音频不移除静音
            silence_threshold=silence_threshold
        )
        source_duration = len(source_y) / source_sr
        
        # 提取源音频MFCC
        source_mfcc, _, _ = extract_audio_mfcc(source_y, source_sr, n_mfcc, window_ms, hop_ms)
        
        # 保存到缓存
        save_source_mfcc_cache(cache_path, source_mfcc, source_y, source_sr, source_duration)
    
    load_elapsed = time.time() - load_start_time
    print(f"音频加载完成，耗时: {load_elapsed:.2f}秒")
    print(f"目标音频长度: {target_duration:.2f}秒")
    print(f"源音频长度: {source_duration:.2f}秒")
    
    # 4. 确保采样率一致
    if target_sr != source_sr:
        print(f"警告: 采样率不一致，重新采样到 {source_sr} Hz")
        target_y = librosa.resample(target_y, orig_sr=target_sr, target_sr=source_sr)
        target_sr = source_sr
        target_mfcc, _, _ = extract_audio_mfcc(target_y, target_sr, n_mfcc, window_ms, hop_ms)
    
    # 5. 计算相似度
    print()
    positions, similarities_dtw, similarities_cosine, dtw_distances = compute_window_similarities(
        target_mfcc, source_mfcc, target_y, source_y, source_sr, hop_ratio
    )
    
    # 6. 归一化DTW相似度
    similarities_dtw = normalize_dtw_similarities(similarities_dtw, dtw_distances)
    
    # 7. 生成匹配列表
    matches_dtw, matches_cosine, matches_both = generate_matches(
        positions, similarities_dtw, similarities_cosine, target_duration, threshold
    )
    
    # 8. 找出最佳匹配
    best_match_dtw, best_match_cosine = find_best_matches(
        positions, similarities_dtw, similarities_cosine, dtw_distances, target_duration
    )
    
    return matches_dtw, matches_cosine, matches_both, best_match_dtw, best_match_cosine, positions, similarities_dtw, similarities_cosine, dtw_distances
