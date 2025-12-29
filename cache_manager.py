"""
缓存管理模块
处理MFCC缓存的保存和加载
"""

import os
import pickle


def get_cache_path(audio_path, feature_type, n_features, window_ms, hop_ms):
    """
    获取音频特征缓存文件路径
    
    参数:
        audio_path: 音频文件路径
        feature_type: 特征类型 ('mfcc' 或 'fbank')
        n_features: 特征维度（MFCC维度或Mel滤波器数量）
        window_ms: 窗口长度(毫秒)
        hop_ms: 跳跃长度(毫秒)
    
    返回:
        cache_path: 缓存文件路径
    """
    # 生成缓存标识（基于参数）
    cache_key = f"{feature_type}_{n_features}_{window_ms}_{hop_ms}"
    audio_dir = os.path.dirname(audio_path) or '.'
    audio_basename = os.path.basename(audio_path)
    audio_name = os.path.splitext(audio_basename)[0]
    cache_filename = f"{audio_name}_cache_{cache_key}.pkl"
    return os.path.join(audio_dir, cache_filename)


def load_source_feature_cache(cache_path, source_path):
    """
    加载源音频特征缓存（MFCC或Fbank）
    
    参数:
        cache_path: 缓存文件路径
        source_path: 源音频文件路径
    
    返回:
        如果缓存有效，返回cache_data字典
        否则返回None
    """
    if not os.path.exists(cache_path):
        return None
    
    # 检查源文件是否被修改
    source_mtime = os.path.getmtime(source_path)
    cache_mtime = os.path.getmtime(cache_path)
    
    if source_mtime > cache_mtime:
        print(f"  缓存已过期（源文件已修改）")
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # 向后兼容：如果是旧格式的MFCC缓存，转换为新格式
        if 'source_mfcc' in cache_data and 'source_features' not in cache_data:
            cache_data['source_features'] = cache_data['source_mfcc']
        
        print(f"  从缓存加载: {os.path.basename(cache_path)}")
        return cache_data
    except Exception as e:
        print(f"  缓存加载失败: {e}")
        return None


# 向后兼容的别名
load_source_mfcc_cache = load_source_feature_cache


def save_source_feature_cache(cache_path, source_features, source_y, source_sr, source_duration):
    """
    保存源音频特征缓存（MFCC或Fbank）
    
    参数:
        cache_path: 缓存文件路径
        source_features: 源音频完整特征（MFCC或Fbank）
        source_y: 源音频时间序列
        source_sr: 源音频采样率
        source_duration: 源音频时长
    """
    try:
        cache_data = {
            'source_features': source_features,
            'source_y': source_y,
            'source_sr': source_sr,
            'source_duration': source_duration
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  缓存已保存: {os.path.basename(cache_path)}")
    except Exception as e:
        print(f"  缓存保存失败: {e}")


# 向后兼容的别名
save_source_mfcc_cache = save_source_feature_cache
