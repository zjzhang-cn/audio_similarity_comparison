"""
音频处理模块
包含音频加载、降噪、静音移除等功能
"""

import librosa
import os

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("警告: noisereduce库未安装，降噪功能不可用。可使用 'pip install noisereduce' 安装")


def reduce_noise(y, sr, stationary=True):
    """
    对音频进行降噪处理
    
    参数:
        y: 音频时间序列
        sr: 采样率
        stationary: 是否使用平稳降噪（适用于持续性噪声）
    
    返回:
        y_denoised: 降噪后的音频
    """
    if not NOISEREDUCE_AVAILABLE:
        print("  降噪库不可用，跳过降噪")
        return y
    
    try:
        # 使用noisereduce进行降噪
        y_denoised = nr.reduce_noise(y=y, sr=sr, stationary=stationary)
        return y_denoised
    except Exception as e:
        print(f"  降噪失败: {e}，使用原始音频")
        return y


def trim_silence(y, sr, top_db=30, frame_length=2048, hop_length=512):
    """
    移除音频的静音部分
    
    参数:
        y: 音频时间序列
        sr: 采样率
        top_db: 静音阈值（dB），低于这个值的声音视为静音
        frame_length: 帧长度
        hop_length: 跳跃长度
    
    返回:
        y_trimmed: 移除静音后的音频
        trimmed_duration: 被移除的静音时长（秒）
    """
    # 使用 librosa.effects.trim 移除开头和结尾的静音
    y_trimmed, index = librosa.effects.trim(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    
    # 计算被移除的时长
    original_duration = len(y) / sr
    trimmed_duration = len(y_trimmed) / sr
    removed_duration = original_duration - trimmed_duration
    
    return y_trimmed, removed_duration


def load_and_preprocess_audio(audio_path, reduce_noise_enabled=False, trim_silence_enabled=False, silence_threshold=30, target_sr=None):
    """
    加载并预处理音频
    
    参数:
        audio_path: 音频文件路径
        reduce_noise_enabled: 是否启用降噪
        trim_silence_enabled: 是否移除静音
        silence_threshold: 静音阈值 (dB)
        target_sr: 目标采样率 (Hz)，如果指定则重采样到此采样率
    
    返回:
        y: 音频时间序列
        sr: 采样率
        removed_silence: 移除的静音时长
    """
    print(f"处理音频: {os.path.basename(audio_path)}")
    
    # 加载音频（如果指定了目标采样率则直接重采样）
    y, sr = librosa.load(audio_path, sr=target_sr)
    
    # 降噪处理
    if reduce_noise_enabled:
        y = reduce_noise(y, sr)
        print(f"  已完成降噪")
    
    # 移除静音
    removed_silence = 0.0
    if trim_silence_enabled:
        y, removed_silence = trim_silence(y, sr, top_db=silence_threshold)
        if removed_silence > 0.01:
            print(f"  移除了 {removed_silence:.2f}秒 的静音")
    
    return y, sr, removed_silence
