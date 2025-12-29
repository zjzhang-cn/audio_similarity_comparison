"""
MFCC特征提取模块
"""

import librosa


def extract_audio_mfcc(y, sr, n_mfcc=13, window_ms=25, hop_ms=10):
    """
    提取音频的MFCC特征
    
    参数:
        y: 音频时间序列
        sr: 采样率
        n_mfcc: MFCC维度
        window_ms: 窗口长度(毫秒)
        hop_ms: 跳跃长度(毫秒)
    
    返回:
        mfcc: MFCC特征矩阵
        n_fft: FFT窗口长度
        hop_length: 跳跃长度
    """
    n_fft = int(sr * window_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc, n_fft, hop_length
