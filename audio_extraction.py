"""
音频特征提取模块
支持MFCC和Fbank特征提取
"""

import librosa
import numpy as np


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


def extract_audio_fbank(y, sr, n_mels=40, window_ms=25, hop_ms=10, use_log=True):
    """
    提取音频的Fbank（Filter Bank）特征
    
    参数:
        y: 音频时间序列
        sr: 采样率
        n_mels: Mel滤波器组数量
        window_ms: 窗口长度(毫秒)
        hop_ms: 跳跃长度(毫秒)
        use_log: 是否使用对数能量（True=log-Fbank, False=Fbank）
    
    返回:
        fbank: Fbank特征矩阵 (n_mels, time_frames)
        n_fft: FFT窗口长度
        hop_length: 跳跃长度
    """
    n_fft = int(sr * window_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    
    # 提取Mel频谱图
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=sr/2
    )
    
    # 转换为对数尺度（可选）
    if use_log:
        # 使用log10并添加小常数避免log(0)
        fbank = np.log10(mel_spectrogram + 1e-10)
    else:
        fbank = mel_spectrogram
    
    return fbank, n_fft, hop_length
