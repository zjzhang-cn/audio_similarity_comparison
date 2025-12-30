import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import warnings

# 过滤 torchaudio.load 的弃用警告
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 黑体或微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_mono_audio(path: str, device: torch.device):
    """读取音频并转单声道。"""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.to(device), sr


def compute_mfcc(waveform: torch.Tensor, sr: int, n_mfcc: int, hop_length: int):
    transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 2048, "hop_length": hop_length},
    ).to(waveform.device)  # 确保 transform 在与输入相同的设备上
    mfcc = transform(waveform)  # (channel, n_mfcc, time)
    return mfcc.mean(dim=0)  # (n_mfcc, time)


def normalize(mfcc: torch.Tensor):
    mean = mfcc.mean(dim=1, keepdim=True)
    std = mfcc.std(dim=1, keepdim=True) + 1e-8
    return (mfcc - mean) / std


def stats_features(mfcc: torch.Tensor):
    return torch.cat(
        [
            mfcc.mean(dim=1),
            mfcc.std(dim=1),
            mfcc.max(dim=1).values,
            mfcc.min(dim=1).values,
        ]
    )


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='音频MFCC特征提取和相似度计算 (PyTorch)')
    parser.add_argument('-t', '--target', required=True, help='目标音频文件路径')
    parser.add_argument('-s', '--source', required=True, help='查询音频文件路径')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 参数配置
    sr = 16000
    n_mfcc = 40
    target_file = args.target
    source_file = args.source
    n_fft = 512  # 25ms
    hop_length = int(sr * 0.01)  # 10ms
    win_length = int(sr * 0.025)  # 25ms

    target_waveform, target_sr = load_mono_audio(target_file, device)
    source_waveform, source_sr = load_mono_audio(source_file, device)

    # 重采样到目标采样率
    if target_sr != sr:
        resampler = torchaudio.transforms.Resample(target_sr, sr).to(device)
        target_waveform = resampler(target_waveform)
    if source_sr != sr:
        resampler = torchaudio.transforms.Resample(source_sr, sr).to(device)
        source_waveform = resampler(source_waveform)

    target_mfcc = compute_mfcc(target_waveform, sr, n_mfcc, hop_length)
    source_mfcc = compute_mfcc(source_waveform, sr, n_mfcc, hop_length)

    # 整体匹配：直接展平后计算余弦相似度
    min_time = min(target_mfcc.shape[1], source_mfcc.shape[1])
    target_flat = target_mfcc[:, :min_time].reshape(-1)
    source_flat = source_mfcc[:, :min_time].reshape(-1)
    overall_similarity = cosine_sim(target_flat, source_flat)

    print("\n整体匹配：")
    print(f"相似度 (余弦相似度): {overall_similarity:.4f}")

    # 滑动窗口匹配
    print("\n滑动窗口匹配：")
    query_frames = source_mfcc.shape[1]
    target_frames = target_mfcc.shape[1]

    if query_frames > target_frames:
        print("警告：查询音频比目标音频长，交换角色进行匹配")
        target_mfcc, source_mfcc = source_mfcc, target_mfcc
        query_frames = source_mfcc.shape[1]
        target_frames = target_mfcc.shape[1]

    print(f"查询片段长度: {query_frames} 帧")
    print(f"目标音频长度: {target_frames} 帧")

    query_norm = normalize(source_mfcc)
    query_flat = query_norm.reshape(-1)
    query_stats = stats_features(source_mfcc)

    best_similarity = -1.0
    best_position = 0
    similarities = []

    for i in range(target_frames - query_frames + 1):
        window = target_mfcc[:, i:i + query_frames]

        window_norm = normalize(window)
        window_flat = window_norm.reshape(-1)
        window_stats = stats_features(window)

        sim_norm = cosine_sim(window_flat, query_flat)
        sim_stats = cosine_sim(window_stats, query_stats)
        sim = 0.7 * sim_norm + 0.3 * sim_stats
        similarities.append(sim)

        if sim > best_similarity:
            best_similarity = sim
            best_position = i

    best_time = best_position * hop_length / sr
    query_duration = query_frames * hop_length / sr

    print(f"\n最佳匹配位置: 帧 {best_position} (时间: {best_time:.2f}秒)")
    print(f"最佳匹配相似度: {best_similarity:.4f}")
    print(f"匹配片段时间范围: {best_time:.2f}秒 - {best_time + query_duration:.2f}秒")

    # 绘制相似度曲线
    plt.figure(figsize=(12, 4))
    time_positions = np.arange(len(similarities)) * hop_length / sr
    plt.plot(time_positions, similarities, linewidth=1)
    plt.axvline(x=best_time, color='r', linestyle='--', label=f'最佳匹配位置 ({best_time:.2f}秒)')
    plt.axhline(y=best_similarity, color='g', linestyle='--', alpha=0.5, label=f'最佳相似度 ({best_similarity:.4f})')
    plt.xlabel('时间 (秒)')
    plt.ylabel('相似度')
    plt.title('滑动窗口相似度曲线 (PyTorch)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()