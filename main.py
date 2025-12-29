import librosa
import numpy as np
from scipy.spatial.distance import cosine
import argparse
import time
import os
import pickle
import hashlib
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


def compute_similarity(mfcc1, mfcc2):
    """
    计算两个MFCC特征矩阵的相似度
    
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


def get_cache_path(audio_path, n_mfcc, window_ms, hop_ms):
    """
    获取音频MFCC缓存文件路径
    
    参数:
        audio_path: 音频文件路径
        n_mfcc: MFCC维度
        window_ms: 窗口长度(毫秒)
        hop_ms: 跳跃长度(毫秒)
    
    返回:
        cache_path: 缓存文件路径
    """
    # 生成缓存标识（基于参数）
    cache_key = f"{n_mfcc}_{window_ms}_{hop_ms}"
    audio_dir = os.path.dirname(audio_path) or '.'
    audio_basename = os.path.basename(audio_path)
    audio_name = os.path.splitext(audio_basename)[0]
    cache_filename = f"{audio_name}_mfcc_cache_{cache_key}.pkl"
    return os.path.join(audio_dir, cache_filename)


def load_source_mfcc_cache(cache_path, source_path):
    """
    加载源音频MFCC缓存
    
    参数:
        cache_path: 缓存文件路径
        source_path: 源音频文件路径
    
    返回:
        如果缓存有效，返回(source_mfcc, source_y, source_sr, source_duration)
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
        print(f"  从缓存加载: {os.path.basename(cache_path)}")
        return cache_data
    except Exception as e:
        print(f"  缓存加载失败: {e}")
        return None


def save_source_mfcc_cache(cache_path, source_mfcc, source_y, source_sr, source_duration):
    """
    保存源音频MFCC缓存
    
    参数:
        cache_path: 缓存文件路径
        source_mfcc: 源音频完整MFCC特征
        source_y: 源音频时间序列
        source_sr: 源音频采样率
        source_duration: 源音频时长
    """
    try:
        cache_data = {
            'source_mfcc': source_mfcc,
            'source_y': source_y,
            'source_sr': source_sr,
            'source_duration': source_duration
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  缓存已保存: {os.path.basename(cache_path)}")
    except Exception as e:
        print(f"  缓存保存失败: {e}")


def find_audio_in_audio(target_path, source_path, n_mfcc=13, threshold=0.7, hop_ratio=0.5, trim_silence_enabled=True, silence_threshold=30, reduce_noise_enabled=False):
    """
    在源音频中查找目标音频片段，同时使用DTW和余弦相似度两种方法
    
    参数:
        target_path: 要查找的目标音频文件路径 (audio1.wav)
        source_path: 源音频文件路径 (audio.wav)
        n_mfcc: MFCC维度
        threshold: 相似度阈值
        hop_ratio: 滑动窗口的跳跃比例 (0-1)
        trim_silence_enabled: 是否移除静音
        silence_threshold: 静音阈值 (dB)
    
    返回:
        matches_dtw: DTW匹配位置列表
        matches_cosine: 余弦相似度匹配位置列表
        best_match_dtw: DTW最佳匹配
        best_match_cosine: 余弦相似度最佳匹配
        positions: 所有搜索位置
        similarities_dtw: DTW相似度列表
        similarities_cosine: 余弦相似度列表
        dtw_distances: DTW距离列表
    """
    print(f"正在加载音频文件...")
    load_start_time = time.time()
    
    if trim_silence_enabled:
        print(f"已启用静音移除（阈值: {silence_threshold}dB）")
    
    # 计算帧参数：帧长25ms，步长10ms
    window_ms = 25
    hop_ms = 10
    
    # 加载目标音频（要查找的片段）
    target_y, target_sr = librosa.load(target_path, sr=None)
    
    # 降噪处理
    if reduce_noise_enabled:
        print(f"处理目标音频: {target_path}")
        target_y = reduce_noise(target_y, target_sr)
        print(f"  已完成降噪")
    
    # 移除目标音频的静音
    if trim_silence_enabled:
        if not reduce_noise_enabled:
            print(f"处理目标音频: {target_path}")
        target_y, removed = trim_silence(target_y, target_sr, top_db=silence_threshold)
        if removed > 0.01:
            print(f"  移除了 {removed:.2f}秒 的静音")
    
    n_fft = int(target_sr * window_ms / 1000)
    hop_length = int(target_sr * hop_ms / 1000)
    target_mfcc = librosa.feature.mfcc(y=target_y, sr=target_sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    target_duration = len(target_y) / target_sr
    
    # 加载源音频（在其中查找）
    # 尝试从缓存加载
    cache_path = get_cache_path(source_path, n_mfcc, window_ms, hop_ms)
    cache_data = load_source_mfcc_cache(cache_path, source_path)
    
    if cache_data is not None:
        # 从缓存加载
        source_mfcc = cache_data['source_mfcc']
        source_y = cache_data['source_y']
        source_sr = cache_data['source_sr']
        source_duration = cache_data['source_duration']
    else:
        # 重新计算
        print(f"处理源音频: {source_path}")
        source_y, source_sr = librosa.load(source_path, sr=None)
        
        # 降噪处理
        if reduce_noise_enabled:
            source_y = reduce_noise(source_y, source_sr)
            print(f"  已完成降噪")
        
        # 注意：源音频不移除静音，保持原始时间位置的准确性
        source_duration = len(source_y) / source_sr
        
        # 计算完整的源音频MFCC
        n_fft_source = int(source_sr * window_ms / 1000)
        hop_length_source = int(source_sr * hop_ms / 1000)
        source_mfcc = librosa.feature.mfcc(y=source_y, sr=source_sr, n_mfcc=n_mfcc, n_fft=n_fft_source, hop_length=hop_length_source)
        
        # 保存到缓存
        save_source_mfcc_cache(cache_path, source_mfcc, source_y, source_sr, source_duration)
    
    load_end_time = time.time()
    load_elapsed = load_end_time - load_start_time
    
    print(f"音频加载完成，耗时: {load_elapsed:.2f}秒")
    print(f"目标音频长度: {target_duration:.2f}秒")
    print(f"源音频长度: {source_duration:.2f}秒")
    
    # 确保采样率一致
    if target_sr != source_sr:
        print(f"警告: 采样率不一致，重新采样到 {source_sr} Hz")
        target_y = librosa.resample(target_y, orig_sr=target_sr, target_sr=source_sr)
        target_sr = source_sr
        n_fft = int(target_sr * window_ms / 1000)
        hop_length = int(target_sr * hop_ms / 1000)
        target_mfcc = librosa.feature.mfcc(y=target_y, sr=target_sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # 计算滑动窗口参数
    target_samples = len(target_y)
    hop_samples = int(target_samples * hop_ratio)
    
    matches_dtw = []
    matches_cosine = []
    similarities_dtw = []
    similarities_cosine = []
    positions = []
    dtw_distances = []
    
    print(f"\n正在搜索匹配位置...")
    print(f"同时使用: DTW对齐 + 余弦相似度")
    search_start_time = time.time()
    
    # 滑动窗口搜索
    for start_sample in range(0, len(source_y) - target_samples + 1, hop_samples):
        end_sample = start_sample + target_samples
        
        # 从预计算的MFCC中提取当前窗口的帧
        # 计算MFCC帧索引
        start_frame = int(start_sample / hop_length)
        # 确保窗口MFCC帧数与目标MFCC帧数相同
        target_frames = target_mfcc.shape[1]
        end_frame = start_frame + target_frames
        
        # 检查是否超出范围
        if end_frame > source_mfcc.shape[1]:
            break
        
        # 切片获取窗口MFCC（无需重新计算）
        window_mfcc = source_mfcc[:, start_frame:end_frame]
        
        # 同时计算两种相似度
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
    
    
    # 重新计算DTW相似度（基于相对距离）
    if len(dtw_distances) > 0:
        min_dist = np.min(dtw_distances)
        max_dist = np.max(dtw_distances)
        dist_range = max_dist - min_dist
        
        if dist_range > 0:
            # 重新计算DTW相似度：最小距离=1.0，最大距离=0.0
            similarities_dtw = [1 - (d - min_dist) / dist_range for d in dtw_distances]
    
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
    matches_both = []
    for i in range(len(similarities_dtw)):
        if similarities_dtw[i] >= threshold and similarities_cosine[i] >= threshold:
            matches_both.append({
                'start_time': positions[i],
                'end_time': positions[i] + target_duration,
                'similarity_dtw': similarities_dtw[i],
                'similarity_cosine': similarities_cosine[i]
            })
    
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
    
    return matches_dtw, matches_cosine, matches_both, best_match_dtw, best_match_cosine, positions, similarities_dtw, similarities_cosine, dtw_distances


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='基于MFCC和DTW的音频相似度比较工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s -t audio1.wav -s audio.wav
  %(prog)s -t audio1.wav -s audio.wav --mfcc 20 --threshold 0.8
  %(prog)s -t audio1.wav -s audio.wav --hop-ratio 0.25
        ''')
    
    parser.add_argument('-t', '--target', 
                        type=str, 
                        default='audio1.wav',
                        help='要查找的目标音频片段 (默认: audio1.wav)')
    
    parser.add_argument('-s', '--source', 
                        type=str, 
                        default='audio.wav',
                        help='源音频文件，在其中搜索 (默认: audio.wav)')
    
    parser.add_argument('--mfcc', 
                        type=int, 
                        default=13,
                        help='MFCC特征维度数量 (默认: 13)')
    
    parser.add_argument('--threshold', 
                        type=float, 
                        default=0.7,
                        help='相似度阈值 (0-1)，超过此值视为匹配 (默认: 0.7)')
    
    parser.add_argument('--hop-ratio', 
                        type=float, 
                        default=0.15,
                        help='滑动窗口跳跃比例 (0-1)，越小越精确但计算越慢 (默认: 0.15)')
    
    parser.add_argument('--no-trim-silence', 
                        action='store_true',
                        default=False,
                        help='禁用静音移除')
    
    parser.add_argument('--silence-threshold', 
                        type=int, 
                        default=30,
                        help='静音阈值 (dB)，低于这个值的声音视为静音 (默认: 30)')
    
    parser.add_argument('--reduce-noise', 
                        action='store_true',
                        help='启用降噪功能（需要安装 noisereduce 库）')
    
    args = parser.parse_args()
    
    # 设置参数
    target_audio = args.target
    source_audio = args.source
    trim_silence_enabled = not args.no_trim_silence
    reduce_noise_enabled = args.reduce_noise
    
    print(f"配置信息:")
    print(f"  目标音频: {target_audio}")
    print(f"  源音频: {source_audio}")
    print(f"  MFCC维度: {args.mfcc}")
    print(f"  相似度阈值: {args.threshold}")
    print(f"  跳跃比例: {args.hop_ratio}")
    print(f"  计算方法: DTW + 余弦相似度（同时计算）")
    print(f"  降噪: {'\u662f' if reduce_noise_enabled else '\u5426'}")
    print(f"  移除静音: {'\u662f' if trim_silence_enabled else '\u5426'}")
    if trim_silence_enabled:
        print(f"  静音阈值: {args.silence_threshold}dB")
    print()
    
    try:
        # 在源音频中查找目标音频，同时使用两种算法
        matches_dtw, matches_cosine, matches_both, best_match_dtw, best_match_cosine, positions, similarities_dtw, similarities_cosine, dtw_distances = find_audio_in_audio(
            target_path=target_audio,
            source_path=source_audio,
            n_mfcc=args.mfcc,
            threshold=args.threshold,
            hop_ratio=args.hop_ratio,
            trim_silence_enabled=trim_silence_enabled,
            silence_threshold=args.silence_threshold,
            reduce_noise_enabled=reduce_noise_enabled
        )
        
        print("\n" + "="*60)
        print("搜索结果:")
        print("="*60)
        # 显示所有超过阈值的匹配 - DTW
        if matches_dtw:
            print(f"\n【DTW匹配位置】 (相似度 ≥ {args.threshold*100:.0f}%):")
            for i, match in enumerate(matches_dtw, 1):
                print(f"  {i}. {match['start_time']:.2f}秒 - {match['end_time']:.2f}秒, "
                      f"相似度: {match['similarity']:.4f} ({match['similarity']*100:.2f}%)")
        
        # 显示所有超过阈值的匹配 - 余弦相似度
        if matches_cosine:
            print(f"\n【余弦相似度匹配位置】 (相似度 ≥ {args.threshold*100:.0f}%):")
            for i, match in enumerate(matches_cosine, 1):
                print(f"  {i}. {match['start_time']:.2f}秒 - {match['end_time']:.2f}秒, "
                      f"相似度: {match['similarity']:.4f} ({match['similarity']*100:.2f}%)")        
         # 显示相似度分布统计
        print(f"\n【相似度统计】")
        print(f"DTW算法:")
        print(f"  平均相似度: {np.mean(similarities_dtw):.4f}")
        print(f"  最大相似度: {np.max(similarities_dtw):.4f}")
        print(f"  最小相似度: {np.min(similarities_dtw):.4f}")
        print(f"\n余弦相似度:")
        print(f"  平均相似度: {np.mean(similarities_cosine):.4f}")
        print(f"  最大相似度: {np.max(similarities_cosine):.4f}")
        print(f"  最小相似度: {np.min(similarities_cosine):.4f}")
        
        print(f"\n【DTW距离统计】")
        print(f"  平均DTW距离: {np.mean(dtw_distances):.2f}")
        print(f"  最小DTW距离: {np.min(dtw_distances):.2f}")
        print(f"  最大DTW距离: {np.max(dtw_distances):.2f}")

        # 显示最佳匹配位置 - DTW
        print(f"\n【最佳匹配位置 - DTW算法】")
        print(f"  起始时间: {best_match_dtw['start_time']:.2f} 秒")
        print(f"  结束时间: {best_match_dtw['end_time']:.2f} 秒")
        print(f"  相似度: {best_match_dtw['similarity']:.4f} ({best_match_dtw['similarity']*100:.2f}%)")
        print(f"  DTW距离: {best_match_dtw['dtw_distance']:.2f} (归一化后，越小越相似)")
        
        # 显示最佳匹配位置 - 余弦相似度
        print(f"\n【最佳匹配位置 - 余弦相似度】")
        print(f"  起始时间: {best_match_cosine['start_time']:.2f} 秒")
        print(f"  结束时间: {best_match_cosine['end_time']:.2f} 秒")
        print(f"  相似度: {best_match_cosine['similarity']:.4f} ({best_match_cosine['similarity']*100:.2f}%)")
        
        # 判断是否存在（综合两种算法）
        # 使用更严格的判断：DTW相对相似度 >= threshold 且 余弦绝对相似度 >= threshold
        dtw_match = best_match_dtw['similarity'] >= args.threshold
        cosine_match = best_match_cosine['similarity'] >= args.threshold
        
        if dtw_match and cosine_match:
            print(f"\n✓ {target_audio} 在 {source_audio} 中存在")
            print(f"  DTW推荐位置: {best_match_dtw['start_time']:.2f}秒 - {best_match_dtw['end_time']:.2f}秒")
            print(f"  余弦相似度推荐位置: {best_match_cosine['start_time']:.2f}秒 - {best_match_cosine['end_time']:.2f}秒")
        elif dtw_match and not cosine_match:
            print(f"\n⚠ {target_audio} 可能在 {source_audio} 中存在（仅DTW检测到）")
            print(f"  DTW推荐位置: {best_match_dtw['start_time']:.2f}秒 - {best_match_dtw['end_time']:.2f}秒")
            print(f"  但余弦相似度较低 ({best_match_cosine['similarity']*100:.2f}%)，可能是误报")
        elif not dtw_match and cosine_match:
            print(f"\n⚠ {target_audio} 可能在 {source_audio} 中存在（仅余弦相似度检测到）")
            print(f"  余弦相似度推荐位置: {best_match_cosine['start_time']:.2f}秒 - {best_match_cosine['end_time']:.2f}秒")
            print(f"  但DTW相似度较低，建议人工核实")
        else:
            print(f"\n✗ {target_audio} 在 {source_audio} 中不存在")
            print(f"  DTW最高相似度: {best_match_dtw['similarity']*100:.2f}%")
            print(f"  余弦最高相似度: {best_match_cosine['similarity']*100:.2f}%")
            print(f"  两种算法均未达到{args.threshold*100:.0f}%阈值")
        
        # 显示同时超过阈值的匹配位置
        if matches_both:
            print(f"\n【两种算法同时匹配的位置】 (DTW ≥ {args.threshold*100:.0f}% 且 余弦 ≥ {args.threshold*100:.0f}%):")
            for i, match in enumerate(matches_both, 1):
                print(f"  {i}. {match['start_time']:.2f}秒 - {match['end_time']:.2f}秒")
                print(f"     DTW相似度: {match['similarity_dtw']:.4f} ({match['similarity_dtw']*100:.2f}%), "
                      f"余弦相似度: {match['similarity_cosine']:.4f} ({match['similarity_cosine']*100:.2f}%)")
        else:
            print(f"\n【两种算法同时匹配的位置】 (DTW ≥ {args.threshold*100:.0f}% 且 余弦 ≥ {args.threshold*100:.0f}%):")
            print(f"  未找到同时满足两种算法阈值的位置")
                
    except FileNotFoundError as e:
        print(f"错误: 找不到音频文件")
        print(f"请确保以下文件存在:")
        print(f"  - {target_audio}")
        print(f"  - {source_audio}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
