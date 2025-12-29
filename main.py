"""
音频相似度比较工具 - 主入口
基于MFCC和DTW的音频片段匹配
"""

import argparse
import numpy as np

from audio_matcher import find_audio_in_audio


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

    parser.add_argument('--feature-type',
                        type=str,
                        choices=['mfcc', 'fbank'],
                        default='mfcc',
                        help='特征类型: mfcc 或 fbank (默认: mfcc)')

    parser.add_argument('--fbank',
                        type=int,
                        default=40,
                        help='Fbank滤波器数量 (默认: 40)')

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

    parser.add_argument('--sample-rate',
                        type=int,
                        default=8000,
                        help='目标采样率 (Hz)，所有音频将重采样到此采样率 (默认: 8000)')

    args = parser.parse_args()

    # 设置参数
    target_audio = args.target
    source_audio = args.source
    trim_silence_enabled = not args.no_trim_silence
    reduce_noise_enabled = args.reduce_noise

    # 根据特征类型确定特征维度
    feature_type = args.feature_type
    if feature_type == 'fbank':
        n_features = args.fbank
        feature_name = f"Fbank（{n_features}个滤波器）"
    else:
        n_features = args.mfcc
        feature_name = f"MFCC（{n_features}维）"
    
    print(f"配置信息:")
    print(f"  目标音频: {target_audio}")
    print(f"  源音频: {source_audio}")
    print(f"  特征类型: {feature_name}")
    print(f"  目标采样率: {args.sample_rate} Hz")
    print(f"  相似度阈值: {args.threshold}")
    print(f"  跳跃比例: {args.hop_ratio}")
    print(f"  计算方法: DTW + 余弦相似度（同时计算）")
    print(f"  降噪: {'是' if reduce_noise_enabled else '否'}")
    print(f"  移除静音: {'是' if trim_silence_enabled else '否'}")
    if trim_silence_enabled:
        print(f"  静音阈值: {args.silence_threshold}dB")
    print()

    try:
        # 在源音频中查找目标音频，同时使用两种算法
        matches_dtw, matches_cosine, matches_both, best_match_dtw, best_match_cosine, positions, similarities_dtw, similarities_cosine, dtw_distances = find_audio_in_audio(
            target_path=target_audio,
            source_path=source_audio,
            feature_type=feature_type,
            n_features=n_features,
            threshold=args.threshold,
            hop_ratio=args.hop_ratio,
            trim_silence_enabled=trim_silence_enabled,
            silence_threshold=args.silence_threshold,
            reduce_noise_enabled=reduce_noise_enabled,
            target_sr=args.sample_rate
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
        print(
            f"  相似度: {best_match_dtw['similarity']:.4f} ({best_match_dtw['similarity']*100:.2f}%)")
        print(f"  DTW距离: {best_match_dtw['dtw_distance']:.2f} (归一化后，越小越相似)")

        # 显示最佳匹配位置 - 余弦相似度
        print(f"\n【最佳匹配位置 - 余弦相似度】")
        print(f"  起始时间: {best_match_cosine['start_time']:.2f} 秒")
        print(f"  结束时间: {best_match_cosine['end_time']:.2f} 秒")
        print(
            f"  相似度: {best_match_cosine['similarity']:.4f} ({best_match_cosine['similarity']*100:.2f}%)")

        # 判断是否存在（综合两种算法）
        # 使用更严格的判断：DTW相对相似度 >= threshold 且 余弦绝对相似度 >= threshold
        dtw_match = best_match_dtw['similarity'] >= args.threshold
        cosine_match = best_match_cosine['similarity'] >= args.threshold

        if dtw_match and cosine_match:
            print(f"\n✓ {target_audio} 在 {source_audio} 中存在")
            print(
                f"  DTW相似度 推荐位置: {best_match_dtw['start_time']:.2f}秒 - {best_match_dtw['end_time']:.2f}秒 相似度({best_match_dtw['similarity']*100:.2f}%)")
            print(
                f"  余弦相似度 推荐位置: {best_match_cosine['start_time']:.2f}秒 - {best_match_cosine['end_time']:.2f}秒 相似度({best_match_cosine['similarity']*100:.2f}%)")
        elif dtw_match and not cosine_match:
            print(f"\n⚠ {target_audio} 可能在 {source_audio} 中存在（仅DTW检测到）")
            print(
                f"  DTW相似度 推荐位置: {best_match_dtw['start_time']:.2f}秒 - {best_match_dtw['end_time']:.2f}秒 相似度({best_match_dtw['similarity']*100:.2f}%)")
            print(f"  余弦相似度较低, 相似度({best_match_cosine['similarity']*100:.2f}%)")
        elif not dtw_match and cosine_match:
            print(f"\n⚠ {target_audio} 可能在 {source_audio} 中存在（仅余弦相似度检测到）")
            print(
                f"  余弦相似度 推荐位置: {best_match_cosine['start_time']:.2f}秒 - {best_match_cosine['end_time']:.2f}秒 相似度({best_match_cosine['similarity']*100:.2f}%)")
            print(f"  DTW相似度较低, 相似度({best_match_dtw['similarity']*100:.2f}%)")
        else:
            print(f"\n✗ {target_audio} 在 {source_audio} 中不存在")
            print(f"  DTW最高相似度: {best_match_dtw['similarity']*100:.2f}%")
            print(f"  余弦最高相似度: {best_match_cosine['similarity']*100:.2f}%")
            print(f"  两种算法均未达到{args.threshold*100:.0f}%阈值")

        # 显示同时超过阈值的匹配位置
        if matches_both:
            print(
                f"\n【两种算法同时匹配的位置】 (DTW ≥ {args.threshold*100:.0f}% 且 余弦 ≥ {args.threshold*100:.0f}%):")
            for i, match in enumerate(matches_both, 1):
                print(
                    f"  {i}. {match['start_time']:.2f}秒 - {match['end_time']:.2f}秒")
                print(f"     DTW相似度: {match['similarity_dtw']:.4f} ({match['similarity_dtw']*100:.2f}%), "
                      f"余弦相似度: {match['similarity_cosine']:.4f} ({match['similarity_cosine']*100:.2f}%)")
        else:
            print(
                f"\n【两种算法同时匹配的位置】 (DTW ≥ {args.threshold*100:.0f}% 且 余弦 ≥ {args.threshold*100:.0f}%):")
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
