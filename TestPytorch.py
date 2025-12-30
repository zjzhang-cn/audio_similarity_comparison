import torch
import torchaudio

# 检查 PyTorch 版本
print("PyTorch 版本:", torch.__version__)

# 检查 TorchAudio 版本
print("TorchAudio 版本:", torchaudio.__version__)

# 检查 CUDA 是否可用（可选）
print("CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 设备数量:", torch.cuda.device_count())
    print("当前 CUDA 设备:", torch.cuda.current_device())
    print("CUDA 设备名称:", torch.cuda.get_device_name(0))

# 简单测试：创建一个音频张量（模拟 1 秒 16kHz 单声道音频）
sample_rate = 16000
duration = 1  # 秒
t = torch.linspace(0, duration, int(sample_rate * duration))
waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440Hz 正弦波，单声道

print("生成的音频张量形状:", waveform.shape)
print("采样率:", sample_rate)

# 可选：保存为 WAV 文件（需要 torchaudio 支持）
try:
    torchaudio.save("test_tone.wav", waveform, sample_rate)
    print("音频已保存为 test_tone.wav")
except Exception as e:
    print("保存音频时出错:", e)