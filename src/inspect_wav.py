import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# WAV file path 
wav_path = "fm_output.wav"

# Read WAV file
fs, data = wavfile.read(wav_path)

print("Sample Rate (Hz):", fs)
print("Data type:", data.dtype)
print("Shape:", data.shape)

# If stereo, take one channel
if data.ndim > 1:
    data = data[:, 0]

# Normalize if needed
data = data / np.max(np.abs(data))

# Time axis (first 0.05 seconds)
t = np.arange(len(data)) / fs
duration = 0.05
samples = int(fs * duration)

# Plot waveform
plt.figure(figsize=(10, 4))
plt.plot(t[:samples], data[:samples])
plt.title("FM Audio Signal (Time Domain)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Frequency Domain (FFT) ---
N = len(data)
fft = np.fft.rfft(data)
freqs = np.fft.rfftfreq(N, d=1/fs)
mag = 20*np.log10(np.abs(fft) + 1e-12)

plt.figure(figsize=(10,4))
plt.plot(freqs, mag)
plt.title("FM Audio Signal (Frequency Domain)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Spectrogram ---
plt.figure(figsize=(10,4))
plt.specgram(data, NFFT=2048, Fs=fs, noverlap=1024)
plt.title("FM Audio Signal (Spectrogram)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
plt.show()

