import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch

wav_path = "fm_output.wav"

# Read WAV
fs, data = wavfile.read(wav_path)

#  If stereo, take 1 channel
if data.ndim > 1:
    data = data[:, 0]

#  Convert to float and normalize
data = data.astype(np.float32)
data = data / (np.max(np.abs(data)) + 1e-12)

print("Sample Rate (Hz):", fs)
print("Shape:", data.shape)
print("Duration (s):", len(data) / fs)

# 4) Frame-based energy to estimate noise floor
frame_ms = 50
frame_len = int(fs * frame_ms / 1000)
hop_len = frame_len  # non-overlap (simple)

num_frames = (len(data) - frame_len) // hop_len
energies = []

for i in range(num_frames):
    start = i * hop_len
    frame = data[start:start + frame_len]
    e = np.mean(frame**2)
    energies.append(e)

energies = np.array(energies)

# pick lowest 20% frames as "noise"
thr = np.percentile(energies, 20)
noise_frames = energies[energies <= thr]
signal_frames = energies[energies > thr]

noise_power = np.mean(noise_frames) + 1e-12
signal_power = np.mean(signal_frames) + 1e-12

snr_db = 10 * np.log10(signal_power / noise_power)

print("\nEstimated noise power:", noise_power)
print("Estimated signal power:", signal_power)
print("Estimated SNR (dB):", snr_db)

#  PSD (Welch) for visualization
f, Pxx = welch(data, fs=fs, nperseg=4096)

plt.figure(figsize=(10,4))
plt.semilogy(f, Pxx)
plt.title("PSD of FM Audio (Welch)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.xlim(0, 20000)
plt.grid(True)
plt.tight_layout()
plt.show()

#  Energy over time plot
plt.figure(figsize=(10,4))
plt.plot(energies)
plt.axhline(thr, linestyle="--")
plt.title("Frame Energy (50 ms) + Noise Threshold (20th percentile)")
plt.xlabel("Frame index")
plt.ylabel("Mean(frame^2)")
plt.grid(True)
plt.tight_layout()
plt.show()
