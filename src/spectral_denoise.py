import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
in_wav = "fm_output.wav"
out_wav = "fm_output_denoised.wav"

frame_sec = 0.05        # 50 ms (same frame idea in report)
overlap = 0.75          # STFT overlap
noise_percentile = 20   # lowest energy %20 frame = "noise-only" assumption
floor_db = -80          # for log  base
eps = 1e-12

# -------------------------
# Read WAV
# -------------------------
fs, x = wavfile.read(in_wav)

# mono do it 
if x.ndim > 1:
    x = x[:, 0]

# normalize (float)
x = x.astype(np.float32)
x = x / (np.max(np.abs(x)) + eps)

# -------------------------
# STFT
# -------------------------
nperseg = int(frame_sec * fs)
nperseg = max(256, nperseg)           
noverlap = int(nperseg * overlap)

f, t, Zxx = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann", padded=False, boundary=None)
mag = np.abs(Zxx)
phase = np.angle(Zxx)

# -------------------------
# Noise profile estimation
# 1) Frame energy -> %20 select lowest energy frames
# -------------------------
frame_energy = np.mean(mag**2, axis=0)  # every time frame energy
thr = np.percentile(frame_energy, noise_percentile)
noise_frames = mag[:, frame_energy <= thr]

# Eğer çok az frame seçildiyse fallback
if noise_frames.shape[1] < 5:
    noise_frames = mag[:, :max(5, mag.shape[1]//10)]

noise_mag = np.median(noise_frames, axis=1, keepdims=True)  # (freq, 1)

# -------------------------
# Spectral gain (Wiener-like)
# gain = max(0, 1 - (noise_mag / mag))^p
# -------------------------
p = 2.0
gain = 1.0 - (noise_mag / (mag + eps))
gain = np.clip(gain, 0.0, 1.0) ** p

# Uygula
mag_d = mag * gain
Zxx_d = mag_d * np.exp(1j * phase)

# -------------------------
# iSTFT -> time signal
# -------------------------
_, y = istft(Zxx_d, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann", input_onesided=True)

# Normalize + int16 saved
y = y / (np.max(np.abs(y)) + eps)
wavfile.write(out_wav, fs, (y * 32767).astype(np.int16))

print(f"Saved: {out_wav}")
print(f"STFT params: nperseg={nperseg}, noverlap={noverlap}, noise_percentile={noise_percentile}, p={p}")

# -------------------------
# Plots (for report)
# -------------------------
# 1) Average spectrum before/after
avg_before = np.mean(mag, axis=1)
avg_after  = np.mean(mag_d, axis=1)

plt.figure(figsize=(10,4))
plt.semilogy(f, avg_before + eps, label="Original avg |Z|")
plt.semilogy(f, avg_after + eps, label="Denoised avg |Z|")
plt.title("Average STFT Magnitude (Before vs After)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (log)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("avg_spectrum_before_after.png")
plt.show()

# 2) Gain heatmap (sample)
plt.figure(figsize=(10,4))
plt.imshow(gain, aspect="auto", origin="lower",
           extent=[t[0], t[-1], f[0], f[-1]])
plt.title("Spectral Gain Mask (time-frequency)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Gain")
plt.tight_layout()
plt.savefig("gain_mask.png")
plt.show()
