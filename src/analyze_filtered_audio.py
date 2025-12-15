import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, welch

wav_path = "fm_output.wav"
fs, data = wavfile.read(wav_path)

if data.ndim > 1:
    data = data[:, 0]

data = data / np.max(np.abs(data))

# Band-pass filter (300 Hz – 15 kHz)
lowcut = 300
highcut = 15000
order = 4

b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
filtered = filtfilt(b, a, data)

# PSD comparison
f1, Pxx1 = welch(data, fs, nperseg=2048)
f2, Pxx2 = welch(filtered, fs, nperseg=2048)

plt.figure(figsize=(10,4))
plt.semilogy(f1, Pxx1, label="Original")
plt.semilogy(f2, Pxx2, label="Filtered")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.title("PSD Before vs After Filtering")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# --- Save filtered audio as WAV ---
filtered_norm = filtered / np.max(np.abs(filtered))

wavfile.write(
    "fm_output_filtered.wav",
    fs,
    (filtered_norm * 32767).astype(np.int16)
)

print("fm_output_filtered.wav saved successfully")

