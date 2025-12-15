import numpy as np
from scipy.io import wavfile
from pathlib import Path
import sys

def to_mono(x: np.ndarray) -> np.ndarray:
    
    if x.ndim > 1:
        x = x[:, 0]
    return x

def to_float(x: np.ndarray) -> np.ndarray:
    
    x = x.astype(np.float32)
    m = np.max(np.abs(x))
    return x / m if m > 0 else x

def frame_energy(x: np.ndarray, fs: int, frame_ms: float = 50.0):
    frame_len = int(fs * (frame_ms / 1000.0))
    if frame_len <= 0:
        raise ValueError("Frame length too small.")

    n_frames = len(x) // frame_len
    if n_frames < 10:
        raise ValueError(f"Audio too short for {frame_ms} ms framing. Frames={n_frames}")

    x = x[: n_frames * frame_len]
    frames = x.reshape(n_frames, frame_len)
    E = np.mean(frames * frames, axis=1)  # mean power per frame
    return E, frame_len, n_frames

def estimate_snr(x: np.ndarray, fs: int, frame_ms: float = 50.0, noise_pct: float = 20.0, signal_pct: float = 80.0):
    """
    Student-level robust SNR estimate:
    - Split audio into non-overlapping frames (default 50ms).
    - Noise frames: lowest 'noise_pct' percentile energy.
    - Signal frames: highest 'signal_pct' percentile energy.
    - SNR = 10log10(P_signal / P_noise)
    """
    E, frame_len, n_frames = frame_energy(x, fs, frame_ms=frame_ms)

    noise_th = np.percentile(E, noise_pct)
    signal_th = np.percentile(E, signal_pct)

    noise_idx = E <= noise_th
    signal_idx = E >= signal_th

    # Safety: if masks too small, relax
    if np.sum(noise_idx) < 5:
        noise_idx = E <= np.percentile(E, 30)
    if np.sum(signal_idx) < 5:
        signal_idx = E >= np.percentile(E, 70)

    noise_power = float(np.mean(E[noise_idx]))
    signal_power = float(np.mean(E[signal_idx]))

    eps = 1e-12
    snr_db = 10.0 * np.log10((signal_power + eps) / (noise_power + eps))

    duration_s = len(x) / fs
    return {
        "fs": fs,
        "duration_s": duration_s,
        "frame_ms": frame_ms,
        "noise_pct": noise_pct,
        "signal_pct": signal_pct,
        "noise_power": noise_power,
        "signal_power": signal_power,
        "snr_db": float(snr_db),
        "n_frames": int(n_frames),
        "frame_len": int(frame_len),
    }

def read_wav(path: Path):
    fs, data = wavfile.read(str(path))
    data = to_mono(data)
    data = to_float(data)
    return fs, data

def pick_existing(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None

if __name__ == "__main__":
    
    if len(sys.argv) >= 3:
        original_path = Path(sys.argv[1])
        filtered_path = Path(sys.argv[2])
    else:
        original_path = pick_existing([
            "fm_output.wav",
            "fm_output_audio.wav",
            "output.wav"
        ])
        filtered_path = pick_existing([
            "fm_output_filtered.wav",
            "filtered_output.wav",
            "fm_filtered.wav",
            "fm_output_filt.wav"
        ])

        if original_path is None or filtered_path is None:
            print("ERROR: WAV files not found automatically.")
            print("Run like this:")
            print("  python compare_snr.py fm_output.wav fm_output_filtered.wav")
            sys.exit(1)

    fs1, x1 = read_wav(original_path)
    fs2, x2 = read_wav(filtered_path)

  
    if fs1 != fs2:
        print(f"WARNING: Sample rates differ! original={fs1}, filtered={fs2}")
        
    r1 = estimate_snr(x1, fs1, frame_ms=50.0, noise_pct=20.0, signal_pct=80.0)
    r2 = estimate_snr(x2, fs2, frame_ms=50.0, noise_pct=20.0, signal_pct=80.0)

    improvement = r2["snr_db"] - r1["snr_db"]

    print("\n=== SNR Comparison (same method, 50ms frames) ===")
    print(f"Original file : {original_path}")
    print(f"Filtered file : {filtered_path}\n")

    print(f"Original  | duration={r1['duration_s']:.2f}s | fs={r1['fs']} Hz | SNR={r1['snr_db']:.2f} dB")
    print(f"Filtered  | duration={r2['duration_s']:.2f}s | fs={r2['fs']} Hz | SNR={r2['snr_db']:.2f} dB")
    print(f"\nSNR improvement (Filtered - Original) = {improvement:.2f} dB")

    print("\n(Extra details)")
    print(f"Original  noise_power={r1['noise_power']:.6e}, signal_power={r1['signal_power']:.6e}")
    print(f"Filtered  noise_power={r2['noise_power']:.6e}, signal_power={r2['signal_power']:.6e}")
