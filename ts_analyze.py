#!/usr/bin/env python3
"""Time series analysis of generated sound textures.

Usage:
    python ts_analyze.py                          # analyze all .wav files in output/
    python ts_analyze.py texture_20260211_001.wav  # analyze a specific file
    python ts_analyze.py --compare                 # compare all textures by features

Produces waveform plots, spectrograms, autocorrelation plots, and feature
summaries. Outputs saved to plots/.
"""

import os
import sys
import glob
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram as scipy_spectrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def load_texture(filepath):
    """Load a .wav file, return (signal, sample_rate)."""
    signal, sr = sf.read(filepath)
    return signal, sr


def plot_waveform(signal, sr, title, save_path):
    """Plot amplitude over time."""
    t = np.arange(len(signal)) / sr
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, signal, linewidth=0.3, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform: {title}")
    ax.set_xlim(0, t[-1])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_spectrogram(signal, sr, title, save_path):
    """Plot frequency content over time."""
    nperseg = min(2048, len(signal))
    f, t, Sxx = scipy_spectrogram(signal, fs=sr, nperseg=nperseg,
                                   noverlap=nperseg // 2)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10),
                       shading="gouraud", cmap="magma")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Spectrogram: {title}")
    ax.set_ylim(0, 8000)
    fig.colorbar(im, ax=ax, label="Power (dB)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_autocorrelation(signal, sr, title, save_path, max_lag_ms=50):
    """Plot autocorrelation up to max_lag_ms milliseconds."""
    max_lag = int(sr * max_lag_ms / 1000)
    # Use a chunk from the middle of the signal for stability
    mid = len(signal) // 2
    chunk_size = min(sr * 2, len(signal))  # 2 seconds or less
    chunk = signal[mid - chunk_size // 2 : mid + chunk_size // 2]

    # Normalize
    chunk = chunk - np.mean(chunk)
    norm = np.sum(chunk ** 2)
    if norm < 1e-10:
        return

    autocorr = np.correlate(chunk, chunk, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]  # positive lags only
    autocorr = autocorr / norm  # normalize so lag 0 = 1.0

    lags_ms = np.arange(max_lag) / sr * 1000

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(lags_ms, autocorr[:max_lag], color="coral", linewidth=0.8)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"Autocorrelation: {title}")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_features(signal, sr):
    """Extract time series features from a signal."""
    # RMS energy
    rms = np.sqrt(np.mean(signal ** 2))

    # Zero-crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal))) > 0)
    zcr = zero_crossings / len(signal) * sr  # crossings per second

    # Spectral centroid (brightness)
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    spectral_centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-10)

    # Spectral rolloff (frequency below which 85% of energy lies)
    cumulative_energy = np.cumsum(fft ** 2)
    total_energy = cumulative_energy[-1]
    rolloff_idx = np.searchsorted(cumulative_energy, 0.85 * total_energy)
    spectral_rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    # Autocorrelation at 1ms lag (smoothness measure)
    lag_1ms = int(sr * 0.001)
    chunk = signal - np.mean(signal)
    norm = np.sum(chunk ** 2)
    if norm > 1e-10 and lag_1ms < len(chunk):
        ac_1ms = np.sum(chunk[:-lag_1ms] * chunk[lag_1ms:]) / norm
    else:
        ac_1ms = 0.0

    return {
        "rms_energy": round(rms, 4),
        "zero_crossing_rate": round(zcr, 1),
        "spectral_centroid_hz": round(spectral_centroid, 1),
        "spectral_rolloff_hz": round(spectral_rolloff, 1),
        "autocorr_1ms": round(ac_1ms, 4),
    }


def analyze_single(filepath):
    """Full analysis of a single texture."""
    name = os.path.splitext(os.path.basename(filepath))[0]
    signal, sr = load_texture(filepath)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"\n=== {name} ===")
    print(f"  Duration: {len(signal)/sr:.2f}s, Sample rate: {sr}Hz")

    # Plots
    plot_waveform(signal, sr, name,
                  os.path.join(PLOTS_DIR, f"{name}_waveform.png"))
    plot_spectrogram(signal, sr, name,
                     os.path.join(PLOTS_DIR, f"{name}_spectrogram.png"))
    plot_autocorrelation(signal, sr, name,
                         os.path.join(PLOTS_DIR, f"{name}_autocorr.png"))

    # Features
    features = compute_features(signal, sr)
    for k, v in features.items():
        print(f"  {k}: {v}")

    print(f"  Plots saved to {PLOTS_DIR}")
    return features


def compare_all():
    """Extract features from all textures and produce comparison plots."""
    wav_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.wav")))
    if not wav_files:
        print("No .wav files found in output/")
        return

    print(f"Analyzing {len(wav_files)} textures...\n")

    names = []
    all_features = []
    for filepath in wav_files:
        name = os.path.splitext(os.path.basename(filepath))[0]
        signal, sr = load_texture(filepath)
        features = compute_features(signal, sr)
        names.append(name)
        all_features.append(features)

    # Print feature table
    keys = list(all_features[0].keys())
    print(f"{'texture':<45} ", end="")
    for k in keys:
        print(f"{k:<22}", end="")
    print()
    print("-" * (45 + 22 * len(keys)))
    for name, feat in zip(names, all_features):
        short = name[-30:] if len(name) > 30 else name
        print(f"{short:<45} ", end="")
        for k in keys:
            print(f"{feat[k]:<22}", end="")
        print()

    # Comparison plots
    os.makedirs(PLOTS_DIR, exist_ok=True)
    values = {k: [f[k] for f in all_features] for k in keys}

    # Scatter: spectral centroid vs autocorrelation
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(values["spectral_centroid_hz"], values["autocorr_1ms"],
               alpha=0.6, color="steelblue", edgecolors="white", s=50)
    ax.set_xlabel("Spectral Centroid (Hz)")
    ax.set_ylabel("Autocorrelation at 1ms lag")
    ax.set_title("Brightness vs Smoothness")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "compare_centroid_vs_autocorr.png"),
                dpi=150)
    plt.close(fig)

    # Histogram of spectral centroids
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values["spectral_centroid_hz"], bins=15,
            color="coral", edgecolor="white")
    ax.set_xlabel("Spectral Centroid (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Brightness Across Textures")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "compare_centroid_hist.png"), dpi=150)
    plt.close(fig)

    # Histogram of zero-crossing rates
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values["zero_crossing_rate"], bins=15,
            color="mediumpurple", edgecolor="white")
    ax.set_xlabel("Zero Crossing Rate (per second)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Zero Crossing Rates")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "compare_zcr_hist.png"), dpi=150)
    plt.close(fig)

    print(f"\nComparison plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_all()
    elif len(sys.argv) > 1:
        filepath = sys.argv[1]
        if not os.path.isabs(filepath):
            filepath = os.path.join(OUTPUT_DIR, filepath)
        analyze_single(filepath)
    else:
        # Analyze the 3 most recent files
        wav_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.wav")))
        if not wav_files:
            print("No .wav files found in output/")
        else:
            recent = wav_files[-3:]
            for f in recent:
                analyze_single(f)
