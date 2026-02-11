#!/usr/bin/env python3
"""Generate sound textures as .wav files with randomized parameters."""

import os
import json
import argparse
import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf

SAMPLE_RATE = 44100
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def generate_white_noise(n_samples):
    return np.random.randn(n_samples)


def generate_pink_noise(n_samples):
    """Voss-McCartney algorithm for pink noise."""
    white = np.random.randn(n_samples)
    # Simple spectral shaping: apply 1/sqrt(f) rolloff via cumulative filtering
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    pink = lfilter(b, a, white)
    # Normalize
    pink = pink / (np.max(np.abs(pink)) + 1e-10)
    return pink


def generate_brown_noise(n_samples):
    """Brown noise via cumulative sum of white noise."""
    white = np.random.randn(n_samples)
    brown = np.cumsum(white)
    brown = brown / (np.max(np.abs(brown)) + 1e-10)
    return brown


NOISE_GENERATORS = {
    "white": generate_white_noise,
    "pink": generate_pink_noise,
    "brown": generate_brown_noise,
}


def apply_lowpass(signal, cutoff, sample_rate):
    """Apply a Butterworth lowpass filter."""
    nyquist = sample_rate / 2.0
    cutoff = min(cutoff, nyquist * 0.99)
    b, a = butter(4, cutoff / nyquist, btype="low")
    return lfilter(b, a, signal)


def apply_modulation(signal, mod_rate, mod_depth, duration, sample_rate):
    """Apply amplitude modulation (tremolo) via LFO."""
    t = np.linspace(0, duration, len(signal), endpoint=False)
    lfo = 1.0 - mod_depth * 0.5 * (1.0 - np.cos(2 * np.pi * mod_rate * t))
    return signal * lfo


def apply_base_frequency(signal, base_freq, duration, sample_rate):
    """Mix a sine tone at the base frequency into the noise."""
    t = np.linspace(0, duration, len(signal), endpoint=False)
    tone = np.sin(2 * np.pi * base_freq * t)
    # Blend: 60% noise, 40% tone
    return 0.6 * signal + 0.4 * tone


def generate_texture(params):
    """Generate a single sound texture from parameters dict."""
    n_samples = int(params["duration"] * SAMPLE_RATE)

    # Generate base noise
    noise_fn = NOISE_GENERATORS[params["noise_type"]]
    signal = noise_fn(n_samples)

    # Apply lowpass filter
    signal = apply_lowpass(signal, params["filter_cutoff"], SAMPLE_RATE)

    # Mix in base frequency tone
    signal = apply_base_frequency(
        signal, params["base_frequency"], params["duration"], SAMPLE_RATE
    )

    # Apply amplitude modulation
    signal = apply_modulation(
        signal, params["mod_rate"], params["mod_depth"],
        params["duration"], SAMPLE_RATE
    )

    # Apply amplitude and normalize
    signal = signal * params["amplitude"]
    peak = np.max(np.abs(signal))
    if peak > 1.0:
        signal = signal / peak * params["amplitude"]

    return signal


def random_params():
    """Generate a random set of texture parameters."""
    return {
        "noise_type": np.random.choice(["white", "pink", "brown"]),
        "base_frequency": round(np.random.uniform(50, 500), 1),
        "duration": round(np.random.uniform(2.0, 8.0), 1),
        "amplitude": round(np.random.uniform(0.3, 1.0), 2),
        "filter_cutoff": round(np.random.uniform(200, 8000), 1),
        "mod_rate": round(np.random.uniform(0.5, 10.0), 2),
        "mod_depth": round(np.random.uniform(0.0, 1.0), 2),
    }


def generate_batch(count=10, seed=None):
    """Generate a batch of textures, returning list of (filename, params)."""
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    for i in range(count):
        params = random_params()
        filename = f"texture_{i+1:03d}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)

        signal = generate_texture(params)
        sf.write(filepath, signal, SAMPLE_RATE)

        params["filename"] = filename
        results.append(params)
        print(f"Generated {filename}: {params['noise_type']}, "
              f"{params['base_frequency']}Hz, {params['duration']}s")

    # Save params as JSON for db_insert.py to use
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGenerated {count} textures in {OUTPUT_DIR}")
    print(f"Manifest saved to {manifest_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sound textures")
    parser.add_argument("-n", "--count", type=int, default=10,
                        help="Number of textures to generate (default: 10)")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_batch(count=args.count, seed=args.seed)
