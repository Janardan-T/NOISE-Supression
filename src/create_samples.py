"""Generate synthetic audio test samples with various background noise types.

Creates:
1. samples/clean_speech_ref.wav       - Clean synthetic vowel/formant tone sequences
2. samples/speech_with_whitenoise.wav  - Speech with broadband white Gaussian noise
3. samples/speech_with_hum.wav        - Speech with 50/100/150 Hz electrical hum
4. samples/speech_with_buzz.wav       - Speech with harsh buzzy noise
"""

from pathlib import Path
import numpy as np
import soundfile as sf


def generate_formant_tone(
    f0: float,
    duration: float,
    sr: int = 48000,
    formants=(800, 1200, 2500),
) -> np.ndarray:
    """Synthesize a voiced speech-like harmonic tone with vocal tract formants."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)

    # Sum harmonics up to 4000 Hz
    num_harmonics = int(4000 / f0)
    for n in range(1, num_harmonics + 1):
        freq = n * f0
        # Formant resonance weighting
        amp = 0.0
        for fc in formants:
            bw = 100.0  # bandwidth
            amp += np.exp(-((freq - fc) ** 2) / (2 * (bw ** 2)))
        amp = max(amp, 0.02) / (1 + (freq / 1000.0))
        signal += amp * np.sin(2 * np.pi * freq * t)

    # Smooth envelope (attack and decay)
    env = np.ones_like(t)
    fade_len = int(sr * 0.05)
    if len(t) > 2 * fade_len:
        env[:fade_len] = np.linspace(0, 1, fade_len)
        env[-fade_len:] = np.linspace(1, 0, fade_len)
    return signal * env


def create_synthetic_sentence(sr: int = 48000) -> np.ndarray:
    """Create a 4-second sequence of vowel/syllable-like bursts with pauses."""
    notes = [
        (130, 0.4, (700, 1200, 2600)),   # /a/
        (145, 0.35, (300, 2200, 3000)),  # /i/
        (138, 0.4, (500, 900, 2400)),    # /o/
        (0,   0.25, ()),                 # pause
        (135, 0.45, (400, 2000, 2700)),  # /e/
        (150, 0.35, (700, 1200, 2600)),  # /a/
        (140, 0.5, (300, 900, 2300)),    # /u/
        (0,   0.3, ()),                  # pause
        (128, 0.5, (500, 1000, 2500)),   # /o/
    ]
    parts = []
    for f0, dur, formants in notes:
        if f0 == 0:
            parts.append(np.zeros(int(sr * dur)))
        else:
            parts.append(generate_formant_tone(f0, dur, sr, formants))
    full = np.concatenate(parts)
    # Normalize peak to 0.7
    full = full / np.max(np.abs(full)) * 0.7
    return full


def add_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix noise into clean audio at a specified Signal-to-Noise Ratio (dB)."""
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return clean

    desired_noise_power = clean_power / (10 ** (snr_db / 10))
    scaled_noise = noise * np.sqrt(desired_noise_power / noise_power)
    noisy = clean + scaled_noise

    # Normalize to prevent digital clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 0.95:
        noisy = noisy / max_val * 0.95
    return noisy


def main():
    samples_dir = Path("samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    sr = 48000

    print("Generating synthetic speech signal...")
    clean = create_synthetic_sentence(sr=sr)
    clean_path = samples_dir / "clean_speech_ref.wav"
    sf.write(str(clean_path), clean, samplerate=sr, subtype="PCM_16")
    print(f"  Saved: {clean_path} ({len(clean)/sr:.2f}s)")

    # 1. White Gaussian Noise (Broadband hiss, SNR = 3 dB)
    rng = np.random.default_rng(seed=42)
    white_noise = rng.normal(0, 1, size=len(clean))
    noisy_white = add_noise(clean, white_noise, snr_db=3.0)
    white_path = samples_dir / "speech_with_whitenoise.wav"
    sf.write(str(white_path), noisy_white, samplerate=sr, subtype="PCM_16")
    print(f"  Saved: {white_path}")

    # 2. Powerline Hum (50Hz + 100Hz + 150Hz harmonics, SNR = 5 dB)
    t = np.linspace(0, len(clean) / sr, len(clean), endpoint=False)
    hum = (
        0.6 * np.sin(2 * np.pi * 50 * t)
        + 0.3 * np.sin(2 * np.pi * 100 * t)
        + 0.15 * np.sin(2 * np.pi * 150 * t)
    )
    noisy_hum = add_noise(clean, hum, snr_db=5.0)
    hum_path = samples_dir / "speech_with_hum.wav"
    sf.write(str(hum_path), noisy_hum, samplerate=sr, subtype="PCM_16")
    print(f"  Saved: {hum_path}")

    # 3. Harsh Buzz / Generator noise (Sawtooth wave + pink-ish noise, SNR = 2 dB)
    buzz = np.zeros_like(t)
    for harmonic in range(1, 20):
        buzz += (1.0 / harmonic) * np.sin(2 * np.pi * 120 * harmonic * t)
    pink_filter = np.convolve(rng.normal(0, 1, size=len(clean)), np.ones(10)/10, mode="same")
    buzzy_noise = buzz * 0.7 + pink_filter * 0.3
    noisy_buzz = add_noise(clean, buzzy_noise, snr_db=2.0)
    buzz_path = samples_dir / "speech_with_buzz.wav"
    sf.write(str(buzz_path), noisy_buzz, samplerate=sr, subtype="PCM_16")
    print(f"  Saved: {buzz_path}")

    print("\nSample generation complete!")


if __name__ == "__main__":
    main()
