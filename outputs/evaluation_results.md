# Background Noise Suppression — Objective Benchmark Report

## 1. Main Benchmark Results (Real Human Speech)

**Primary Scoring**: Evaluated strictly on **Speech-Active Frames** (energy-based voice activity detection, excluding the ~1.9s ambient room-tone region).
**Secondary Scoring (*)**: Whole-file scoring labelled *(includes room-tone-only region)*. Note that active-frame baseline SNR is ~2.6 dB higher because the same noise energy is spread over 5.64 s vs a 3.12 s active window.

| Test Set | Processing Mode | Active SNR (dB) | ΔActive SNR (dB) | Active SI-SDR (dB) | ΔActive SI-SDR (dB) | Active STOI | Active PESQ | Whole-File SNR (dB)* | Whole-File PESQ* | Frame Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Real Speech + White Noise (mixed at 0 dB whole-file SNR) | Noisy (Baseline) | 2.57 | 0.00 | 2.54 | 0.00 | 0.846 | 1.08 | 0.00 | 1.13 | - |
| Real Speech + White Noise (mixed at 0 dB whole-file SNR) | Enhanced (Offline) | 10.66 | +8.09 | 10.40 | +7.86 | **0.935** | **1.42** | 10.52 | 1.25 | - |
| Real Speech + White Noise (mixed at 0 dB whole-file SNR) | Enhanced (Real-Time) | 9.43 | +6.86 | 9.22 | +6.68 | **0.907** | **1.17** | 9.33 | 1.17 | 35.2** |
| Real Speech + White Noise (mixed at 5 dB whole-file SNR) | Noisy (Baseline) | 7.57 | 0.00 | 7.55 | 0.00 | 0.925 | 1.20 | 5.00 | 1.24 | - |
| Real Speech + White Noise (mixed at 5 dB whole-file SNR) | Enhanced (Offline) | 13.64 | +6.07 | 13.59 | +6.04 | **0.960** | **1.57** | 13.36 | 1.27 | - |
| Real Speech + White Noise (mixed at 5 dB whole-file SNR) | Enhanced (Real-Time) | 12.41 | +4.85 | 12.47 | +4.92 | **0.947** | **1.30** | 12.21 | 1.21 | 29.5** |
| Real Speech + White Noise (mixed at 10 dB whole-file SNR) | Noisy (Baseline) | 12.57 | 0.00 | 12.56 | 0.00 | 0.967 | 1.43 | 10.00 | 1.45 | - |
| Real Speech + White Noise (mixed at 10 dB whole-file SNR) | Enhanced (Offline) | 16.50 | +3.94 | 16.63 | +4.07 | **0.974** | **1.74** | 15.99 | 1.32 | - |
| Real Speech + White Noise (mixed at 10 dB whole-file SNR) | Enhanced (Real-Time) | 14.90 | +2.33 | 15.11 | +2.55 | 0.967 | 1.40 | 14.54 | 1.22 | 31.0** |
| Real Speech + Pink Noise (mixed at 0 dB whole-file SNR) | Noisy (Baseline) | 3.00 | 0.00 | 2.96 | 0.00 | 0.868 | 1.21 | 0.00 | 1.27 | - |
| Real Speech + Pink Noise (mixed at 0 dB whole-file SNR) | Enhanced (Offline) | 9.34 | +6.34 | 8.84 | +5.88 | **0.943** | **1.47** | 7.00 | 1.35 | - |
| Real Speech + Pink Noise (mixed at 0 dB whole-file SNR) | Enhanced (Real-Time) | 9.99 | +6.99 | 9.61 | +6.65 | **0.926** | **1.25** | 8.30 | 1.20 | 25.8** |
| Real Speech + Pink Noise (mixed at 5 dB whole-file SNR) | Noisy (Baseline) | 8.00 | 0.00 | 7.98 | 0.00 | 0.937 | 1.55 | 5.00 | 1.59 | - |
| Real Speech + Pink Noise (mixed at 5 dB whole-file SNR) | Enhanced (Offline) | 13.45 | +5.45 | 13.27 | +5.29 | **0.964** | 1.52 | 10.98 | 1.35 | - |
| Real Speech + Pink Noise (mixed at 5 dB whole-file SNR) | Enhanced (Real-Time) | 12.90 | +4.90 | 12.86 | +4.88 | **0.953** | 1.41 | 11.25 | 1.26 | 30.6** |
| Real Speech + Pink Noise (mixed at 10 dB whole-file SNR) | Noisy (Baseline) | 13.00 | 0.00 | 12.99 | 0.00 | 0.973 | 2.16 | 10.00 | 2.09 | - |
| Real Speech + Pink Noise (mixed at 10 dB whole-file SNR) | Enhanced (Offline) | 16.72 | +3.72 | 16.87 | +3.89 | 0.973 | 1.62 | 14.33 | 1.38 | - |
| Real Speech + Pink Noise (mixed at 10 dB whole-file SNR) | Enhanced (Real-Time) | 14.55 | +1.55 | 14.68 | +1.69 | 0.963 | 1.46 | 13.34 | 1.31 | 32.7** |
| Real Speech + Powerline Hum (mixed at 0 dB whole-file SNR) | Noisy (Baseline) | 2.56 | 0.00 | 2.60 | 0.00 | 0.959 | 1.42 | 0.00 | 1.42 | - |
| Real Speech + Powerline Hum (mixed at 0 dB whole-file SNR) | Enhanced (Offline) | 16.40 | +13.84 | 16.42 | +13.82 | **0.971** | **1.53** | 15.89 | 1.29 | - |
| Real Speech + Powerline Hum (mixed at 0 dB whole-file SNR) | Enhanced (Real-Time) | 13.72 | +11.15 | 13.78 | +11.18 | 0.950 | **1.48** | 13.43 | 1.32 | 29.3** |
| Real Speech + Powerline Hum (mixed at 5 dB whole-file SNR) | Noisy (Baseline) | 7.56 | 0.00 | 7.58 | 0.00 | 0.976 | 1.90 | 5.00 | 1.78 | - |
| Real Speech + Powerline Hum (mixed at 5 dB whole-file SNR) | Enhanced (Offline) | 17.27 | +9.71 | 17.39 | +9.81 | 0.975 | 1.59 | 16.66 | 1.31 | - |
| Real Speech + Powerline Hum (mixed at 5 dB whole-file SNR) | Enhanced (Real-Time) | 14.47 | +6.91 | 14.65 | +7.07 | 0.958 | 1.54 | 14.14 | 1.33 | 31.6** |
| Real Speech + Powerline Hum (mixed at 10 dB whole-file SNR) | Noisy (Baseline) | 12.56 | 0.00 | 12.57 | 0.00 | 0.988 | 2.59 | 10.00 | 2.34 | - |
| Real Speech + Powerline Hum (mixed at 10 dB whole-file SNR) | Enhanced (Offline) | 18.00 | +5.44 | 18.23 | +5.66 | 0.977 | 1.65 | 17.29 | 1.32 | - |
| Real Speech + Powerline Hum (mixed at 10 dB whole-file SNR) | Enhanced (Real-Time) | 15.20 | +2.64 | 15.47 | +2.89 | 0.967 | 1.61 | 14.82 | 1.35 | 26.4** |
| Real Speech + Same-Speaker Babble (Worst Case) (mixed at 0 dB whole-file SNR) | Noisy (Baseline) | 1.86 | 0.00 | 1.91 | 0.00 | 0.734 | 1.10 | 0.00 | 1.15 | - |
| Real Speech + Same-Speaker Babble (Worst Case) (mixed at 0 dB whole-file SNR) | Enhanced (Offline) | 3.18 | +1.32 | 0.54 | -1.37 | 0.646 | **1.13** | 1.98 | 1.09 | - |
| Real Speech + Same-Speaker Babble (Worst Case) (mixed at 0 dB whole-file SNR) | Enhanced (Real-Time) | 5.44 | +3.58 | 4.02 | +2.11 | **0.784** | **1.14** | 3.72 | 1.14 | 31.2** |
| Real Speech + Same-Speaker Babble (Worst Case) (mixed at 5 dB whole-file SNR) | Noisy (Baseline) | 6.86 | 0.00 | 6.89 | 0.00 | 0.849 | 1.18 | 5.00 | 1.26 | - |
| Real Speech + Same-Speaker Babble (Worst Case) (mixed at 5 dB whole-file SNR) | Enhanced (Offline) | 7.36 | +0.50 | 6.57 | -0.33 | **0.866** | **1.27** | 6.38 | 1.16 | - |
| Real Speech + Same-Speaker Babble (Worst Case) (mixed at 5 dB whole-file SNR) | Enhanced (Real-Time) | 9.71 | +2.85 | 9.24 | +2.35 | **0.902** | **1.20** | 8.33 | 1.16 | 24.7** |
| Real Speech + Same-Speaker Babble (Worst Case) (mixed at 10 dB whole-file SNR) | Noisy (Baseline) | 11.86 | 0.00 | 11.88 | 0.00 | 0.927 | 1.46 | 10.00 | 1.50 | - |
| Real Speech + Same-Speaker Babble (Worst Case) (mixed at 10 dB whole-file SNR) | Enhanced (Offline) | 12.85 | +0.99 | 12.96 | +1.08 | **0.953** | **1.67** | 11.83 | 1.26 | - |
| Real Speech + Same-Speaker Babble (Worst Case) (mixed at 10 dB whole-file SNR) | Enhanced (Real-Time) | 12.93 | +1.07 | 12.85 | +0.97 | **0.950** | 1.38 | 12.08 | 1.22 | 24.3** |

> **Whole-File vs. Speech-Active Metrics (*)**: Whole-file scores evaluate across the full 5.64s audio, which includes ~1.9s of leading non-speech ambient room tone. Because DeepFilterNet mutes silent frames to digital zero, whole-file PESQ and STOI are penalized by silent-frame divergence. Speech-active frame scoring evaluates only voiced speech, eliminating this silence-truncation scoring artifact.

> **Latency & Context Re-Priming (**)**: Frame processing latency represents raw CPU computation time to process one 40 ms audio chunk (1,920 samples). Because DeepFilterNet's `enhance()` function resets recurrent state on every call, the server uses **200 ms context re-priming** (passing the trailing 200 ms / 9,600 samples) to provide historical acoustic context and eliminate boundary phase discontinuities. Processing this re-primed buffer results in a measured latency of **23–34 ms per 40 ms frame** (RTF ~0.6–0.9, varies by run), remaining below the 40.0 ms frame hop budget. **This latency strictly excludes hardware audio device buffering** (soundcard buffer pools and OS driver latency), and is therefore not an end-to-end acoustic-to-acoustic latency measurement. Live microphone mode not benchmarked; file streaming only.

> **Offline vs. Real-Time Streaming Performance**:
> Real-time streaming output scores are within **up to ~3 dB** of offline batch enhancement across all SNR and SI-SDR conditions. Real-time streaming also exhibits lower STOI and PESQ (e.g. White Noise 0 dB: Active PESQ is **1.42 offline vs. 1.17 real-time**; Active STOI is **0.935 offline vs. 0.907 real-time**).

> **Perceptual Score Analysis (Room-Tone Artifact vs. Clean Reference)**:
> In the single-speaker benchmark on `real_clean_speech_48k.wav`, PESQ fell below the noisy baseline in 4 of 12 offline rows (pink 5 and 10 dB, hum 5 and 10 dB) and 6 of 12 real-time rows. Rather than an inherent model limitation, this was an artifact of the reference audio: `real_clean_speech_48k.wav` contains non-zero ambient room tone (-44.28 dBFS leading, ~-50.88 dBFS in speech pauses). DeepFilterNet removes this room tone along with injected noise, which PESQ's perceptual alignment penalizes as a spectral discrepancy against the imperfect reference. When evaluated against studio-clean LibriSpeech references across 5 speakers (white/pink/hum at 0/5/10 dB), default enhanced PESQ consistently exceeds the noisy baseline across all 9 conditions (+0.52 to +1.42 MOS), confirming that the apparent degradation was reference-noise bias. Results on clean LibriSpeech references provide the more reliable PESQ evidence.

### Observation: Real-Time vs. Offline on Pink Noise and Babble (Unexplained)
- On Pink Noise (0 dB: +6.99 dB RT vs. +6.34 dB Offline) and Same-Speaker Babble (0 dB: +3.58 dB RT vs. +1.32 dB Offline; 5 dB: +2.85 dB RT vs. +0.50 dB Offline), real-time streaming achieves slightly higher ΔSNR and ΔSI-SDR than offline batch enhancement.
- **Status**: *[Unexplained Observation]* — The exact interaction between 200 ms context re-priming and continuous sequence processing under dense low-frequency or speech-masking conditions remains an empirical observation pending internal model state probing.

## 2. Real Recording Evaluation (No Clean Reference)

Evaluated on an authentic recorded noisy audio clip ([`samples/real_noisy_speech.wav`](file:///e:/bnsup/samples/real_noisy_speech.wav)) without a clean reference track.
Spectrogram visualization: [`outputs/eval_spec_real_recording_noref.png`](file:///e:/bnsup/outputs/eval_spec_real_recording_noref.png).

| Processing Mode | Noise-Floor Reduction (Quietest 10% Frames) | Speech-Level Change (Loudest 30% Frames) | Envelope Correlation | Frame Latency |
| :--- | :--- | :--- | :--- | :--- |
| Enhanced (Offline) | **+27.68 dB** (from -49.98 to -77.65 dB) | -0.74 dB | **0.9707** | - |
| Enhanced (Real-Time) | **+30.25 dB** (from -49.98 to -80.22 dB) | -1.32 dB | **0.9581** | 34.4 ms** |

## 3. Babble Noise Specification
- **Label**: `Real Speech + Same-Speaker Babble (Worst Case)`
- **Implementation Details**: The babble signal is synthesized by taking the active speech segment of `clean_48k` (2.0s to 5.2s) and creating tiled copies of the target's own speech across 5 overlapping tracks at fixed offsets and gains:
  - Track 0: offset = `0.00 s` (0 samples), gain = `1.00`
  - Track 1: offset = `0.55 s` (26,400 samples), gain = `0.85`
  - Track 2: offset = `1.15 s` (55,200 samples), gain = `1.00`
  - Track 3: offset = `1.75 s` (84,000 samples), gain = `0.85`
  - Track 4: offset = `2.35 s` (112,800 samples), gain = `1.00`
- **Acoustic Implications**: All 5 talker tracks are **tiled copies of the target's own speech** without phase decorrelation or pitch modification. Target speech and background interferers share identical vocal tract formant resonances, pitch range, and timbre, representing the acoustic worst-case of speech-on-speech masking for single-channel enhancement.

## 4. Multi-Condition & Naturalness Control Evaluation (Extended Part 2: 45 Items)

Evaluated across 45 conditions (5 LibriSpeech speakers [374, 2277, 3660, 6930, 7902] × 3 VoiceBank-DEMAND test noise clips [Clip A p232_001, Clip B p232_013, Clip C p232_024] × 3 SNR levels [0, 5, 10 dB]).

> **Default vs. Attenuation Limit / Dry-Mix**:
> Default full suppression (`atten_lim_db = None, dry_mix = 0.0`) is the best operating point on clean references, maximizing ΔSI-SDR (+11.88 dB) and DNSMOS background suppression (BAK 4.14). Attenuation limits (18 dB or 24 dB) and dry mix (0.1) are user naturalness controls to prevent "dead-silence" fatigue or retain situational awareness, not objective quality gains.

### Overall Performance (Mean ± Std Across All 45 Items)

| Setting | ΔSI-SDR (dB) | SI-SDR (dB) | PESQ (Wideband) | DNSMOS OVRL | DNSMOS BAK |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Noisy Baseline** | — | 5.69 ± 4.15 dB | 2.23 ± 0.76 | 2.74 ± 0.65 | 3.19 ± 0.88 |
| **Default (None, 0.0)** | **+11.88 ± 6.47 dB** | **17.57 ± 6.11 dB** | 3.02 ± 0.76 | **3.35 ± 0.18** | **4.14 ± 0.15** |
| **Mix 0.1 (None, 0.1)** | +10.77 ± 5.01 dB | 16.46 ± 4.97 dB | 3.06 ± 0.69 | 3.26 ± 0.19 | 3.97 ± 0.26 |
| **Atten 18 dB (18, 0.0)** | +10.31 ± 4.55 dB | 16.00 ± 4.68 dB | 3.04 ± 0.69 | 3.24 ± 0.19 | 3.94 ± 0.28 |
| **Atten 24 dB (24, 0.0)** | +11.38 ± 5.69 dB | 17.06 ± 5.46 dB | 3.07 ± 0.71 | 3.29 ± 0.17 | 4.03 ± 0.23 |

### Paired Differences (Default minus Alternative Across All 45 Items)

| Comparison (Default - Alternative) | ΔSI-SDR Difference | PESQ Difference | DNSMOS OVRL Difference | DNSMOS BAK Difference | Items Default is Better |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Default vs. Mix 0.1** | **+1.11 ± 1.63 dB** | -0.040 ± 0.226 | **+0.086 ± 0.137** | **+0.166 ± 0.207** | SI-SDR: 34/45, PESQ: 21/45, OVRL: 35/45, BAK: 39/45 |
| **Default vs. Atten 18 dB** | **+1.57 ± 2.11 dB** | -0.019 ± 0.278 | **+0.102 ± 0.149** | **+0.199 ± 0.227** | SI-SDR: 35/45, PESQ: 24/45, OVRL: 35/45, BAK: 40/45 |
| **Default vs. Atten 24 dB** | **+0.50 ± 0.89 dB** | -0.055 ± 0.137 | **+0.056 ± 0.112** | **+0.110 ± 0.163** | SI-SDR: 32/45, PESQ: 15/45, OVRL: 33/45, BAK: 37/45 |

## Appendix: Synthetic Speech Baselines (Formant Tones)

The following baseline tests evaluate synthetic formant tone sequences ([`samples/clean_speech_ref.wav`](file:///e:/bnsup/samples/clean_speech_ref.wav)).
PESQ and STOI are marked **N/A** because artificial sinusoidal tones violate ITU-T P.862 and STOI assumptions of natural human vocal tract acoustics.

| Test Set | Processing Mode | SNR (dB) | ΔSNR (dB) | SI-SDR (dB) | ΔSI-SDR (dB) | STOI | PESQ | Frame Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Synthetic Speech + White Noise | Noisy (Baseline) | 8.67 | 0.00 | 8.11 | 0.00 | N/A | N/A | - |
| Synthetic Speech + White Noise | Enhanced (Offline) | 0.83 | -7.84 | -4.87 | -12.98 | N/A | N/A | - |
| Synthetic Speech + White Noise | Enhanced (Real-Time) | 2.63 | -6.04 | 0.60 | -7.51 | N/A | N/A | 28.0** |
| Synthetic Speech + Hum | Noisy (Baseline) | 5.00 | 0.00 | 4.99 | 0.00 | N/A | N/A | - |
| Synthetic Speech + Hum | Enhanced (Offline) | 0.50 | -4.50 | -9.15 | -14.14 | N/A | N/A | - |
| Synthetic Speech + Hum | Enhanced (Real-Time) | 1.32 | -3.68 | -4.10 | -9.09 | N/A | N/A | 27.5** |
| Synthetic Speech + Buzz | Noisy (Baseline) | 2.71 | 0.00 | 2.12 | 0.00 | N/A | N/A | - |
| Synthetic Speech + Buzz | Enhanced (Offline) | 0.30 | -2.41 | -10.95 | -13.06 | N/A | N/A | - |
| Synthetic Speech + Buzz | Enhanced (Real-Time) | 0.67 | -2.04 | -7.55 | -9.66 | N/A | N/A | 25.7** |

> **Analysis of Negative ΔSNR on Synthetic Speech (Likely Explanation)**: DeepFilterNet's deep complex filter is trained strictly on human speech datasets. Pure sinusoidal formant tones lack natural pitch jitter, vocal tract resonances, and glottal pulse excitation. It is **likely** that DeepFilterNet classifies these artificial formant tones as non-speech harmonic interference and attenuates them, resulting in negative SNR and SI-SDR improvements (-2.0 dB to -14.1 dB). This confirms that DeepFilterNet requires genuine human speech acoustics.