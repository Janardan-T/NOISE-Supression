# Broader Speech Enhancement Evaluation Report
## Test Dataset Composition
- **Clean Speech**: 5 diverse speakers from LibriSpeech (`6930`, `7902`, `2277`, `3660`, `374`).
- **Real Acoustic Noise**: 3 distinct environments from DEMAND test set (`TMETRO` metro, `CAFE` cafe, `TBUS` bus).
- **Operating Point**: Evaluated at 5 dB input SNR.
- **Comparison**: Default Settings (`atten=None, mix=0.0`) vs. Tuned Setting (`atten=None dB, mix=0.1`).

## Statistical Distribution (Mean & Spread)

| Metric | Noisy Baseline | Default DeepFilterNet3 | Tuned Setting |
| :--- | :--- | :--- | :--- |
| **ΔSI-SDR (dB)** | - | 12.13 ± 6.07 [1.00, 21.50] | 11.06 ± 4.65 [2.01, 17.54] |
| **SI-SDR (dB)** | 5.69 ± 0.74 | 17.81 ± 5.99 [6.24, 26.72] | 16.75 ± 4.62 [7.25, 22.76] |
| **STOI** | 0.97 ± 0.04 | 0.98 ± 0.03 [0.90, 1.00] | 0.98 ± 0.02 [0.93, 1.00] |
| **PESQ (MOS-WB)** | 2.21 ± 0.65 | 3.06 ± 0.71 [1.42, 3.82] | 3.08 ± 0.61 [1.74, 3.90] |
| **DNSMOS Overall** | 2.80 ± 0.52 | 3.35 ± 0.16 [2.79, 3.51] | 3.27 ± 0.16 [3.01, 3.52] |
| **DNSMOS Signal Quality (SIG)** | 3.43 ± 0.34 | 3.61 ± 0.09 [3.33, 3.74] | 3.61 ± 0.06 [3.47, 3.72] |
| **DNSMOS Background Noise (BAK)** | 3.24 ± 0.76 | 4.13 ± 0.17 [3.51, 4.23] | 3.98 ± 0.23 [3.53, 4.22] |

---

## Per-Condition Evaluation Table

| Speaker | Real Noise Type | Baseline SI-SDR | Baseline PESQ | Baseline DNSMOS | Def ΔSI-SDR | Def PESQ | Def DNSMOS | Tuned ΔSI-SDR | Tuned PESQ | Tuned DNSMOS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Speaker_374 | DEMAND_TMETRO | 5.21 dB | 2.71 | 2.94 | **+19.20 dB** | 3.44 | 3.44 | **+16.46 dB** | 3.43 | 3.43 |
| Speaker_374 | DEMAND_CAFE | 5.13 dB | 1.37 | 1.90 | **+11.47 dB** | 2.77 | 3.33 | **+11.12 dB** | 2.62 | 3.01 |
| Speaker_374 | DEMAND_TBUS | 5.12 dB | 2.36 | 2.92 | **+21.29 dB** | 3.82 | 3.45 | **+17.43 dB** | 3.72 | 3.44 |
| Speaker_2277 | DEMAND_TMETRO | 5.50 dB | 2.98 | 2.97 | **+19.68 dB** | 3.38 | 3.39 | **+16.85 dB** | 3.56 | 3.32 |
| Speaker_2277 | DEMAND_CAFE | 5.52 dB | 1.45 | 1.72 | **+10.52 dB** | 2.90 | 3.28 | **+10.15 dB** | 2.86 | 3.10 |
| Speaker_2277 | DEMAND_TBUS | 5.23 dB | 2.59 | 2.94 | **+21.50 dB** | 3.80 | 3.51 | **+17.54 dB** | 3.90 | 3.15 |
| Speaker_3660 | DEMAND_TMETRO | 5.98 dB | 2.88 | 3.24 | **+14.25 dB** | 3.47 | 3.38 | **+13.29 dB** | 3.53 | 3.38 |
| Speaker_3660 | DEMAND_CAFE | 5.39 dB | 1.38 | 2.31 | **+8.69 dB** | 2.52 | 3.35 | **+8.45 dB** | 2.39 | 3.11 |
| Speaker_3660 | DEMAND_TBUS | 5.55 dB | 2.60 | 3.12 | **+13.21 dB** | 3.38 | 3.40 | **+11.92 dB** | 3.47 | 3.33 |
| Speaker_6930 | DEMAND_TMETRO | 5.58 dB | 2.41 | 3.35 | **+8.38 dB** | 3.11 | 3.46 | **+7.75 dB** | 3.14 | 3.52 |
| Speaker_6930 | DEMAND_CAFE | 5.24 dB | 1.22 | 2.64 | **+1.00 dB** | 1.42 | 3.28 | **+2.01 dB** | 1.74 | 3.20 |
| Speaker_6930 | DEMAND_TBUS | 5.08 dB | 1.94 | 3.49 | **+4.74 dB** | 1.83 | 3.41 | **+5.37 dB** | 2.41 | 3.42 |
| Speaker_7902 | DEMAND_TMETRO | 8.00 dB | 3.39 | 3.25 | **+10.93 dB** | 3.77 | 3.39 | **+10.88 dB** | 3.70 | 3.36 |
| Speaker_7902 | DEMAND_CAFE | 6.56 dB | 1.63 | 2.14 | **+4.42 dB** | 2.50 | 2.79 | **+4.76 dB** | 2.50 | 3.02 |
| Speaker_7902 | DEMAND_TBUS | 6.20 dB | 2.21 | 3.03 | **+12.60 dB** | 3.72 | 3.43 | **+11.95 dB** | 3.31 | 3.35 |
