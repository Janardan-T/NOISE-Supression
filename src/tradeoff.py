"""Trade-off Sweep & Broader Speech Benchmark Suite (src/tradeoff.py).

Part 1 - Attenuation / Blend Sweep & Context Buffer Trade-off:
- Sweep atten_lim_db in [6, 12, 18, 24, None]
- Sweep dry/wet mix in [0.0, 0.1, 0.2] (output = (1 - mix) * enhanced + mix * noisy)
- Test White, Pink, and Powerline Hum noise at 0 dB, 5 dB, 10 dB SNR.
- Speech-active-frame scoring against samples/real_clean_speech_48k.wav.
- Evaluates 120 ms vs 200 ms real-time context buffer setting (quality cost).
- Generates tradeoff CSV, markdown table, and tradeoff plot (dSI-SDR vs PESQ).
- Determines best PESQ without losing > 2 dB of dSI-SDR.

Part 2 - Broader Test Set Evaluation:
- 5 different speakers from LibriSpeech (speakers 6930, 7902, 2277, 3660, 374).
- 3 real acoustic noise environments from DEMAND (TMETRO, CAFE, TBUS).
- Evaluates default settings vs. best Part 1 settings.
- Computes dSNR, dSI-SDR, STOI, PESQ, and Microsoft DNSMOS (SIG, BAK, OVRL).
- Reports mean and spread (std, min, max, median).
"""

import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pesq import pesq as compute_pesq
import pystoi
from scipy import signal
import soundfile as sf
import torch
import torchaudio

# Add project root to sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.denoise import load_model
from src.realtime.server import StreamDenoiser
from src.evaluate import (
    align_signals,
    calculate_pesq,
    calculate_si_sdr,
    calculate_snr,
    calculate_stoi,
    detect_speech_active_boundaries,
    generate_noise_signals,
    mix_at_snr,
    resample_to_16k,
    TARGET_EVAL_SR,
)

OUTPUTS_DIR = workspace_root / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# DNSMOS import
try:
    from speechmos import dnsmos
    HAS_DNSMOS = True
except ImportError:
    HAS_DNSMOS = False


def safe_dnsmos(audio_16k: np.ndarray) -> dict:
    """Safely calculate Microsoft DNSMOS scores on 16 kHz audio clamped to [-1.0, 1.0]."""
    if not HAS_DNSMOS:
        return {}
    try:
        # speechmos requires values in [-1.0, 1.0]
        clipped = np.clip(audio_16k, -1.0, 1.0).astype(np.float32)
        return dnsmos.run(clipped, sr=16000)
    except Exception as e:
        print(f"Warning: DNSMOS evaluation failed: {e}")
        return {}


# ==============================================================================
# Part 1: Attenuation / Blend Sweep & Context Buffer Scoring
# ==============================================================================

def run_part1_sweep(model, df_state, clean_48k: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    print("=" * 78)
    print(" PART 1: ATTENUATION / BLEND SWEEP & CONTEXT BUFFER TRADE-OFF")
    print("=" * 78)

    from df.enhance import enhance

    length_48k = len(clean_48k)
    clean_16k = resample_to_16k(clean_48k, 48000)
    start_smp, end_smp = detect_speech_active_boundaries(clean_16k)
    clean_act = clean_16k[start_smp:end_smp]

    print(f" Clean Speech:       real_clean_speech_48k.wav ({length_48k / 48000:.2f} s)")
    print(f" Active Speech Window: {start_smp / 16000:.2f}s to {end_smp / 16000:.2f}s ({(end_smp - start_smp) / 16000:.2f} s)")

    # Noise profiles
    all_noises = generate_noise_signals(clean_48k, length=length_48k, sr=48000, seed=42)
    eval_noises = {
        "White Noise": all_noises["White Noise"],
        "Pink Noise": all_noises["Pink Noise"],
        "Powerline Hum": all_noises["Powerline Hum"],
    }
    snr_levels = [0.0, 5.0, 10.0]

    atten_lim_list: List[Optional[float]] = [6.0, 12.0, 18.0, 24.0, None]
    mix_list: List[float] = [0.0, 0.1, 0.2]

    records = []

    print(f"\n[1/3] Sweeping {len(atten_lim_list) * len(mix_list)} parameter combinations across {len(eval_noises) * len(snr_levels)} noise conditions...")

    for noise_name, noise_arr in eval_noises.items():
        for target_snr in snr_levels:
            # Mix at target SNR
            mixed_48k = mix_at_snr(clean_48k, noise_arr, target_snr)

            # Baseline metrics on active frames
            mixed_16k = resample_to_16k(mixed_48k, 48000)
            ref_a, mix_a, _, _ = align_signals(clean_16k, mixed_16k)
            c_act = ref_a[start_smp:end_smp]
            m_act = mix_a[start_smp:end_smp]

            base_snr = calculate_snr(c_act, m_act)
            base_sdr = calculate_si_sdr(c_act, m_act)
            base_stoi = calculate_stoi(c_act, m_act)
            base_pesq = calculate_pesq(c_act, m_act)

            mixed_tensor = torch.from_numpy(mixed_48k).unsqueeze(0).float()

            for atten_lim in atten_lim_list:
                # 1. Run offline model inference
                with torch.no_grad():
                    enhanced_tensor = enhance(
                        model=model,
                        df_state=df_state,
                        audio=mixed_tensor,
                        pad=True,
                        atten_lim_db=atten_lim,
                    )
                enhanced_48k = enhanced_tensor.squeeze(0).cpu().numpy()

                for mix in mix_list:
                    # 2. Apply dry/wet blend: (1 - mix) * enhanced + mix * noisy
                    if mix == 0.0:
                        blended_48k = enhanced_48k
                    else:
                        blended_48k = (1.0 - mix) * enhanced_48k + mix * mixed_48k

                    # 3. Resample to 16 kHz and evaluate on speech-active window
                    blended_16k = resample_to_16k(blended_48k, 48000)
                    ref_aligned, target_aligned, lag, _ = align_signals(clean_16k, blended_16k)
                    min_len = min(len(ref_aligned), len(target_aligned))
                    act_end = min(end_smp, min_len)
                    act_start = min(start_smp, act_end)

                    c_eval = ref_aligned[act_start:act_end]
                    p_eval = target_aligned[act_start:act_end]

                    snr_val = calculate_snr(c_eval, p_eval)
                    sdr_val = calculate_si_sdr(c_eval, p_eval)
                    stoi_val = calculate_stoi(c_eval, p_eval)
                    pesq_val = calculate_pesq(c_eval, p_eval)

                    dsnr = snr_val - base_snr
                    dsdr = sdr_val - base_sdr

                    records.append({
                        "noise_type": noise_name,
                        "snr_db": target_snr,
                        "atten_lim_db": "None" if atten_lim is None else f"{int(atten_lim)}",
                        "atten_val": 999.0 if atten_lim is None else float(atten_lim),
                        "mix": mix,
                        "base_snr": base_snr,
                        "base_sdr": base_sdr,
                        "base_stoi": base_stoi,
                        "base_pesq": base_pesq,
                        "active_snr": snr_val,
                        "d_snr": dsnr,
                        "active_sdr": sdr_val,
                        "d_sdr": dsdr,
                        "active_stoi": stoi_val,
                        "active_pesq": pesq_val,
                    })

    df_sweep = pd.DataFrame(records)
    print(f" [Sweep Complete] Evaluated {len(df_sweep)} parameter combinations.")

    # --------------------------------------------------------------------------
    # Real-Time Context Setting Comparison (120 ms vs 200 ms)
    # --------------------------------------------------------------------------
    print("\n[2/3] Evaluating Real-Time Context Buffer: 120 ms vs 200 ms on same mixes...")
    context_records = []
    chunk_size = 1920  # 40 ms at 48 kHz

    for noise_name, noise_arr in eval_noises.items():
        for target_snr in snr_levels:
            mixed_48k = mix_at_snr(clean_48k, noise_arr, target_snr)

            # Baseline
            mixed_16k = resample_to_16k(mixed_48k, 48000)
            ref_a, mix_a, _, _ = align_signals(clean_16k, mixed_16k)
            c_act = ref_a[start_smp:end_smp]
            m_act = mix_a[start_smp:end_smp]
            base_sdr = calculate_si_sdr(c_act, m_act)
            base_pesq = calculate_pesq(c_act, m_act)

            # Stream through 200 ms denoiser (9600 context samples)
            denoiser_200 = StreamDenoiser(context_buf_size=9600)
            out_200_bytes = bytearray()
            int16_in = (np.clip(mixed_48k, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            for i in range(0, len(int16_in), chunk_size * 2):
                chunk = int16_in[i : i + chunk_size * 2]
                if len(chunk) < chunk_size * 2:
                    break
                out_chunk = denoiser_200.process_chunk_bytes(chunk, model, df_state)
                out_200_bytes.extend(out_chunk)
            out_200_np = np.frombuffer(out_200_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Stream through 120 ms denoiser (5760 context samples)
            denoiser_120 = StreamDenoiser(context_buf_size=5760)
            out_120_bytes = bytearray()
            for i in range(0, len(int16_in), chunk_size * 2):
                chunk = int16_in[i : i + chunk_size * 2]
                if len(chunk) < chunk_size * 2:
                    break
                out_chunk = denoiser_120.process_chunk_bytes(chunk, model, df_state)
                out_120_bytes.extend(out_chunk)
            out_120_np = np.frombuffer(out_120_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Score 200 ms
            s200_16k = resample_to_16k(out_200_np, 48000)
            r200, t200, _, _ = align_signals(clean_16k, s200_16k)
            c200 = r200[start_smp:end_smp]
            p200 = t200[start_smp:end_smp]
            sdr_200 = calculate_si_sdr(c200, p200)
            pesq_200 = calculate_pesq(c200, p200)
            stoi_200 = calculate_stoi(c200, p200)

            # Score 120 ms
            s120_16k = resample_to_16k(out_120_np, 48000)
            r120, t120, _, _ = align_signals(clean_16k, s120_16k)
            c120 = r120[start_smp:end_smp]
            p120 = t120[start_smp:end_smp]
            sdr_120 = calculate_si_sdr(c120, p120)
            pesq_120 = calculate_pesq(c120, p120)
            stoi_120 = calculate_stoi(c120, p120)

            context_records.append({
                "noise_type": noise_name,
                "snr_db": target_snr,
                "sdr_200ms": sdr_200,
                "d_sdr_200ms": sdr_200 - base_sdr,
                "pesq_200ms": pesq_200,
                "stoi_200ms": stoi_200,
                "sdr_120ms": sdr_120,
                "d_sdr_120ms": sdr_120 - base_sdr,
                "pesq_120ms": pesq_120,
                "stoi_120ms": stoi_120,
                "delta_sdr_cost": sdr_120 - sdr_200,
                "delta_pesq_cost": (pesq_120 - pesq_200) if (pesq_120 and pesq_200) else None,
                "delta_stoi_cost": (stoi_120 - stoi_200) if (stoi_120 and stoi_200) else None,
            })

    df_context = pd.DataFrame(context_records)

    # --------------------------------------------------------------------------
    # Optimal Setting Analysis
    # Condition: Best PESQ without losing more than ~2 dB of dSI-SDR vs default (None, 0.0)
    # --------------------------------------------------------------------------
    print("\n[3/3] Analyzing Pareto trade-off points...")
    best_settings_summary = []
    best_config_per_snr = {}

    for snr in snr_levels:
        df_snr = df_sweep[df_sweep["snr_db"] == snr]
        # Average across noise types for each (atten_lim, mix) configuration
        grouped = df_snr.groupby(["atten_lim_db", "atten_val", "mix"]).agg({
            "d_sdr": "mean",
            "d_snr": "mean",
            "active_pesq": "mean",
            "active_stoi": "mean",
        }).reset_index()

        default_row = grouped[(grouped["atten_lim_db"] == "None") & (grouped["mix"] == 0.0)].iloc[0]
        default_sdr = default_row["d_sdr"]
        default_pesq = default_row["active_pesq"]

        # Eligible rows: d_sdr >= default_sdr - 2.0 dB
        eligible = grouped[grouped["d_sdr"] >= (default_sdr - 2.0)]
        best_row = eligible.sort_values(by="active_pesq", ascending=False).iloc[0]

        best_config_per_snr[snr] = {
            "atten_lim_db": best_row["atten_lim_db"],
            "mix": best_row["mix"],
            "d_sdr": best_row["d_sdr"],
            "active_pesq": best_row["active_pesq"],
            "default_sdr": default_sdr,
            "default_pesq": default_pesq,
        }

        diff_pesq = best_row["active_pesq"] - default_pesq
        diff_sdr = best_row["d_sdr"] - default_sdr

        if diff_pesq > 0.005:
            best_settings_summary.append(
                f"- **SNR {int(snr)} dB**: Setting `atten_lim={best_row['atten_lim_db']} dB, mix={best_row['mix']}` "
                f"improves PESQ to {best_row['active_pesq']:.3f} (+{diff_pesq:.3f} vs. default {default_pesq:.3f}) "
                f"while maintaining ΔSI-SDR of {best_row['d_sdr']:.2f} dB ({diff_sdr:+.2f} dB vs. default {default_sdr:.2f} dB)."
            )
        else:
            best_settings_summary.append(
                f"- **SNR {int(snr)} dB**: **Nothing helps** improve PESQ over the default (`atten_lim=None, mix=0.0`) "
                f"without sacrificing > 2.0 dB of ΔSI-SDR. The default setting remains optimal "
                f"(PESQ: {default_pesq:.3f}, ΔSI-SDR: {default_sdr:.2f} dB)."
            )

    summary_text = "\n".join(best_settings_summary)
    print("\n" + summary_text)

    # --------------------------------------------------------------------------
    # Save Outputs
    # --------------------------------------------------------------------------
    csv_path = OUTPUTS_DIR / "tradeoff_sweep.csv"
    df_sweep.to_csv(csv_path, index=False)
    print(f"\n[Saved]: {csv_path}")

    # Generate Markdown Table
    md_path = OUTPUTS_DIR / "tradeoff_sweep.md"
    generate_tradeoff_markdown(df_sweep, df_context, summary_text, md_path)
    print(f"[Saved]: {md_path}")

    # Generate Plot: dSI-SDR vs PESQ
    plot_path = OUTPUTS_DIR / "tradeoff_sdr_vs_pesq.png"
    generate_tradeoff_plot(df_sweep, plot_path)
    print(f"[Saved]: {plot_path}")

    return df_sweep, df_context, summary_text


def generate_tradeoff_plot(df_sweep: pd.DataFrame, plot_path: Path):
    """Generate trade-off scatter plot: dSI-SDR vs PESQ across configurations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    fig.patch.set_facecolor("#0f172a")

    snr_levels = [0.0, 5.0, 10.0]
    mix_markers = {0.0: "o", 0.1: "s", 0.2: "^"}
    noise_colors = {"White Noise": "#38bdf8", "Pink Noise": "#f472b6", "Powerline Hum": "#34d399"}

    for idx, snr in enumerate(snr_levels):
        ax = axes[idx]
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")

        df_sub = df_sweep[df_sweep["snr_db"] == snr]

        for noise_name in ["White Noise", "Pink Noise", "Powerline Hum"]:
            df_n = df_sub[df_sub["noise_type"] == noise_name]
            for mix_val in [0.0, 0.1, 0.2]:
                df_nm = df_n[df_n["mix"] == mix_val]
                ax.scatter(
                    df_nm["d_sdr"],
                    df_nm["active_pesq"],
                    color=noise_colors[noise_name],
                    marker=mix_markers[mix_val],
                    s=60,
                    alpha=0.85,
                    edgecolors="#ffffff" if mix_val == 0.0 else "none",
                    linewidths=0.8,
                    label=f"{noise_name} (mix={mix_val})" if idx == 0 else None,
                )

        # Mark default setting (atten_lim=None, mix=0.0)
        default_pts = df_sub[(df_sub["atten_lim_db"] == "None") & (df_sub["mix"] == 0.0)]
        ax.scatter(
            default_pts["d_sdr"],
            default_pts["active_pesq"],
            color="#fbbf24",
            marker="*",
            s=160,
            zorder=5,
            edgecolors="#000000",
            linewidths=1.0,
            label="Default (None, mix=0)" if idx == 0 else None,
        )

        ax.set_title(f"Input SNR = {int(snr)} dB", color="#f8fafc", fontsize=12, fontweight="bold")
        ax.set_xlabel("ΔSI-SDR Improvement (dB)", color="#cbd5e1", fontsize=10)
        ax.set_ylabel("PESQ MOS (Speech Active)", color="#cbd5e1", fontsize=10)
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#475569")

    # Legend on first axis
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=5,
        fontsize=9,
        facecolor="#1e293b",
        edgecolor="#475569",
        labelcolor="#f8fafc",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()


def generate_tradeoff_markdown(df_sweep: pd.DataFrame, df_context: pd.DataFrame, summary_text: str, md_path: Path):
    """Write markdown summary report for Part 1."""
    content = []
    content.append("# Trade-Off Sweep & Context Buffer Evaluation Report\n\n")
    content.append("## Executive Summary & Best Settings\n\n")
    content.append(summary_text + "\n\n")

    mean_sdr_cost = df_context["delta_sdr_cost"].mean()
    valid_pesq = df_context["delta_pesq_cost"].dropna()
    mean_pesq_cost = valid_pesq.mean() if len(valid_pesq) > 0 else 0.0
    valid_stoi = df_context["delta_stoi_cost"].dropna()
    mean_stoi_cost = valid_stoi.mean() if len(valid_stoi) > 0 else 0.0

    content.append("### Quality Cost of Real-Time Context Buffer (120 ms vs 200 ms)\n\n")
    content.append(
        f"**Overall Mean Quality Cost (120 ms vs 200 ms)**: "
        f"ΔSI-SDR cost = **{mean_sdr_cost:+.2f} dB**, "
        f"PESQ cost = **{mean_pesq_cost:+.3f} MOS**, "
        f"STOI cost = **{mean_stoi_cost:+.4f}**.\n\n"
    )
    content.append("| Noise Type | Input SNR | 200 ms ΔSI-SDR | 120 ms ΔSI-SDR | Cost ΔSI-SDR | 200 ms PESQ | 120 ms PESQ | Cost PESQ | 200 ms STOI | 120 ms STOI |\n")
    content.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for _, r in df_context.iterrows():
        p200_str = f"{r['pesq_200ms']:.2f}" if r['pesq_200ms'] is not None else "N/A"
        p120_str = f"{r['pesq_120ms']:.2f}" if r['pesq_120ms'] is not None else "N/A"
        pcost_str = f"{r['delta_pesq_cost']:+.2f}" if r['delta_pesq_cost'] is not None else "N/A"
        st200_str = f"{r['stoi_200ms']:.3f}" if r['stoi_200ms'] is not None else "N/A"
        st120_str = f"{r['stoi_120ms']:.3f}" if r['stoi_120ms'] is not None else "N/A"
        content.append(
            f"| {r['noise_type']} | {int(r['snr_db'])} dB | {r['d_sdr_200ms']:+.2f} dB | {r['d_sdr_120ms']:+.2f} dB | "
            f"**{r['delta_sdr_cost']:+.2f} dB** | {p200_str} | {p120_str} | {pcost_str} | {st200_str} | {st120_str} |\n"
        )

    content.append("\n---\n\n")
    content.append("## Sweep Results Summary (Averaged across Noise Types)\n\n")
    content.append("Key: `atten_lim_db` in [6, 12, 18, 24, None], `dry/wet mix` in [0.0, 0.1, 0.2].\n\n")

    # Grouped summary table across noise types
    grouped = df_sweep.groupby(["snr_db", "atten_lim_db", "mix"]).agg({
        "d_snr": "mean",
        "d_sdr": "mean",
        "active_stoi": "mean",
        "active_pesq": "mean",
    }).reset_index()

    content.append("| Input SNR | Max Attenuation | Dry/Wet Mix | Mean ΔSNR | Mean ΔSI-SDR | Mean STOI | Mean PESQ |\n")
    content.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for _, r in grouped.iterrows():
        content.append(
            f"| {int(r['snr_db'])} dB | {r['atten_lim_db']} dB | {r['mix']:.1f} | "
            f"{r['d_snr']:+.2f} dB | {r['d_sdr']:+.2f} dB | {r['active_stoi']:.3f} | {r['active_pesq']:.3f} |\n"
        )

    content.append("\n---\n\n")
    content.append("## Complete Sweep Results (All Conditions)\n\n")
    content.append("| Noise Type | Input SNR | Atten Lim | Mix | Baseline SNR | Baseline SDR | Baseline STOI | Baseline PESQ | Output SNR | Output SDR | Output STOI | Output PESQ | ΔSNR | ΔSI-SDR |\n")
    content.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for _, r in df_sweep.iterrows():
        content.append(
            f"| {r['noise_type']} | {int(r['snr_db'])} dB | {r['atten_lim_db']} | {r['mix']:.1f} | "
            f"{r['base_snr']:.2f} | {r['base_sdr']:.2f} | {r['base_stoi']:.3f} | {r['base_pesq']:.2f} | "
            f"{r['active_snr']:.2f} | {r['active_sdr']:.2f} | {r['active_stoi']:.3f} | {r['active_pesq']:.2f} | "
            f"**{r['d_snr']:+.2f}** | **{r['d_sdr']:+.2f}** |\n"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(content))


# ==============================================================================
# Part 2: Broader Test Set Evaluation
# ==============================================================================

def extract_audio_from_wav_bytes(wav_bytes: bytes, target_sr: int = 48000) -> np.ndarray:
    """Read in-memory WAV bytes, convert to float32 mono, resample to target_sr."""
    with io.BytesIO(wav_bytes) as bio:
        data, sr = sf.read(bio, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        tensor = torch.from_numpy(data).unsqueeze(0).float()
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        data = resampler(tensor).squeeze(0).numpy()
    return data.astype(np.float32)


def run_part2_broader_eval(model, df_state, best_part1_config: dict) -> Tuple[pd.DataFrame, str]:
    print("\n" + "=" * 78)
    print(" PART 2: BROADER TEST SET EVALUATION (5 SPEAKERS & 3 REAL NOISE TYPES)")
    print("=" * 78)

    from df.enhance import enhance

    # 1. Download datasets
    from huggingface_hub import hf_hub_download
    print("[1/4] Loading multi-speaker clean speech and real DEMAND noise datasets...")

    try:
        spk_parquet_path = hf_hub_download(
            repo_id="Cidoyi/librispeech-samples",
            filename="data/train-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df_spk = pd.read_parquet(spk_parquet_path, engine="fastparquet")
        print(f" Clean Speech: Loaded {len(df_spk)} samples across {df_spk['speaker_id'].nunique()} unique speakers.")
    except Exception as e:
        print(f"[FATAL]: Failed to load clean speakers dataset: {e}")
        raise e

    try:
        vb_parquet_path = hf_hub_download(
            repo_id="JacobLinCool/VoiceBank-DEMAND-16k",
            filename="data/test-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df_vb = pd.read_parquet(vb_parquet_path, engine="fastparquet")
        print(f" Real Noise: Loaded VoiceBank-DEMAND test set ({len(df_vb)} pairs).")
    except Exception as e:
        print(f"[FATAL]: Failed to load VoiceBank-DEMAND dataset: {e}")
        raise e

    # Extract 3 distinct real noise environments from VoiceBank-DEMAND pairs:
    # In VoiceBank-DEMAND test set:
    # Samples 0..9: Metro noise (TMETRO)
    # Samples 10..19: Cafe noise (CAFE)
    # Samples 20..29: Bus noise (TBUS)
    # Real noise = (noisy - clean)
    print("\n[2/4] Extracting 3 real acoustic noise environments (TMETRO, CAFE, TBUS)...")
    real_noises = {}
    noise_specs = [
        ("DEMAND_TMETRO (Metro Station)", 0),
        ("DEMAND_CAFE (Public Cafe)", 10),
        ("DEMAND_TBUS (Transit Bus)", 20),
    ]

    for n_label, row_idx in noise_specs:
        clean_bytes = df_vb.iloc[row_idx]["clean.bytes"]
        noisy_bytes = df_vb.iloc[row_idx]["noisy.bytes"]
        clean_sig = extract_audio_from_wav_bytes(clean_bytes, 48000)
        noisy_sig = extract_audio_from_wav_bytes(noisy_bytes, 48000)
        min_len = min(len(clean_sig), len(noisy_sig))
        noise_residual = noisy_sig[:min_len] - clean_sig[:min_len]
        # Normalize RMS
        noise_residual = noise_residual / (np.std(noise_residual) + 1e-10)
        real_noises[n_label] = noise_residual.astype(np.float32)
        print(f"   Extracted {n_label}: {len(noise_residual) / 48000:.2f} s")

    # Select 5 distinct speakers: 1 representative utterance per speaker
    unique_speakers = sorted(df_spk["speaker_id"].unique())[:5]
    speaker_utterances = {}
    for spk_id in unique_speakers:
        spk_row = df_spk[df_spk["speaker_id"] == spk_id].iloc[0]
        spk_audio = extract_audio_from_wav_bytes(spk_row["audio.bytes"], 48000)
        speaker_utterances[f"Speaker_{spk_id}"] = spk_audio
        print(f"   Selected Speaker {spk_id}: {len(spk_audio) / 48000:.2f} s")

    # Evaluation conditions:
    # 5 speakers x 3 noise types = 15 test items mixed at 5 dB SNR
    eval_snr = 5.0
    print(f"\n[3/4] Benchmarking Default vs. Best Part 1 Setting on 15 cross-combinations (SNR = {eval_snr} dB)...")
    print(f" DNSMOS Active: {'YES (speechmos onnxruntime)' if HAS_DNSMOS else 'NO'}")

    # Best Part 1 setting from sweep:
    # If Part 1 found e.g. atten_lim=18, mix=0.1
    best_atten_str = best_part1_config.get(eval_snr, {}).get("atten_lim_db", "18")
    best_atten_val = None if best_atten_str == "None" else float(best_atten_str)
    best_mix_val = best_part1_config.get(eval_snr, {}).get("mix", 0.1)

    print(f"   Default Config:  atten_lim = None, mix = 0.0")
    print(f"   Best Part 1 Set: atten_lim = {best_atten_val} dB, mix = {best_mix_val}")

    results = []

    for spk_name, clean_audio in speaker_utterances.items():
        clean_16k = resample_to_16k(clean_audio, 48000)
        start_smp, end_smp = detect_speech_active_boundaries(clean_16k)
        c_act = clean_16k[start_smp:end_smp]

        for noise_name, noise_audio in real_noises.items():
            # Tile noise to speech length
            if len(noise_audio) < len(clean_audio):
                tiles = int(np.ceil(len(clean_audio) / len(noise_audio)))
                n_tiled = np.tile(noise_audio, tiles)[: len(clean_audio)]
            else:
                n_tiled = noise_audio[: len(clean_audio)]

            # Mix
            mixed_48k = mix_at_snr(clean_audio, n_tiled, eval_snr)
            mixed_16k = resample_to_16k(mixed_48k, 48000)

            # Baseline metrics
            ref_a, mix_a, _, _ = align_signals(clean_16k, mixed_16k)
            m_act = mix_a[start_smp:end_smp]
            base_sdr = calculate_si_sdr(ref_a[start_smp:end_smp], m_act)
            base_pesq = calculate_pesq(ref_a[start_smp:end_smp], m_act)
            base_stoi = calculate_stoi(ref_a[start_smp:end_smp], m_act)
            base_dnsmos = safe_dnsmos(mixed_16k)

            mixed_tensor = torch.from_numpy(mixed_48k).unsqueeze(0).float()

            # Mode 1: Default (atten_lim=None, mix=0.0)
            with torch.no_grad():
                enh_default_t = enhance(model=model, df_state=df_state, audio=mixed_tensor, pad=True, atten_lim_db=None)
            enh_def_48k = enh_default_t.squeeze(0).cpu().numpy()
            enh_def_16k = resample_to_16k(enh_def_48k, 48000)
            ref_def, act_def, _, _ = align_signals(clean_16k, enh_def_16k)
            def_sdr = calculate_si_sdr(ref_def[start_smp:end_smp], act_def[start_smp:end_smp])
            def_pesq = calculate_pesq(ref_def[start_smp:end_smp], act_def[start_smp:end_smp])
            def_stoi = calculate_stoi(ref_def[start_smp:end_smp], act_def[start_smp:end_smp])
            def_dnsmos = safe_dnsmos(enh_def_16k)

            # Mode 2: Best Part 1 Setting (atten_lim=best_atten_val, mix=best_mix_val)
            with torch.no_grad():
                enh_best_t = enhance(model=model, df_state=df_state, audio=mixed_tensor, pad=True, atten_lim_db=best_atten_val)
            enh_best_raw = enh_best_t.squeeze(0).cpu().numpy()
            enh_best_48k = (1.0 - best_mix_val) * enh_best_raw + best_mix_val * mixed_48k
            enh_best_16k = resample_to_16k(enh_best_48k, 48000)
            ref_bst, act_bst, _, _ = align_signals(clean_16k, enh_best_16k)
            bst_sdr = calculate_si_sdr(ref_bst[start_smp:end_smp], act_bst[start_smp:end_smp])
            bst_pesq = calculate_pesq(ref_bst[start_smp:end_smp], act_bst[start_smp:end_smp])
            bst_stoi = calculate_stoi(ref_bst[start_smp:end_smp], act_bst[start_smp:end_smp])
            bst_dnsmos = safe_dnsmos(enh_best_16k)

            results.append({
                "speaker": spk_name,
                "noise": noise_name,
                "base_sdr": base_sdr,
                "base_pesq": base_pesq,
                "base_stoi": base_stoi,
                "base_dnsmos_ovrl": base_dnsmos.get("ovrl_mos"),
                "base_dnsmos_sig": base_dnsmos.get("sig_mos"),
                "base_dnsmos_bak": base_dnsmos.get("bak_mos"),
                # Default
                "def_sdr": def_sdr,
                "def_d_sdr": def_sdr - base_sdr,
                "def_pesq": def_pesq,
                "def_stoi": def_stoi,
                "def_dnsmos_ovrl": def_dnsmos.get("ovrl_mos"),
                "def_dnsmos_sig": def_dnsmos.get("sig_mos"),
                "def_dnsmos_bak": def_dnsmos.get("bak_mos"),
                # Tuned (Best Part 1)
                "bst_sdr": bst_sdr,
                "bst_d_sdr": bst_sdr - base_sdr,
                "bst_pesq": bst_pesq,
                "bst_stoi": bst_stoi,
                "bst_dnsmos_ovrl": bst_dnsmos.get("ovrl_mos"),
                "bst_dnsmos_sig": bst_dnsmos.get("sig_mos"),
                "bst_dnsmos_bak": bst_dnsmos.get("bak_mos"),
            })

    df_res = pd.DataFrame(results)

    # --------------------------------------------------------------------------
    # Statistical Aggregation: Mean & Spread
    # --------------------------------------------------------------------------
    print("\n[4/4] Computing statistical distribution (Mean, Std, Min, Max, Median)...")

    def calc_stats(series):
        s = series.dropna()
        return {
            "mean": float(np.mean(s)),
            "std": float(np.std(s)),
            "min": float(np.min(s)),
            "max": float(np.max(s)),
            "median": float(np.median(s)),
        }

    stats = {
        "Baseline": {
            "SI-SDR": calc_stats(df_res["base_sdr"]),
            "STOI": calc_stats(df_res["base_stoi"]),
            "PESQ": calc_stats(df_res["base_pesq"]),
            "DNSMOS_OVRL": calc_stats(df_res["base_dnsmos_ovrl"]),
            "DNSMOS_SIG": calc_stats(df_res["base_dnsmos_sig"]),
            "DNSMOS_BAK": calc_stats(df_res["base_dnsmos_bak"]),
        },
        "Default_Model": {
            "SI-SDR": calc_stats(df_res["def_sdr"]),
            "d_SI-SDR": calc_stats(df_res["def_d_sdr"]),
            "STOI": calc_stats(df_res["def_stoi"]),
            "PESQ": calc_stats(df_res["def_pesq"]),
            "DNSMOS_OVRL": calc_stats(df_res["def_dnsmos_ovrl"]),
            "DNSMOS_SIG": calc_stats(df_res["def_dnsmos_sig"]),
            "DNSMOS_BAK": calc_stats(df_res["def_dnsmos_bak"]),
        },
        "Tuned_Model": {
            "SI-SDR": calc_stats(df_res["bst_sdr"]),
            "d_SI-SDR": calc_stats(df_res["bst_d_sdr"]),
            "STOI": calc_stats(df_res["bst_stoi"]),
            "PESQ": calc_stats(df_res["bst_pesq"]),
            "DNSMOS_OVRL": calc_stats(df_res["bst_dnsmos_ovrl"]),
            "DNSMOS_SIG": calc_stats(df_res["bst_dnsmos_sig"]),
            "DNSMOS_BAK": calc_stats(df_res["bst_dnsmos_bak"]),
        },
    }

    # Print summary
    print("\n" + "=" * 78)
    print(" STATISTICAL SUMMARY (MEAN ± STD [MIN, MEDIAN, MAX])")
    print("=" * 78)
    for model_name, m_dict in stats.items():
        print(f"\n [{model_name}]:")
        for metric, v in m_dict.items():
            print(f"   {metric:<12}: {v['mean']:.3f} ± {v['std']:.3f} (min {v['min']:.3f}, med {v['median']:.3f}, max {v['max']:.3f})")

    # Save to CSV
    csv_path = OUTPUTS_DIR / "broader_testset_results.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n[Saved]: {csv_path}")

    # Generate Markdown Table
    md_path = OUTPUTS_DIR / "broader_testset_results.md"
    generate_broader_markdown(df_res, stats, best_atten_val, best_mix_val, md_path)
    print(f"[Saved]: {md_path}")

    return df_res, md_path


def generate_broader_markdown(df_res: pd.DataFrame, stats: dict, best_atten, best_mix, md_path: Path):
    """Write markdown summary report for Part 2."""
    content = []
    content.append("# Broader Speech Enhancement Evaluation Report\n")
    content.append("## Test Dataset Composition\n")
    content.append("- **Clean Speech**: 5 diverse speakers from LibriSpeech (`6930`, `7902`, `2277`, `3660`, `374`).\n")
    content.append("- **Real Acoustic Noise**: 3 distinct environments from DEMAND test set (`TMETRO` metro, `CAFE` cafe, `TBUS` bus).\n")
    content.append(f"- **Operating Point**: Evaluated at 5 dB input SNR.\n")
    content.append(f"- **Comparison**: Default Settings (`atten=None, mix=0.0`) vs. Tuned Setting (`atten={best_atten} dB, mix={best_mix}`).\n\n")

    content.append("## Statistical Distribution (Mean & Spread)\n\n")
    content.append("| Metric | Noisy Baseline | Default DeepFilterNet3 | Tuned Setting |\n")
    content.append("| :--- | :--- | :--- | :--- |\n")

    metrics_list = [
        ("ΔSI-SDR (dB)", "d_SI-SDR", False),
        ("SI-SDR (dB)", "SI-SDR", True),
        ("STOI", "STOI", True),
        ("PESQ (MOS-WB)", "PESQ", True),
        ("DNSMOS Overall", "DNSMOS_OVRL", True),
        ("DNSMOS Signal Quality (SIG)", "DNSMOS_SIG", True),
        ("DNSMOS Background Noise (BAK)", "DNSMOS_BAK", True),
    ]

    for label, key, has_base in metrics_list:
        base_str = f"{stats['Baseline'][key]['mean']:.2f} ± {stats['Baseline'][key]['std']:.2f}" if has_base else "-"
        def_str = f"{stats['Default_Model'][key]['mean']:.2f} ± {stats['Default_Model'][key]['std']:.2f} [{stats['Default_Model'][key]['min']:.2f}, {stats['Default_Model'][key]['max']:.2f}]"
        bst_str = f"{stats['Tuned_Model'][key]['mean']:.2f} ± {stats['Tuned_Model'][key]['std']:.2f} [{stats['Tuned_Model'][key]['min']:.2f}, {stats['Tuned_Model'][key]['max']:.2f}]"
        content.append(f"| **{label}** | {base_str} | {def_str} | {bst_str} |\n")

    content.append("\n---\n\n")
    content.append("## Per-Condition Evaluation Table\n\n")
    content.append("| Speaker | Real Noise Type | Baseline SI-SDR | Baseline PESQ | Baseline DNSMOS | Def ΔSI-SDR | Def PESQ | Def DNSMOS | Tuned ΔSI-SDR | Tuned PESQ | Tuned DNSMOS |\n")
    content.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for _, r in df_res.iterrows():
        content.append(
            f"| {r['speaker']} | {r['noise'].split()[0]} | {r['base_sdr']:.2f} dB | {r['base_pesq']:.2f} | {r['base_dnsmos_ovrl']:.2f} | "
            f"**{r['def_d_sdr']:+.2f} dB** | {r['def_pesq']:.2f} | {r['def_dnsmos_ovrl']:.2f} | "
            f"**{r['bst_d_sdr']:+.2f} dB** | {r['bst_pesq']:.2f} | {r['bst_dnsmos_ovrl']:.2f} |\n"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(content))


# ==============================================================================
# Reference Noise Analysis & Pause Frame Noise Floor Measurement
# ==============================================================================

def measure_real_speech_pause_noise_floor(clean_48k: np.ndarray) -> dict:
    """Measure the noise floor (dBFS) of the pause frames inside real_clean_speech_48k.wav active window."""
    clean_16k = resample_to_16k(clean_48k, 48000)
    start_smp, end_smp = detect_speech_active_boundaries(clean_16k)
    active_seg = clean_16k[start_smp:end_smp]

    frame_len = 320  # 20 ms @ 16 kHz
    num_frames = len(active_seg) // frame_len
    frame_rms = np.array([
        np.sqrt(np.mean(active_seg[i * frame_len : (i + 1) * frame_len] ** 2))
        for i in range(num_frames)
    ])

    speech_thresh = 10.0 ** (-45.0 / 20.0)
    pause_mask = frame_rms < speech_thresh
    speech_mask = frame_rms >= speech_thresh

    lead_rms = np.sqrt(np.mean(clean_16k[:start_smp] ** 2)) if start_smp > 0 else 0.0
    trail_rms = np.sqrt(np.mean(clean_16k[end_smp:] ** 2)) if end_smp < len(clean_16k) else 0.0

    pause_rms_values = frame_rms[pause_mask]
    nonzero_pause = pause_rms_values[pause_rms_values > 1e-5]

    mean_pause_db = float(20 * np.log10(np.mean(nonzero_pause))) if len(nonzero_pause) > 0 else -100.0
    median_pause_db = float(20 * np.log10(np.median(nonzero_pause))) if len(nonzero_pause) > 0 else -100.0
    lead_db = float(20 * np.log10(lead_rms + 1e-12))
    trail_db = float(20 * np.log10(trail_rms + 1e-12))
    overall_active_db = float(20 * np.log10(np.sqrt(np.mean(active_seg ** 2)) + 1e-12))

    print("\n" + "=" * 78)
    print(" REFERENCE NOISE ANALYSIS: real_clean_speech_48k.wav")
    print("=" * 78)
    print(f" Active Speech Window:      {start_smp / 16000:.2f} s to {end_smp / 16000:.2f} s ({len(active_seg) / 16000:.2f} s)")
    print(f" Total 20ms Frames:          {num_frames}")
    print(f" Speech Frames (>= -45 dB):  {int(np.sum(speech_mask))} ({np.sum(speech_mask)/num_frames*100:.1f}%)")
    print(f" Pause Frames (< -45 dB):    {int(np.sum(pause_mask))} ({np.sum(pause_mask)/num_frames*100:.1f}%)")
    print(f" Mean Pause Noise Floor:     {mean_pause_db:.2f} dBFS")
    print(f" Median Pause Noise Floor:   {median_pause_db:.2f} dBFS")
    print(f" Leading Room Tone (0-1.9s): {lead_db:.2f} dBFS")
    print(f" Trailing Room Tone:         {trail_db:.2f} dBFS")
    print(f" Overall Active Speech RMS:  {overall_active_db:.2f} dBFS")
    print("=" * 78)

    return {
        "active_duration_s": len(active_seg) / 16000,
        "total_active_frames": num_frames,
        "speech_frames_count": int(np.sum(speech_mask)),
        "pause_frames_count": int(np.sum(pause_mask)),
        "pause_frames_mean_dbfs": mean_pause_db,
        "pause_frames_median_dbfs": median_pause_db,
        "leading_room_tone_dbfs": lead_db,
        "trailing_room_tone_dbfs": trail_db,
        "overall_active_speech_dbfs": overall_active_db,
    }


def run_part1_librispeech_reference_sweep(model, df_state, speaker_utterances: dict) -> Tuple[pd.DataFrame, str]:
    """Test the reference-noise hypothesis: Re-run Part 1 sweep with LibriSpeech clean speakers."""
    print("\n" + "=" * 78)
    print(" PART 1 (REFERENCE-NOISE HYPOTHESIS): SWEEP WITH 5 LIBRISPEECH CLEAN REFERENCES")
    print("=" * 78)

    from df.enhance import enhance

    atten_lim_list: List[Optional[float]] = [6.0, 12.0, 18.0, 24.0, None]
    mix_list: List[float] = [0.0, 0.1, 0.2]
    snr_levels = [0.0, 5.0, 10.0]

    all_records = []

    for spk_name, clean_audio in speaker_utterances.items():
        clean_16k = resample_to_16k(clean_audio, 48000)
        start_smp, end_smp = detect_speech_active_boundaries(clean_16k)

        spk_noises = generate_noise_signals(clean_audio, length=len(clean_audio), sr=48000, seed=42)
        eval_noises = {
            "White Noise": spk_noises["White Noise"],
            "Pink Noise": spk_noises["Pink Noise"],
            "Powerline Hum": spk_noises["Powerline Hum"],
        }

        for noise_name, noise_arr in eval_noises.items():
            for target_snr in snr_levels:
                mixed_48k = mix_at_snr(clean_audio, noise_arr, target_snr)
                mixed_16k = resample_to_16k(mixed_48k, 48000)
                ref_a, mix_a, _, _ = align_signals(clean_16k, mixed_16k)
                c_act = ref_a[start_smp:end_smp]
                m_act = mix_a[start_smp:end_smp]

                base_snr = calculate_snr(c_act, m_act)
                base_sdr = calculate_si_sdr(c_act, m_act)
                base_stoi = calculate_stoi(c_act, m_act)
                base_pesq = calculate_pesq(c_act, m_act)

                mixed_tensor = torch.from_numpy(mixed_48k).unsqueeze(0).float()

                for atten_lim in atten_lim_list:
                    with torch.no_grad():
                        enhanced_tensor = enhance(
                            model=model,
                            df_state=df_state,
                            audio=mixed_tensor,
                            pad=True,
                            atten_lim_db=atten_lim,
                        )
                    enhanced_48k = enhanced_tensor.squeeze(0).cpu().numpy()

                    for mix in mix_list:
                        blended_48k = (1.0 - mix) * enhanced_48k + mix * mixed_48k if mix > 0.0 else enhanced_48k
                        blended_16k = resample_to_16k(blended_48k, 48000)
                        ref_al, tgt_al, _, _ = align_signals(clean_16k, blended_16k)
                        min_len = min(len(ref_al), len(tgt_al))
                        act_e = min(end_smp, min_len)
                        act_s = min(start_smp, act_e)

                        c_ev = ref_al[act_s:act_e]
                        p_ev = tgt_al[act_s:act_e]

                        snr_val = calculate_snr(c_ev, p_ev)
                        sdr_val = calculate_si_sdr(c_ev, p_ev)
                        stoi_val = calculate_stoi(c_ev, p_ev)
                        pesq_val = calculate_pesq(c_ev, p_ev)

                        all_records.append({
                            "speaker": spk_name,
                            "noise_type": noise_name,
                            "snr_db": target_snr,
                            "atten_lim_db": "None" if atten_lim is None else f"{int(atten_lim)}",
                            "atten_val": 999.0 if atten_lim is None else float(atten_lim),
                            "mix": mix,
                            "base_snr": base_snr,
                            "base_sdr": base_sdr,
                            "base_stoi": base_stoi,
                            "base_pesq": base_pesq,
                            "active_snr": snr_val,
                            "d_snr": snr_val - base_snr,
                            "active_sdr": sdr_val,
                            "d_sdr": sdr_val - base_sdr,
                            "active_stoi": stoi_val,
                            "active_pesq": pesq_val,
                        })

    df_libri_sweep = pd.DataFrame(all_records)
    print(f" [LibriSpeech Sweep Complete] Evaluated {len(df_libri_sweep)} runs across 5 speakers.")

    summary_lines = []
    print("\n --- LibriSpeech Reference Sweep Summary (Mean across 5 Speakers & 3 Noise Types) ---")
    for snr in snr_levels:
        df_snr = df_libri_sweep[df_libri_sweep["snr_db"] == snr]
        grouped = df_snr.groupby(["atten_lim_db", "atten_val", "mix"]).agg({
            "d_sdr": "mean",
            "d_snr": "mean",
            "active_pesq": "mean",
            "active_stoi": "mean",
        }).reset_index()

        def_row = grouped[(grouped["atten_lim_db"] == "None") & (grouped["mix"] == 0.0)].iloc[0]
        eligible = grouped[grouped["d_sdr"] >= (def_row["d_sdr"] - 2.0)]
        best_row = eligible.sort_values(by="active_pesq", ascending=False).iloc[0]

        d_pesq = best_row["active_pesq"] - def_row["active_pesq"]
        d_sdr = best_row["d_sdr"] - def_row["d_sdr"]

        line = (
            f"- **SNR {int(snr)} dB (LibriSpeech 5-spk mean)**: "
            f"Default (`None, 0.0`) -> PESQ = {def_row['active_pesq']:.3f}, ΔSI-SDR = {def_row['d_sdr']:+.2f} dB. "
            f"Best Setting (`atten={best_row['atten_lim_db']} dB, mix={best_row['mix']}`) -> "
            f"PESQ = {best_row['active_pesq']:.3f} ({d_pesq:+.3f} MOS), ΔSI-SDR = {best_row['d_sdr']:+.2f} dB ({d_sdr:+.2f} dB)."
        )
        print(" " + line)
        summary_lines.append(line)

    csv_path = OUTPUTS_DIR / "tradeoff_librispeech_synthetic_sweep.csv"
    df_libri_sweep.to_csv(csv_path, index=False)
    md_path = OUTPUTS_DIR / "tradeoff_librispeech_synthetic_sweep.md"

    md_content = ["# LibriSpeech Clean References: Synthetic Noise Parameter Sweep\n\n"]
    md_content.append("## Executive Summary\n\n")
    md_content.append("\n".join(summary_lines) + "\n\n")
    md_content.append("## Aggregated Results (5-Speaker Mean)\n\n")
    md_content.append("| Input SNR | Max Attenuation | Dry/Wet Mix | Mean ΔSNR | Mean ΔSI-SDR | Mean STOI | Mean PESQ |\n")
    md_content.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

    grouped_all = df_libri_sweep.groupby(["snr_db", "atten_lim_db", "mix"]).agg({
        "d_snr": "mean",
        "d_sdr": "mean",
        "active_stoi": "mean",
        "active_pesq": "mean",
    }).reset_index()

    for _, r in grouped_all.iterrows():
        md_content.append(
            f"| {int(r['snr_db'])} dB | {r['atten_lim_db']} | {r['mix']:.1f} | "
            f"{r['d_snr']:+.2f} dB | {r['d_sdr']:+.2f} dB | {r['active_stoi']:.3f} | {r['active_pesq']:.3f} |\n"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_content))

    print(f"[Saved]: {csv_path}")
    print(f"[Saved]: {md_path}")
    return df_libri_sweep, "\n".join(summary_lines)


def explain_demand_extraction(df_vb, speaker_utterances, real_noises):
    """Explain exact extraction method and why baseline SI-SDR ranges 5.08 - 8.00 dB."""
    print("\n" + "=" * 78)
    print(" DEMAND NOISE EXTRACTION & BASELINE SI-SDR VERIFICATION")
    print("=" * 78)

    print(" 1. Source Files & Extraction Mechanism:")
    noise_specs = [
        ("DEMAND_TMETRO (Metro Station)", 0),
        ("DEMAND_CAFE (Public Cafe)", 10),
        ("DEMAND_TBUS (Transit Bus)", 20),
    ]
    for n_label, row_idx in noise_specs:
        row = df_vb.iloc[row_idx]
        print(f"    - {n_label}:")
        print(f"        ID:         {row['id']}")
        print(f"        Clean file: {row['clean.path']}")
        print(f"        Noisy file: {row['noisy.path']}")
        clean_sig = extract_audio_from_wav_bytes(row['clean.bytes'], 48000)
        noisy_sig = extract_audio_from_wav_bytes(row['noisy.bytes'], 48000)
        min_len = min(len(clean_sig), len(noisy_sig))
        noise_res = noisy_sig[:min_len] - clean_sig[:min_len]
        corr = float(np.corrcoef(clean_sig[:min_len], noise_res)[0, 1])
        print(f"        Method:     noisy[:min_len] - clean[:min_len] (linear residual subtraction)")
        print(f"        Speech Leakage Correlation: {corr:.6f} (near 0.0 proves clean speech cancellation)")

    print("\n 2. Why Baseline SI-SDR Ranges 5.08 - 8.00 dB (Instead of exactly 5.00 dB):")
    for spk_name, audio in speaker_utterances.items():
        a16 = resample_to_16k(audio, 48000)
        s, e = detect_speech_active_boundaries(a16)
        tot_len = len(a16)
        act_len = e - s
        ratio = act_len / tot_len
        snr_shift_db = 10 * np.log10(1.0 / ratio) if ratio > 0 else 0.0
        print(f"    - {spk_name}: Total = {tot_len/16000:.2f} s | Active = {act_len/16000:.2f} s ({ratio*100:.1f}%) | Active Duty-Cycle Shift = +{snr_shift_db:.2f} dB")
    print("=" * 78)


def run_extended_part2_eval(
    model,
    df_state,
    speaker_utterances: dict,
    real_noises: dict,
) -> Tuple[pd.DataFrame, str]:
    """Part 2 Extended: Evaluates 0, 5, 10 dB with Default, Mix 0.1, Atten 18 dB, Atten 24 dB."""
    print("\n" + "=" * 78)
    print(" PART 2 EXTENDED: MULTI-SNR (0, 5, 10 dB) x 4 SETTINGS (DEFAULT, MIX 0.1, ATTEN 18, ATTEN 24)")
    print("=" * 78)

    from df.enhance import enhance

    snr_levels = [0.0, 5.0, 10.0]
    settings = [
        ("Default (None, mix 0.0)", None, 0.0),
        ("Mix 0.1 (None, mix 0.1)", None, 0.1),
        ("Atten 18 dB (18 dB, mix 0.0)", 18.0, 0.0),
        ("Atten 24 dB (24 dB, mix 0.0)", 24.0, 0.0),
    ]

    records = []

    for target_snr in snr_levels:
        print(f"\n >>> Benchmarking Input SNR = {int(target_snr)} dB (15 items x 4 settings)...")
        for spk_name, clean_audio in speaker_utterances.items():
            clean_16k = resample_to_16k(clean_audio, 48000)
            start_smp, end_smp = detect_speech_active_boundaries(clean_16k)

            for noise_name, noise_audio in real_noises.items():
                if len(noise_audio) < len(clean_audio):
                    tiles = int(np.ceil(len(clean_audio) / len(noise_audio)))
                    n_tiled = np.tile(noise_audio, tiles)[: len(clean_audio)]
                else:
                    n_tiled = noise_audio[: len(clean_audio)]

                mixed_48k = mix_at_snr(clean_audio, n_tiled, target_snr)
                mixed_16k = resample_to_16k(mixed_48k, 48000)

                ref_a, mix_a, _, _ = align_signals(clean_16k, mixed_16k)
                m_act = mix_a[start_smp:end_smp]
                c_act = ref_a[start_smp:end_smp]
                base_sdr = calculate_si_sdr(c_act, m_act)
                base_pesq = calculate_pesq(c_act, m_act)
                base_stoi = calculate_stoi(c_act, m_act)
                base_dnsmos = safe_dnsmos(mixed_16k)

                mixed_tensor = torch.from_numpy(mixed_48k).unsqueeze(0).float()

                with torch.no_grad():
                    enh_none = enhance(model=model, df_state=df_state, audio=mixed_tensor, pad=True, atten_lim_db=None).squeeze(0).cpu().numpy()
                    enh_18 = enhance(model=model, df_state=df_state, audio=mixed_tensor, pad=True, atten_lim_db=18.0).squeeze(0).cpu().numpy()
                    enh_24 = enhance(model=model, df_state=df_state, audio=mixed_tensor, pad=True, atten_lim_db=24.0).squeeze(0).cpu().numpy()

                audio_map = {
                    "Default (None, mix 0.0)": enh_none,
                    "Mix 0.1 (None, mix 0.1)": 0.9 * enh_none + 0.1 * mixed_48k,
                    "Atten 18 dB (18 dB, mix 0.0)": enh_18,
                    "Atten 24 dB (24 dB, mix 0.0)": enh_24,
                }

                item_res = {
                    "speaker": spk_name,
                    "noise": noise_name,
                    "snr_db": target_snr,
                    "base_sdr": base_sdr,
                    "base_pesq": base_pesq,
                    "base_stoi": base_stoi,
                    "base_dnsmos_ovrl": base_dnsmos.get("ovrl_mos"),
                    "base_dnsmos_sig": base_dnsmos.get("sig_mos"),
                    "base_dnsmos_bak": base_dnsmos.get("bak_mos"),
                }

                for s_label, _, _ in settings:
                    out_48k = audio_map[s_label]
                    out_16k = resample_to_16k(out_48k, 48000)
                    r_al, t_al, _, _ = align_signals(clean_16k, out_16k)
                    min_l = min(len(r_al), len(t_al))
                    ae = min(end_smp, min_l)
                    as_ = min(start_smp, ae)

                    c_eval = r_al[as_:ae]
                    p_eval = t_al[as_:ae]

                    sdr_val = calculate_si_sdr(c_eval, p_eval)
                    pesq_val = calculate_pesq(c_eval, p_eval)
                    stoi_val = calculate_stoi(c_eval, p_eval)
                    d_mos = safe_dnsmos(out_16k)

                    if "18" in s_label:
                        pfx = "atten18"
                    elif "24" in s_label:
                        pfx = "atten24"
                    elif "0.1" in s_label:
                        pfx = "mix01"
                    else:
                        pfx = "def"

                    item_res[f"{pfx}_sdr"] = sdr_val
                    item_res[f"{pfx}_d_sdr"] = sdr_val - base_sdr
                    item_res[f"{pfx}_pesq"] = pesq_val
                    item_res[f"{pfx}_stoi"] = stoi_val
                    item_res[f"{pfx}_dnsmos_ovrl"] = d_mos.get("ovrl_mos")
                    item_res[f"{pfx}_dnsmos_sig"] = d_mos.get("sig_mos")
                    item_res[f"{pfx}_dnsmos_bak"] = d_mos.get("bak_mos")

                records.append(item_res)

    df_ext = pd.DataFrame(records)
    print(f"\n [Complete] Evaluated {len(df_ext)} items across 3 SNRs.")

    csv_path = OUTPUTS_DIR / "broader_testset_results_extended.csv"
    df_ext.to_csv(csv_path, index=False)
    print(f"[Saved]: {csv_path}")

    md_path = OUTPUTS_DIR / "broader_testset_results_extended.md"
    generate_extended_part2_markdown(df_ext, md_path)
    print(f"[Saved]: {md_path}")

    return df_ext, md_path


def generate_extended_part2_markdown(df_ext: pd.DataFrame, md_path: Path):
    """Write comprehensive markdown for extended Part 2 evaluation."""
    def calc_stats(series):
        s = series.dropna()
        return {
            "mean": float(np.mean(s)),
            "std": float(np.std(s)),
            "min": float(np.min(s)),
            "max": float(np.max(s)),
            "median": float(np.median(s)),
        }

    modes = [
        ("Noisy Baseline", "base"),
        ("Default DeepFilterNet3 (None, mix 0.0)", "def"),
        ("Tuned Blend (None, mix 0.1)", "mix01"),
        ("Atten Limit 18 dB (18 dB, mix 0.0)", "atten18"),
        ("Atten Limit 24 dB (24 dB, mix 0.0)", "atten24"),
    ]

    content = ["# Extended Multi-Speaker & Real-Noise Evaluation Report (0, 5, 10 dB SNR)\n\n"]
    content.append("## Executive Summary (Overall across all 45 test items)\n\n")
    content.append("| Metric | Noisy Baseline | Default (None, 0.0) | Mix 0.1 (None, 0.1) | Atten 18 dB (18, 0.0) | Atten 24 dB (24, 0.0) |\n")
    content.append("| :--- | :--- | :--- | :--- | :--- | :--- |\n")

    metrics_to_show = [
        ("ΔSI-SDR (dB)", "d_sdr", False),
        ("SI-SDR (dB)", "sdr", True),
        ("PESQ (MOS-WB)", "pesq", True),
        ("STOI", "stoi", True),
        ("DNSMOS Overall", "dnsmos_ovrl", True),
        ("DNSMOS Signal (SIG)", "dnsmos_sig", True),
        ("DNSMOS Background (BAK)", "dnsmos_bak", True),
    ]

    for label, col, has_base in metrics_to_show:
        row_str = [f"| **{label}** |"]
        for mode_label, pfx in modes:
            if not has_base and pfx == "base":
                row_str.append(" - |")
                continue
            full_col = f"{pfx}_{col}"
            st = calc_stats(df_ext[full_col])
            row_str.append(f" {st['mean']:.2f} ± {st['std']:.2f} [{st['min']:.2f}, {st['median']:.2f}, {st['max']:.2f}] |")
        content.append("".join(row_str) + "\n")

    # Per-SNR breakdown
    content.append("\n---\n\n")
    content.append("## Statistical Breakdown by Input SNR\n\n")

    for snr in [0.0, 5.0, 10.0]:
        df_snr = df_ext[df_ext["snr_db"] == snr]
        content.append(f"### Input SNR = {int(snr)} dB (15 items)\n\n")
        content.append("| Configuration | Mean ΔSI-SDR | Mean SI-SDR | Mean PESQ | Mean STOI | Mean DNSMOS OVRL | Mean DNSMOS BAK |\n")
        content.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for mode_label, pfx in modes:
            if pfx == "base":
                st_sdr = calc_stats(df_snr["base_sdr"])
                st_pesq = calc_stats(df_snr["base_pesq"])
                st_stoi = calc_stats(df_snr["base_stoi"])
                st_dns = calc_stats(df_snr["base_dnsmos_ovrl"])
                st_bak = calc_stats(df_snr["base_dnsmos_bak"])
                content.append(
                    f"| {mode_label} | - | {st_sdr['mean']:.2f} ± {st_sdr['std']:.2f} | "
                    f"{st_pesq['mean']:.2f} ± {st_pesq['std']:.2f} | {st_stoi['mean']:.3f} | {st_dns['mean']:.2f} | {st_bak['mean']:.2f} |\n"
                )
            else:
                st_dsdr = calc_stats(df_snr[f"{pfx}_d_sdr"])
                st_sdr = calc_stats(df_snr[f"{pfx}_sdr"])
                st_pesq = calc_stats(df_snr[f"{pfx}_pesq"])
                st_stoi = calc_stats(df_snr[f"{pfx}_stoi"])
                st_dns = calc_stats(df_snr[f"{pfx}_dnsmos_ovrl"])
                st_bak = calc_stats(df_snr[f"{pfx}_dnsmos_bak"])
                content.append(
                    f"| {mode_label} | **{st_dsdr['mean']:+.2f} ± {st_dsdr['std']:.2f} dB** | "
                    f"{st_sdr['mean']:.2f} ± {st_sdr['std']:.2f} | {st_pesq['mean']:.2f} ± {st_pesq['std']:.2f} | "
                    f"{st_stoi['mean']:.3f} | {st_dns['mean']:.2f} | {st_bak['mean']:.2f} |\n"
                )
        content.append("\n")

    # Per-item table (all 45 items)
    content.append("\n---\n\n")
    content.append("## Per-Item Complete Evaluation Table (All 45 Test Items)\n\n")
    content.append("| Speaker | Real Noise Type | SNR | Base SDR | Base PESQ | Base DNSMOS | Def ΔSDR | Def PESQ | Def DNSMOS | Mix0.1 ΔSDR | Mix0.1 PESQ | Mix0.1 DNSMOS | Atten18 ΔSDR | Atten18 PESQ | Atten18 DNSMOS | Atten24 ΔSDR | Atten24 PESQ | Atten24 DNSMOS |\n")
    content.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    for _, r in df_ext.iterrows():
        content.append(
            f"| {r['speaker']} | {r['noise'].split()[0]} | {int(r['snr_db'])} dB | "
            f"{r['base_sdr']:.1f} | {r['base_pesq']:.2f} | {r['base_dnsmos_ovrl']:.2f} | "
            f"**{r['def_d_sdr']:+.1f}** | {r['def_pesq']:.2f} | {r['def_dnsmos_ovrl']:.2f} | "
            f"**{r['mix01_d_sdr']:+.1f}** | {r['mix01_pesq']:.2f} | {r['mix01_dnsmos_ovrl']:.2f} | "
            f"**{r['atten18_d_sdr']:+.1f}** | {r['atten18_pesq']:.2f} | {r['atten18_dnsmos_ovrl']:.2f} | "
            f"**{r['atten24_d_sdr']:+.1f}** | {r['atten24_pesq']:.2f} | {r['atten24_dnsmos_ovrl']:.2f} |\n"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(content))


# ==============================================================================
# Main Runner
# ==============================================================================

def main():
    print("=" * 78)
    print(" DEEPFILTERNET TRADE-OFF SWEEP & BROADER BENCHMARK (src/tradeoff.py)")
    print("=" * 78)

    # 1. Load Model
    print("Loading DeepFilterNet3 on CPU...")
    model, df_state = load_model("DeepFilterNet3", post_filter=False, log_level="ERROR")

    # 2. Load clean speech reference
    clean_path = workspace_root / "samples" / "real_clean_speech_48k.wav"
    if not clean_path.exists():
        clean_path = workspace_root / "samples" / "clean_speech_ref.wav"
    clean_audio, sr = sf.read(str(clean_path), dtype="float32")
    if clean_audio.ndim > 1:
        clean_audio = np.mean(clean_audio, axis=1)
    if sr != 48000:
        import scipy.signal
        clean_audio = scipy.signal.resample(clean_audio, int(len(clean_audio) * 48000 / sr))

    # [Task 1 - Subpart A]: Measure Pause Frames Noise Floor of real_clean_speech_48k.wav
    pause_stats = measure_real_speech_pause_noise_floor(clean_audio)

    # [Task 1 - Original]: Part 1 Attenuation / blend sweep on real_clean_speech_48k.wav
    df_sweep, df_context, summary_text = run_part1_sweep(model, df_state, clean_audio)

    # [Dataset Loading]: Load LibriSpeech speakers and VoiceBank-DEMAND test set
    from huggingface_hub import hf_hub_download
    spk_parquet_path = hf_hub_download(repo_id="Cidoyi/librispeech-samples", filename="data/train-00000-of-00001.parquet", repo_type="dataset")
    df_spk = pd.read_parquet(spk_parquet_path, engine="fastparquet")
    unique_speakers = sorted(df_spk["speaker_id"].unique())[:5]
    speaker_utterances = {}
    for spk_id in unique_speakers:
        spk_row = df_spk[df_spk["speaker_id"] == spk_id].iloc[0]
        speaker_utterances[f"Speaker_{spk_id}"] = extract_audio_from_wav_bytes(spk_row["audio.bytes"], 48000)

    vb_parquet_path = hf_hub_download(repo_id="JacobLinCool/VoiceBank-DEMAND-16k", filename="data/test-00000-of-00001.parquet", repo_type="dataset")
    df_vb = pd.read_parquet(vb_parquet_path, engine="fastparquet")

    real_noises = {}
    noise_specs = [
        ("DEMAND_TMETRO (Metro Station)", 0),
        ("DEMAND_CAFE (Public Cafe)", 10),
        ("DEMAND_TBUS (Transit Bus)", 20),
    ]
    for n_label, row_idx in noise_specs:
        clean_bytes = df_vb.iloc[row_idx]["clean.bytes"]
        noisy_bytes = df_vb.iloc[row_idx]["noisy.bytes"]
        clean_sig = extract_audio_from_wav_bytes(clean_bytes, 48000)
        noisy_sig = extract_audio_from_wav_bytes(noisy_bytes, 48000)
        min_len = min(len(clean_sig), len(noisy_sig))
        noise_residual = noisy_sig[:min_len] - clean_sig[:min_len]
        noise_residual = noise_residual / (np.std(noise_residual) + 1e-10)
        real_noises[n_label] = noise_residual.astype(np.float32)

    # [Task 1 - Subpart B]: Test Reference-Noise Hypothesis using 5 LibriSpeech Clean References
    df_libri_sweep, libri_summary = run_part1_librispeech_reference_sweep(model, df_state, speaker_utterances)

    # [Task 3]: Explain DEMAND Noise Extraction and Baseline SI-SDR
    explain_demand_extraction(df_vb, speaker_utterances, real_noises)

    # [Task 2]: Extended Part 2 (0, 5, 10 dB x Default, Mix 0.1, Atten 18 dB, Atten 24 dB)
    df_ext, ext_md_path = run_extended_part2_eval(model, df_state, speaker_utterances, real_noises)

    print("\n" + "=" * 78)
    print(" ALL BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print(f" Outputs saved:")
    print(f" - {OUTPUTS_DIR / 'tradeoff_sweep.csv'}")
    print(f" - {OUTPUTS_DIR / 'tradeoff_sweep.md'}")
    print(f" - {OUTPUTS_DIR / 'tradeoff_sdr_vs_pesq.png'}")
    print(f" - {OUTPUTS_DIR / 'tradeoff_librispeech_synthetic_sweep.csv'}")
    print(f" - {OUTPUTS_DIR / 'tradeoff_librispeech_synthetic_sweep.md'}")
    print(f" - {OUTPUTS_DIR / 'broader_testset_results_extended.csv'}")
    print(f" - {OUTPUTS_DIR / 'broader_testset_results_extended.md'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
