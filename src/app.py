"""Interactive Web UI for Background Noise Suppression using Gradio and DeepFilterNet.

Features:
- File upload & microphone recording
- Tunable noise attenuation limit (dB) & optional post-filtering
- Cleaned audio playback
- Matplotlib before/after spectrograms with a shared color scale
- Aligned waveform comparison plot
- Real-time performance metrics (Inference time, duration, RTF)
- Comprehensive error handling
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import soundfile as sf
import torch

# Add workspace root to sys.path to ensure src imports work seamlessly
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.denoise import denoise_audio, load_audio_file, load_model, save_audio_file

# ==============================================================================
# Model Initialization at Startup (Loaded ONCE)
# ==============================================================================
print("Initializing DeepFilterNet model for Web UI...")
DEFAULT_MODEL, DEFAULT_DF_STATE = load_model(model_name="DeepFilterNet3", post_filter=False)
print("Model initialized and cached successfully!")


def generate_spectrogram_plot(
    noisy_audio: np.ndarray,
    denoised_audio: np.ndarray,
    sr: int = 48000,
) -> plt.Figure:
    """Generate side-by-side Before/After spectrograms with an exact shared color scale."""
    # Convert multi-channel to mono for spectrogram display
    noisy_mono = np.mean(noisy_audio, axis=0) if noisy_audio.ndim > 1 else noisy_audio
    denoised_mono = np.mean(denoised_audio, axis=0) if denoised_audio.ndim > 1 else denoised_audio

    # STFT Spectrogram calculation
    nperseg = 1024
    noverlap = 768
    f1, t1, Sxx1 = signal.spectrogram(noisy_mono, fs=sr, nperseg=nperseg, noverlap=noverlap)
    f2, t2, Sxx2 = signal.spectrogram(denoised_mono, fs=sr, nperseg=nperseg, noverlap=noverlap)

    # Convert power spectral density to dB scale
    Sxx1_db = 10 * np.log10(np.maximum(Sxx1, 1e-10))
    Sxx2_db = 10 * np.log10(np.maximum(Sxx2, 1e-10))

    # Shared color bounds across BOTH spectrograms for fair visual comparison
    vmax = float(max(np.max(Sxx1_db), np.max(Sxx2_db)))
    vmin = float(max(min(np.min(Sxx1_db), np.min(Sxx2_db)), vmax - 80))  # 80 dB dynamic range

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, constrained_layout=True)
    f_khz = f1 / 1000.0  # kHz display

    # Top: Input Noisy Spectrogram
    im1 = axes[0].pcolormesh(t1, f_khz, Sxx1_db, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0].set_title("Input Audio (Noisy) Spectrogram", fontsize=11, fontweight="bold", pad=6)
    axes[0].set_ylabel("Frequency (kHz)", fontsize=10)
    axes[0].set_ylim(0, 24)

    # Bottom: Denoised Spectrogram
    im2 = axes[1].pcolormesh(t2, f_khz, Sxx2_db, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title("Denoised Audio Spectrogram", fontsize=11, fontweight="bold", pad=6)
    axes[1].set_ylabel("Frequency (kHz)", fontsize=10)
    axes[1].set_xlabel("Time (seconds)", fontsize=10)
    axes[1].set_ylim(0, 24)

    # Shared unified colorbar
    cbar = fig.colorbar(im2, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label("Magnitude (dB)", fontsize=9)

    return fig


def generate_waveform_plot(
    noisy_audio: np.ndarray,
    denoised_audio: np.ndarray,
    sr: int = 48000,
) -> plt.Figure:
    """Generate time-domain waveform comparison plot."""
    noisy_mono = np.mean(noisy_audio, axis=0) if noisy_audio.ndim > 1 else noisy_audio
    denoised_mono = np.mean(denoised_audio, axis=0) if denoised_audio.ndim > 1 else denoised_audio

    t_noisy = np.arange(len(noisy_mono)) / sr
    t_denoised = np.arange(len(denoised_mono)) / sr

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True, sharey=True, constrained_layout=True)

    # Top: Input Noisy Waveform
    axes[0].plot(t_noisy, noisy_mono, color="#e74c3c", linewidth=0.6, alpha=0.9)
    axes[0].set_title("Input Audio (Noisy) Waveform", fontsize=11, fontweight="bold", pad=6)
    axes[0].set_ylabel("Amplitude", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylim(-1.05, 1.05)

    # Bottom: Denoised Waveform
    axes[1].plot(t_denoised, denoised_mono, color="#27ae60", linewidth=0.6, alpha=0.9)
    axes[1].set_title("Denoised Audio Waveform", fontsize=11, fontweight="bold", pad=6)
    axes[1].set_ylabel("Amplitude", fontsize=10)
    axes[1].set_xlabel("Time (seconds)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].set_ylim(-1.05, 1.05)

    return fig


def process_audio(
    audio_path: Optional[str],
    no_limit_atten: bool,
    atten_slider_val: float,
    enable_post_filter: bool,
) -> Tuple[Optional[str], Optional[plt.Figure], Optional[plt.Figure], str]:
    """Enhance input audio and generate evaluation plots and metrics.

    Returns:
        Tuple of (cleaned_audio_path, spectrogram_fig, waveform_fig, metrics_markdown).
    """
    # 1. Validation & Error Handling
    if not audio_path or not os.path.exists(audio_path):
        return (
            None,
            None,
            None,
            "⚠️ **No input audio detected.** Please upload a valid audio file or record speech from your microphone.",
        )

    try:
        # Determine attenuation limit
        atten_lim_db = None if no_limit_atten else float(atten_slider_val)

        # Select model (reuse cached model, switch if post_filter toggled)
        model, df_state = load_model(model_name="DeepFilterNet3", post_filter=enable_post_filter)

        # 2. Load audio and resample to 48kHz
        audio_tensor, orig_sr = load_audio_file(audio_path, target_sr=df_state.sr())

        # 3. Perform enhancement
        enhanced_tensor, meta = denoise_audio(
            audio=audio_tensor,
            sr=df_state.sr(),
            model=model,
            df_state=df_state,
            atten_lim_db=atten_lim_db,
        )

        # 4. Save enhanced audio to a temporary file
        temp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_out_path = temp_out.name
        temp_out.close()
        save_audio_file(temp_out_path, enhanced_tensor, sr=df_state.sr())

        # 5. Prepare numpy arrays for visualization
        noisy_np = audio_tensor.detach().cpu().numpy()
        denoised_np = enhanced_tensor.detach().cpu().numpy()

        spec_fig = generate_spectrogram_plot(noisy_np, denoised_np, sr=df_state.sr())
        wave_fig = generate_waveform_plot(noisy_np, denoised_np, sr=df_state.sr())

        # 6. Format metrics
        rtf = meta["rtf"]
        speedup = f"{1.0 / max(rtf, 1e-4):.1f}x faster than real-time" if rtf < 1.0 else "Real-time"
        atten_display = "No limit (Full suppression)" if atten_lim_db is None else f"{atten_lim_db:.1f} dB"

        metrics_md = f"""
### 📊 Enhancement Performance Metrics
| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Duration** | `{meta['duration_sec']:.2f} s` | Total audio length |
| **Inference Time** | `{meta['process_time_sec']:.4f} s` | Processing time on CPU |
| **Real-Time Factor (RTF)** | `{rtf:.4f}` | **{speedup}** |
| **Sample Rate** | `{meta['sample_rate']} Hz` | DeepFilterNet native 48 kHz |
| **Attenuation Limit** | `{atten_display}` | Max noise suppression ceiling |
| **Post-Filter** | `{'Enabled' if enable_post_filter else 'Disabled'}` | Additional stage filtering |
"""
        return temp_out_path, spec_fig, wave_fig, metrics_md

    except Exception as e:
        error_msg = f"❌ **Error processing audio:** `{str(e)}`\n\nPlease ensure the uploaded file is a valid audio format (WAV, MP3, FLAC, OGG, etc.)."
        return None, None, None, error_msg


CUSTOM_CSS = """
.main-header { text-align: center; margin-bottom: 20px; }
.main-header h1 { font-size: 2.2rem; font-weight: 700; margin-bottom: 6px; }
.main-header p { font-size: 1.05rem; color: #666; }
"""


def build_app() -> gr.Blocks:
    """Construct the Gradio UI layout."""
    with gr.Blocks(title="Background Noise Suppression") as demo:
        with gr.Column(elem_classes=["main-header"]):
            gr.Markdown(
                """
                # 🎙️ Background Noise Suppression
                ### DeepFilterNet-powered fullband speech enhancement running on CPU
                """
            )

        with gr.Row():
            # Left Column: Inputs & Controls
            with gr.Column(scale=1):
                gr.Markdown("### 1. Audio Input")
                input_audio = gr.Audio(
                    label="Upload Audio File or Record from Microphone",
                    sources=["upload", "microphone"],
                    type="filepath",
                )

                gr.Markdown("### 2. Processing Parameters")
                no_limit_atten = gr.Checkbox(
                    label="No Limit / Maximum Attenuation (Default)",
                    value=True,
                    info="Recommended for maximum background noise elimination.",
                )
                atten_slider = gr.Slider(
                    label="Noise Attenuation Limit (dB)",
                    minimum=0,
                    maximum=100,
                    value=100,
                    step=1,
                    interactive=False,
                    info="Only active when 'No Limit' is unchecked. Lower values preserve more ambient sound.",
                )
                post_filter = gr.Checkbox(
                    label="Enable Post-Filtering",
                    value=False,
                    info="Applies extra spectral thresholding for aggressive noise reduction.",
                )

                # Dynamically enable/disable slider based on checkbox
                def toggle_slider(is_unlimited: bool):
                    return gr.update(interactive=not is_unlimited)

                no_limit_atten.change(
                    fn=toggle_slider,
                    inputs=[no_limit_atten],
                    outputs=[atten_slider],
                )

                process_btn = gr.Button("🚀 Denoise Audio", variant="primary", size="lg")

                # Clean labeled audio presets (audio input only, no raw parameter columns)
                raw_presets = [
                    ("samples/real_noisy_speech.wav", "Real Recorded Speech (Noisy)"),
                    ("samples/speech_with_whitenoise.wav", "Speech + White Noise"),
                    ("samples/speech_with_hum.wav", "Speech + Powerline Hum"),
                    ("samples/speech_with_buzz.wav", "Speech + Harsh Buzz"),
                ]
                existing_presets = [p for p in raw_presets if os.path.exists(p[0])]
                if existing_presets:
                    gr.Markdown("#### Quick Test Presets")
                    gr.Examples(
                        examples=[[p[0]] for p in existing_presets],
                        inputs=[input_audio],
                        example_labels=[p[1] for p in existing_presets],
                        label="Click any test preset to load into player:",
                    )

            # Right Column: Outputs & Analysis
            with gr.Column(scale=1):
                gr.Markdown("### 3. Denoised Output")
                output_audio = gr.Audio(
                    label="Cleaned Audio",
                    type="filepath",
                    interactive=False,
                )

                metrics_output = gr.Markdown(
                    value="*Upload or record audio and click **Denoise Audio** to see results.*"
                )

        gr.Markdown("---")
        gr.Markdown("### 4. Visual Audio Analysis")
        with gr.Tabs():
            with gr.TabItem("📈 Spectrogram Comparison"):
                spec_plot = gr.Plot(label="Spectrogram Comparison (Shared dB scale)")
            with gr.TabItem("🌊 Waveform Comparison"):
                wave_plot = gr.Plot(label="Waveform Comparison")

        # Connect button click
        process_btn.click(
            fn=process_audio,
            inputs=[input_audio, no_limit_atten, atten_slider, post_filter],
            outputs=[output_audio, spec_plot, wave_plot, metrics_output],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
