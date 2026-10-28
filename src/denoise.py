"""Core Denoising Engine using DeepFilterNet.

Provides functions to load, resample, enhance, and save audio files,
as well as a command-line interface for single-file and batch processing.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Global model cache to avoid reloading on every function call
_CACHED_MODEL = None
_CACHED_DF_STATE = None
_CACHED_MODEL_NAME = None


def load_model(
    model_name: str = "DeepFilterNet3",
    post_filter: bool = False,
    log_level: str = "WARNING",
):
    """Load and cache the DeepFilterNet model and state.

    Args:
        model_name: Model identifier (e.g., 'DeepFilterNet3', 'DeepFilterNet2').
        post_filter: Whether to enable post-filter stage for aggressive attenuation.
        log_level: Logging verbosity for DeepFilterNet internals.

    Returns:
        Tuple of (model, df_state).
    """
    global _CACHED_MODEL, _CACHED_DF_STATE, _CACHED_MODEL_NAME

    if _CACHED_MODEL is not None and _CACHED_MODEL_NAME == (model_name, post_filter):
        return _CACHED_MODEL, _CACHED_DF_STATE

    from df.enhance import init_df

    model, df_state, _ = init_df(
        default_model=model_name,
        post_filter=post_filter,
        log_level=log_level,
    )
    model.eval()

    _CACHED_MODEL = model
    _CACHED_DF_STATE = df_state
    _CACHED_MODEL_NAME = (model_name, post_filter)
    return model, df_state


def load_audio_file(file_path: Union[str, Path], target_sr: int = 48000) -> Tuple[torch.Tensor, int]:
    """Load an audio file, ensure 2D shape [channels, samples], and resample to target_sr.

    Args:
        file_path: Path to the audio file.
        target_sr: Target sample rate in Hz (DeepFilterNet requires 48000 Hz).

    Returns:
        Tuple of (audio_tensor, original_sr).
    """
    path_str = str(file_path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Audio file not found: {path_str}")

    # Use soundfile to load audio as float32 in [-1.0, 1.0]
    data, orig_sr = sf.read(path_str, dtype="float32")

    # Shape conversion:
    # sf.read returns (samples,) for mono or (samples, channels) for multi-channel.
    # PyTorch/DeepFilterNet expects (channels, samples).
    if data.ndim == 1:
        audio = torch.from_numpy(data).unsqueeze(0)  # [1, samples]
    elif data.ndim == 2:
        audio = torch.from_numpy(data.T)  # [channels, samples]
    else:
        raise ValueError(f"Unsupported audio dimensions: {data.ndim}")

    # Resample if sample rate doesn't match DeepFilterNet native rate
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        audio = resampler(audio)

    return audio, orig_sr


def save_audio_file(
    file_path: Union[str, Path],
    audio: Union[torch.Tensor, np.ndarray],
    sr: int = 48000,
    subtype: str = "PCM_16",
) -> str:
    """Save audio tensor or numpy array to disk as a WAV file.

    Args:
        file_path: Destination file path.
        audio: Audio data as tensor [channels, samples] or array.
        sr: Sample rate in Hz.
        subtype: Audio bit-depth subtype (default: 'PCM_16').

    Returns:
        Resolved absolute string path to the saved file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = audio

    # soundfile expects (samples,) or (samples, channels)
    if audio_np.ndim == 2:
        audio_np = audio_np.T

    # Peak normalization / clipping protection
    max_val = np.max(np.abs(audio_np))
    if max_val > 1.0:
        audio_np = audio_np / max_val

    sf.write(str(path), audio_np, samplerate=sr, subtype=subtype)
    return str(path.resolve())


def denoise_audio(
    audio: torch.Tensor,
    sr: int = 48000,
    model=None,
    df_state=None,
    atten_lim_db: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Denoise an in-memory audio tensor using DeepFilterNet.

    Args:
        audio: Audio tensor of shape [channels, samples].
        sr: Sample rate in Hz (must be 48000 Hz).
        model: Optional pre-loaded model (loads default if None).
        df_state: Optional pre-loaded df_state.
        atten_lim_db: Optional maximum attenuation in dB (e.g. 20, 40).
                      None means full attenuation / maximum suppression.

    Returns:
        Tuple of (enhanced_audio_tensor, metadata_dict).
    """
    if model is None or df_state is None:
        model, df_state = load_model()

    target_sr = df_state.sr()
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)
        sr = target_sr

    from df.enhance import enhance

    num_channels, num_samples = audio.shape
    duration_sec = num_samples / sr

    t_start = time.perf_counter()
    with torch.no_grad():
        enhanced = enhance(
            model=model,
            df_state=df_state,
            audio=audio,
            pad=True,
            atten_lim_db=atten_lim_db,
        )
    t_end = time.perf_counter()

    process_time = t_end - t_start
    rtf = process_time / max(duration_sec, 1e-6)

    meta = {
        "duration_sec": round(duration_sec, 3),
        "process_time_sec": round(process_time, 4),
        "rtf": round(rtf, 4),
        "sample_rate": sr,
        "channels": num_channels,
        "atten_lim_db": atten_lim_db,
    }
    return enhanced, meta


def denoise_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    atten_lim_db: Optional[float] = None,
    post_filter: bool = False,
    model_name: str = "DeepFilterNet3",
) -> Tuple[str, Dict[str, float]]:
    """Denoise a single audio file and save the output.

    Args:
        input_path: Path to input audio file.
        output_path: Optional output path. If None, saves to 'outputs/<name>_denoised.wav'.
        atten_lim_db: Optional attenuation limit in dB.
        post_filter: Whether to apply post-filtering.
        model_name: DeepFilterNet model to use.

    Returns:
        Tuple of (output_file_path, metadata_dict).
    """
    in_path = Path(input_path)
    if output_path is None:
        out_path = Path("outputs") / f"{in_path.stem}_denoised.wav"
    else:
        out_path = Path(output_path)

    model, df_state = load_model(model_name=model_name, post_filter=post_filter)
    audio, orig_sr = load_audio_file(in_path, target_sr=df_state.sr())

    enhanced, meta = denoise_audio(
        audio=audio,
        sr=df_state.sr(),
        model=model,
        df_state=df_state,
        atten_lim_db=atten_lim_db,
    )

    saved_path = save_audio_file(out_path, enhanced, sr=df_state.sr())
    meta["input_file"] = str(in_path)
    meta["output_file"] = saved_path
    meta["original_sample_rate"] = orig_sr
    return saved_path, meta


def denoise_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path] = "outputs",
    atten_lim_db: Optional[float] = None,
    post_filter: bool = False,
    model_name: str = "DeepFilterNet3",
    extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
) -> List[Dict[str, float]]:
    """Batch-denoise all supported audio files in a directory.

    Args:
        input_dir: Directory containing input audio files.
        output_dir: Directory to save enhanced files.
        atten_lim_db: Optional attenuation limit in dB.
        post_filter: Whether to apply post-filtering.
        model_name: Model identifier.
        extensions: Tuple of file extensions to process.

    Returns:
        List of metadata dictionaries for each processed file.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    if not audio_files:
        print(f"No audio files found in {in_dir} with extensions {extensions}")
        return []

    print(f"Found {len(audio_files)} audio file(s) to process...")
    model, df_state = load_model(model_name=model_name, post_filter=post_filter)

    results = []
    for i, file_path in enumerate(audio_files, 1):
        target_path = out_dir / f"{file_path.stem}_denoised.wav"
        print(f"[{i}/{len(audio_files)}] Processing {file_path.name} -> {target_path.name}...")
        _, meta = denoise_file(
            input_path=file_path,
            output_path=target_path,
            atten_lim_db=atten_lim_db,
            post_filter=post_filter,
            model_name=model_name,
        )
        results.append(meta)
        print(f"       Duration: {meta['duration_sec']}s | Time: {meta['process_time_sec']}s | RTF: {meta['rtf']}")

    return results


def parse_args():
    """Parse command line arguments for denoise.py CLI."""
    parser = argparse.ArgumentParser(
        description="DeepFilterNet Audio Denoiser — Background Noise Suppression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to an input audio file or directory of audio files",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output file or output directory (defaults to outputs/)",
    )
    parser.add_argument(
        "-a", "--atten-lim",
        type=float,
        default=None,
        help="Max noise attenuation in dB (e.g. 20, 40). Omit for maximum suppression",
    )
    parser.add_argument(
        "-p", "--post-filter",
        action="store_true",
        help="Enable post-filter for extra noise attenuation",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="DeepFilterNet3",
        help="Model to use (DeepFilterNet3 or DeepFilterNet2)",
    )
    return parser.parse_args()


def main():
    """Main CLI entrypoint."""
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(" Background Noise Suppression (DeepFilterNet CLI)")
    print("=" * 60)
    print(f" Model:         {args.model}")
    print(f" Post-filter:   {args.post_filter}")
    print(f" Attenuation:   {args.atten_lim} dB" if args.atten_lim else " Attenuation:   Maximum suppression (default)")
    print(f" Input path:    {input_path}")
    print("-" * 60)

    if input_path.is_dir():
        out_dir = args.output if args.output else "outputs"
        results = denoise_directory(
            input_dir=input_path,
            output_dir=out_dir,
            atten_lim_db=args.atten_lim,
            post_filter=args.post_filter,
            model_name=args.model,
        )
        print("-" * 60)
        print(f"Batch completed! Processed {len(results)} file(s).")
    else:
        out_file, meta = denoise_file(
            input_path=input_path,
            output_path=args.output,
            atten_lim_db=args.atten_lim,
            post_filter=args.post_filter,
            model_name=args.model,
        )
        print("-" * 60)
        print("Success! File denoised.")
        print(f" Output:   {out_file}")
        print(f" Duration: {meta['duration_sec']}s")
        print(f" Time:     {meta['process_time_sec']}s")
        print(f" RTF:      {meta['rtf']} (values < 1.0 mean faster than real-time)")
    print("=" * 60)


if __name__ == "__main__":
    main()
