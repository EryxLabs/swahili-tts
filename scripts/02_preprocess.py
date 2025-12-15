#!/usr/bin/env python3
"""
Eryx Labs - Swahili TTS Data Preprocessing
Step 2: Prepare audio data for XTTS-v2 fine-tuning

This script:
1. Resamples audio to target sample rate
2. Normalizes audio levels
3. Creates train/validation splits
4. Formats data for Coqui TTS training
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

console = Console()

# Configure logging
Path("logs").mkdir(exist_ok=True)
logger.add(
    "logs/preprocess_{time}.log",
    rotation="50 MB",
    level="INFO"
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent.parent / config_path
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def resample_audio(audio_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """Resample audio to target sample rate."""
    import librosa

    audio, sr = librosa.load(str(audio_path), sr=target_sr)
    return audio, sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to -3dB peak."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        # Normalize to -3dB (0.707)
        audio = audio * (0.707 / peak)
    return audio


def preprocess_data(config: dict) -> None:
    """Preprocess audio data for TTS training."""
    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]ERYX LABS - SWAHILI TTS PREPROCESSING[/bold cyan]")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]\n")

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_path = raw_dir / "metadata.json"
    if not metadata_path.exists():
        console.print("[red]Error: metadata.json not found![/red]")
        console.print("Run data collection first: python scripts/01_download_data.py")
        return

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    console.print(f"[green]Loaded {len(metadata):,} samples from metadata[/green]\n")

    target_sr = config['data']['sample_rate']

    # Process audio files
    console.print("[bold blue]Processing audio files...[/bold blue]")

    processed_audio_dir = processed_dir / "wavs"
    processed_audio_dir.mkdir(exist_ok=True)

    processed_samples = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[cyan]Processing", total=len(metadata))

        for i, sample in enumerate(metadata):
            try:
                import soundfile as sf

                # Load and resample audio
                audio_path = raw_dir / sample['audio_file']
                audio, sr = resample_audio(audio_path, target_sr)

                # Normalize audio
                audio = normalize_audio(audio)

                # Save processed audio
                output_filename = f"sw_{i:06d}.wav"
                output_path = processed_audio_dir / output_filename

                sf.write(str(output_path), audio, target_sr)

                # Add to processed samples
                processed_samples.append({
                    'audio_file': f"wavs/{output_filename}",
                    'text': sample['text'],
                    'duration': len(audio) / target_sr,
                    'speaker_id': 'swahili_speaker'
                })

                progress.update(task, advance=1)

            except Exception as e:
                logger.warning(f"Error processing {sample['audio_file']}: {e}")
                continue

    console.print(f"[green]Processed {len(processed_samples):,} samples[/green]\n")

    # Split into train/val
    console.print("[bold blue]Creating train/validation splits...[/bold blue]")

    random.seed(42)
    random.shuffle(processed_samples)

    train_ratio = config['data']['train_split']
    split_idx = int(len(processed_samples) * train_ratio)

    train_samples = processed_samples[:split_idx]
    val_samples = processed_samples[split_idx:]

    console.print(f"  Train: {len(train_samples):,} samples")
    console.print(f"  Val: {len(val_samples):,} samples")

    # Save in XTTS format (CSV-style metadata)
    # Format: audio_file|text|speaker_id

    def save_metadata_csv(samples: List[Dict], output_path: Path):
        """Save metadata in Coqui TTS format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                # Format: path|text|speaker
                line = f"{sample['audio_file']}|{sample['text']}|{sample['speaker_id']}\n"
                f.write(line)

    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"

    save_metadata_csv(train_samples, train_csv)
    save_metadata_csv(val_samples, val_csv)

    console.print(f"\n[green]Train metadata: {train_csv}[/green]")
    console.print(f"[green]Val metadata: {val_csv}[/green]")

    # Also save as JSON for flexibility
    with open(processed_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)

    with open(processed_dir / "val.json", 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)

    # Calculate and save statistics
    train_duration = sum(s['duration'] for s in train_samples)
    val_duration = sum(s['duration'] for s in val_samples)

    stats = {
        'total_samples': len(processed_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'train_duration_hours': train_duration / 3600,
        'val_duration_hours': val_duration / 3600,
        'total_duration_hours': (train_duration + val_duration) / 3600,
        'sample_rate': target_sr,
    }

    with open(processed_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Display summary
    table = Table(title="Preprocessing Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total samples", f"{stats['total_samples']:,}")
    table.add_row("Train samples", f"{stats['train_samples']:,}")
    table.add_row("Val samples", f"{stats['val_samples']:,}")
    table.add_row("Train duration", f"{stats['train_duration_hours']:.2f} hours")
    table.add_row("Val duration", f"{stats['val_duration_hours']:.2f} hours")
    table.add_row("Sample rate", f"{stats['sample_rate']} Hz")

    console.print("\n")
    console.print(table)


def main():
    """Main entry point."""
    console.print("\n[bold]Eryx Labs - Swahili TTS Preprocessing[/bold]\n")

    config = load_config()
    preprocess_data(config)

    console.print("\n[bold green]Preprocessing complete![/bold green]")
    console.print("[yellow]Next step: python scripts/03_train.py[/yellow]\n")


if __name__ == "__main__":
    main()
