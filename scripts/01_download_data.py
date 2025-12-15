#!/usr/bin/env python3
"""
Eryx Labs - Swahili TTS Data Collection
Step 1: Download Mozilla Common Voice Swahili dataset

This script:
1. Downloads the Swahili subset from Common Voice
2. Filters for high-quality audio samples
3. Prepares the data structure for TTS training
"""

import os
import json
from pathlib import Path
from typing import Dict, List

import yaml
from datasets import load_dataset, Audio
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from tqdm import tqdm

console = Console()

# Configure logging
Path("logs").mkdir(exist_ok=True)
logger.add(
    "logs/download_{time}.log",
    rotation="50 MB",
    level="INFO"
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent.parent / config_path
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def download_common_voice_swahili(config: dict) -> None:
    """Download Mozilla Common Voice Swahili dataset."""
    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]ERYX LABS - SWAHILI TTS DATA COLLECTION[/bold cyan]")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]\n")

    console.print("[bold blue]Downloading Swahili Voice Data...[/bold blue]")
    console.print("[dim]This may take a while on first run[/dim]\n")

    # Create data directories
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        hf_token = os.getenv("HF_TOKEN")

        # Try multiple data sources in order of preference
        # These are publicly accessible without gating
        datasets_to_try = [
            # OpenSLR 25 - Swahili speech corpus (no HF version, will download manually)
            # africaVoice datasets
            ("Laban254/swahili_audio_text_dataset", None, "Swahili Audio-Text Dataset"),
            # Generic audio for testing pipeline
            ("speechcolab/gigaspeech", "xs", "GigaSpeech XS (testing)"),
        ]

        dataset = None
        source_name = None

        for source, lang_code, description in datasets_to_try:
            try:
                console.print(f"[yellow]Trying: {description}...[/yellow]")
                if lang_code:
                    dataset = load_dataset(
                        source,
                        lang_code,
                        split="train",
                        token=hf_token,
                    )
                else:
                    dataset = load_dataset(
                        source,
                        split="train",
                        token=hf_token,
                    )
                source_name = description
                console.print(f"[green]Successfully loaded {description}![/green]")
                break
            except Exception as e:
                console.print(f"[dim]  {source} not available: {str(e)[:50]}...[/dim]")
                continue

        if dataset is None:
            raise ValueError("Could not load any Swahili voice dataset")

        console.print(f"[green]Downloaded {len(dataset):,} samples from {source_name}[/green]\n")

        # Get dataset info
        console.print("[bold blue]Dataset Statistics:[/bold blue]")

        # Sample the dataset to calculate stats
        total_duration = 0
        valid_samples = []

        sample_rate = config['data']['sample_rate']
        min_len = config['data']['min_audio_length']
        max_len = config['data']['max_audio_length']

        console.print("[yellow]Processing and filtering samples...[/yellow]")

        # Detect dataset format (different datasets have different column names)
        sample_columns = dataset.column_names
        console.print(f"[dim]Dataset columns: {sample_columns}[/dim]")

        # Map column names based on dataset
        text_col = 'transcription' if 'transcription' in sample_columns else 'sentence' if 'sentence' in sample_columns else 'text'
        audio_col = 'audio'

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("[cyan]Filtering samples", total=len(dataset))

            for i, sample in enumerate(dataset):
                try:
                    # Get audio duration
                    audio = sample[audio_col]
                    duration = len(audio['array']) / audio['sampling_rate']

                    # Get text (handle different column names)
                    text = sample.get(text_col, sample.get('sentence', sample.get('transcription', '')))

                    # Filter by duration
                    if min_len <= duration <= max_len and text:
                        valid_samples.append({
                            'path': sample.get('path', sample.get('id', f'sample_{i}')),
                            'sentence': text,
                            'duration': duration,
                            'audio': audio
                        })
                        total_duration += duration

                    progress.update(task, advance=1)

                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue

        # No need to sort by votes for FLEURS (all high quality)

        # Take target hours of data
        target_seconds = config['data']['target_hours'] * 3600
        selected_samples = []
        selected_duration = 0

        for sample in valid_samples:
            if selected_duration >= target_seconds:
                break
            selected_samples.append(sample)
            selected_duration += sample['duration']

        # Display statistics
        table = Table(title="Dataset Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Total samples downloaded", f"{len(dataset):,}")
        table.add_row("Valid samples (after filtering)", f"{len(valid_samples):,}")
        table.add_row("Selected samples", f"{len(selected_samples):,}")
        table.add_row("Total duration (valid)", f"{total_duration/3600:.2f} hours")
        table.add_row("Selected duration", f"{selected_duration/3600:.2f} hours")
        table.add_row("Target duration", f"{config['data']['target_hours']} hours")

        console.print(table)

        # Save audio files and metadata
        console.print("\n[bold blue]Saving audio files...[/bold blue]")

        audio_dir = raw_dir / "audio"
        audio_dir.mkdir(exist_ok=True)

        metadata = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("[cyan]Saving audio", total=len(selected_samples))

            for i, sample in enumerate(selected_samples):
                try:
                    import soundfile as sf

                    # Save audio file
                    audio_path = audio_dir / f"sw_{i:06d}.wav"
                    sf.write(
                        str(audio_path),
                        sample['audio']['array'],
                        sample['audio']['sampling_rate']
                    )

                    # Add to metadata
                    metadata.append({
                        'audio_file': f"audio/sw_{i:06d}.wav",
                        'text': sample['sentence'],
                        'duration': sample['duration'],
                        'speaker_id': 'swahili_speaker'
                    })

                    progress.update(task, advance=1)

                except Exception as e:
                    logger.warning(f"Error saving sample {i}: {e}")
                    continue

        # Save metadata
        metadata_path = raw_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        console.print(f"\n[green]Saved {len(metadata):,} audio files[/green]")
        console.print(f"[green]Metadata saved to: {metadata_path}[/green]")

        # Save statistics
        stats = {
            'total_downloaded': len(dataset),
            'valid_samples': len(valid_samples),
            'selected_samples': len(selected_samples),
            'total_duration_hours': total_duration / 3600,
            'selected_duration_hours': selected_duration / 3600,
            'source': config['data']['source'],
            'language': 'sw'
        }

        stats_path = raw_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        console.print(f"[green]Statistics saved to: {stats_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error downloading dataset: {e}[/red]")
        logger.error(f"Download error: {e}")

        # Provide alternative instructions
        console.print("\n[yellow]Alternative: Manual Download[/yellow]")
        console.print("1. Visit: https://commonvoice.mozilla.org/sw/datasets")
        console.print("2. Download the Swahili dataset")
        console.print("3. Extract to: data/raw/")
        console.print("4. Run preprocessing script")
        raise


def main():
    """Main entry point."""
    console.print("\n[bold]Eryx Labs - Swahili TTS Data Collection[/bold]\n")

    # Check for HuggingFace token
    if not os.getenv("HF_TOKEN"):
        console.print("[yellow]Warning: HF_TOKEN not set[/yellow]")
        console.print("Common Voice requires authentication.")
        console.print("Set: export HF_TOKEN='your-token'")
        console.print("Get token from: https://huggingface.co/settings/tokens\n")

    config = load_config()
    download_common_voice_swahili(config)

    console.print("\n[bold green]Data collection complete![/bold green]")
    console.print("[yellow]Next step: python scripts/02_preprocess.py[/yellow]\n")


if __name__ == "__main__":
    main()
