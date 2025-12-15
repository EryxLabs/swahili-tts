#!/usr/bin/env python3
"""
Eryx Labs - Swahili TTS Data Collection via OpenSLR
Downloads the Swahili speech corpus from OpenSLR (SLR25)

This is a public domain dataset with ~10 hours of Swahili speech.
"""

import json
import os
import tarfile
import urllib.request
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn
from rich.table import Table

console = Console()

# Configure logging
Path("logs").mkdir(exist_ok=True)
logger.add(
    "logs/download_{time}.log",
    rotation="50 MB",
    level="INFO"
)

# OpenSLR Swahili dataset URL (SLR25 - Swahili Broadcast News)
# Using EU mirror for reliability
OPENSLR_URL = "https://openslr.elda.org/resources/25/data_broadcastnews_sw.tar.bz2"


def download_with_progress(url: str, output_path: Path) -> bool:
    """Download a file with progress bar."""
    try:
        console.print(f"[yellow]Downloading from: {url}[/yellow]")

        # Get file size
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('content-length', 0))

        # Download with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Downloading", total=total_size)

            with open(output_path, 'wb') as f:
                block_size = 8192
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

        console.print(f"[green]Downloaded: {output_path}[/green]")
        return True

    except Exception as e:
        console.print(f"[red]Download error: {e}[/red]")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract tar.bz2 archive."""
    try:
        console.print(f"[yellow]Extracting archive (this may take a few minutes)...[/yellow]")

        # Determine compression type from extension
        if str(archive_path).endswith('.tar.bz2'):
            mode = 'r:bz2'
        elif str(archive_path).endswith('.tar.gz'):
            mode = 'r:gz'
        else:
            mode = 'r:*'  # Auto-detect

        with tarfile.open(archive_path, mode) as tar:
            tar.extractall(path=extract_to)

        console.print(f"[green]Extracted to: {extract_to}[/green]")
        return True

    except Exception as e:
        console.print(f"[red]Extraction error: {e}[/red]")
        return False


def process_openslr_data(data_dir: Path) -> dict:
    """Process the extracted OpenSLR Swahili data."""
    console.print("[yellow]Processing Swahili speech data...[/yellow]")

    # The OpenSLR 25 dataset structure:
    # swahili/
    #   ├── test/
    #   ├── train/
    #   └── validated.txt (or similar)

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = raw_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    metadata = []
    total_duration = 0

    # Look for audio files and transcripts
    swahili_dir = data_dir / "swahili"
    if not swahili_dir.exists():
        # Try to find any wav files
        swahili_dir = data_dir

    # Find all wav files
    wav_files = list(swahili_dir.rglob("*.wav")) + list(swahili_dir.rglob("*.flac"))
    console.print(f"[cyan]Found {len(wav_files)} audio files[/cyan]")

    # Look for transcript files
    transcript_files = list(swahili_dir.rglob("*.txt")) + list(swahili_dir.rglob("*.tsv"))

    # Build transcript mapping
    transcripts = {}
    for tf in transcript_files:
        try:
            with open(tf, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        file_id = parts[0].replace('.wav', '').replace('.flac', '')
                        text = parts[1] if len(parts) > 1 else parts[0]
                        transcripts[file_id] = text
        except:
            continue

    console.print(f"[cyan]Found {len(transcripts)} transcripts[/cyan]")

    # Process audio files
    import shutil

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[cyan]Processing files", total=len(wav_files))

        for i, wav_path in enumerate(wav_files):
            try:
                # Get file ID
                file_id = wav_path.stem

                # Get transcript
                text = transcripts.get(file_id, f"Sample {i}")

                # Copy audio file
                new_name = f"sw_{i:06d}.wav"
                dest_path = audio_dir / new_name
                shutil.copy2(wav_path, dest_path)

                # Get duration (rough estimate from file size)
                file_size = wav_path.stat().st_size
                # Assuming 16kHz, 16-bit mono
                duration = file_size / (16000 * 2)

                metadata.append({
                    'audio_file': f"audio/{new_name}",
                    'text': text,
                    'duration': duration,
                    'speaker_id': 'swahili_speaker',
                    'original_file': str(wav_path.name)
                })

                total_duration += duration
                progress.update(task, advance=1)

            except Exception as e:
                logger.warning(f"Error processing {wav_path}: {e}")
                continue

    # Save metadata
    metadata_path = raw_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    stats = {
        'total_samples': len(metadata),
        'total_duration_hours': total_duration / 3600,
        'source': 'OpenSLR 25 - Swahili Speech Corpus'
    }

    stats_path = raw_dir / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    """Main entry point."""
    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]ERYX LABS - SWAHILI TTS DATA COLLECTION[/bold cyan]")
    console.print("[bold cyan]OpenSLR 25 - Swahili Speech Corpus[/bold cyan]")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]\n")

    # Create directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    download_dir = data_dir / "downloads"
    download_dir.mkdir(exist_ok=True)

    # Download the archive
    archive_path = download_dir / "data_broadcastnews_sw.tar.bz2"

    if not archive_path.exists():
        console.print("[bold blue]Downloading OpenSLR Swahili dataset...[/bold blue]")
        console.print("[dim]This is a ~500MB download[/dim]\n")

        if not download_with_progress(OPENSLR_URL, archive_path):
            console.print("[red]Download failed![/red]")
            return
    else:
        console.print(f"[green]Archive already exists: {archive_path}[/green]")

    # Extract the archive
    extract_dir = download_dir / "extracted"
    if not extract_dir.exists():
        extract_dir.mkdir()
        if not extract_archive(archive_path, extract_dir):
            console.print("[red]Extraction failed![/red]")
            return

    # Process the data
    stats = process_openslr_data(extract_dir)

    # Display summary
    table = Table(title="Download Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Source", stats['source'])
    table.add_row("Total Samples", f"{stats['total_samples']:,}")
    table.add_row("Total Duration", f"{stats['total_duration_hours']:.2f} hours")

    console.print("\n")
    console.print(table)

    console.print("\n[bold green]Data collection complete![/bold green]")
    console.print("[yellow]Next step: python scripts/02_preprocess.py[/yellow]\n")


if __name__ == "__main__":
    main()
