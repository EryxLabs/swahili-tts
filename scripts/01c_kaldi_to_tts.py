#!/usr/bin/env python3
"""
Eryx Labs - Convert Kaldi ASR format to TTS format
Properly processes the OpenSLR Swahili data using Kaldi format files
"""

import json
import os
import shutil
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

console = Console()

Path("logs").mkdir(exist_ok=True)
logger.add("logs/kaldi_to_tts_{time}.log", rotation="50 MB", level="INFO")


def parse_kaldi_text(text_file: Path) -> dict:
    """Parse Kaldi text file to get utterance_id -> text mapping."""
    transcripts = {}
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                utt_id, text = parts
                transcripts[utt_id] = text
            elif len(parts) == 1 and ' ' in parts[0]:
                # Try space separator
                first_space = parts[0].index(' ')
                utt_id = parts[0][:first_space]
                text = parts[0][first_space+1:]
                transcripts[utt_id] = text
    return transcripts


def parse_kaldi_wavscp(wavscp_file: Path) -> dict:
    """Parse Kaldi wav.scp file to get utterance_id -> wav_path mapping."""
    wav_paths = {}
    with open(wavscp_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                utt_id, wav_path = parts
                wav_paths[utt_id] = wav_path
    return wav_paths


def main():
    """Main entry point."""
    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]ERYX LABS - KALDI TO TTS CONVERSION[/bold cyan]")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]\n")

    extract_dir = Path("data/downloads/extracted/data_broadcastnews_sw")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = raw_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    if not extract_dir.exists():
        console.print("[red]Error: Extracted data not found![/red]")
        return

    # Find all wav files in the extracted directory
    console.print("[cyan]Scanning for audio files...[/cyan]")
    all_wavs = {}
    for wav_file in extract_dir.rglob("*.wav"):
        # Index by filename (without extension)
        key = wav_file.stem
        all_wavs[key] = wav_file
    console.print(f"[green]Found {len(all_wavs)} audio files[/green]")

    # Process train and test sets
    metadata = []
    total_duration = 0

    for split in ['train', 'test']:
        split_dir = extract_dir / "data" / split
        if not split_dir.exists():
            console.print(f"[yellow]Split {split} not found, skipping[/yellow]")
            continue

        console.print(f"\n[bold blue]Processing {split} split...[/bold blue]")

        # Load transcripts
        text_file = split_dir / "text"
        if not text_file.exists():
            console.print(f"[red]No text file for {split}[/red]")
            continue

        transcripts = parse_kaldi_text(text_file)
        console.print(f"  Loaded {len(transcripts)} transcripts")

        # For each transcript, find matching audio
        matched = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task(f"[cyan]Processing {split}", total=len(transcripts))

            for utt_id, text in transcripts.items():
                # Skip music markers and short texts
                if text.strip() in ['<music>', '<noise>'] or len(text.strip()) < 3:
                    progress.update(task, advance=1)
                    continue

                # Try to find matching audio
                # Format: SWH-05-20101106_16k-emission_swahili_...
                # Audio file format: 16k-emission_swahili_... or SWH-05-DATE_16k-...
                audio_path = None

                # Try exact match with utt_id
                if utt_id in all_wavs:
                    audio_path = all_wavs[utt_id]
                else:
                    # Extract the 16k- part from utt_id
                    # SWH-05-20101106_16k-emission_swahili_05h30_-_06h00_tu_20101106_part10
                    if '_16k-' in utt_id:
                        audio_name = utt_id.split('_16k-', 1)[1]
                        audio_name = '16k-' + audio_name
                        if audio_name in all_wavs:
                            audio_path = all_wavs[audio_name]

                if audio_path and audio_path.exists():
                    matched += 1

                    # Copy audio to raw dir
                    new_name = f"sw_{len(metadata):06d}.wav"
                    dest_path = audio_dir / new_name
                    shutil.copy2(audio_path, dest_path)

                    # Get duration
                    file_size = audio_path.stat().st_size
                    duration = file_size / (16000 * 2)  # 16kHz, 16-bit
                    total_duration += duration

                    metadata.append({
                        'audio_file': f"audio/{new_name}",
                        'text': text.strip(),
                        'duration': duration,
                        'speaker_id': 'swahili_speaker',
                        'original_utt_id': utt_id
                    })

                progress.update(task, advance=1)

        console.print(f"  [green]Matched {matched} samples[/green]")

    # Save metadata
    with open(raw_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save stats
    stats = {
        'total_samples': len(metadata),
        'total_duration_hours': total_duration / 3600,
        'source': 'OpenSLR 25 - Swahili Speech Corpus (Kaldi format)'
    }
    with open(raw_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Display summary
    table = Table(title="Kaldi to TTS Conversion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total samples", f"{len(metadata):,}")
    table.add_row("Total duration", f"{total_duration/3600:.2f} hours")
    table.add_row("Average duration", f"{(total_duration/len(metadata) if metadata else 0):.2f} sec")

    console.print("\n")
    console.print(table)

    # Show sample transcripts
    console.print("\n[bold]Sample Swahili Transcripts:[/bold]")
    for i, sample in enumerate(metadata[:5]):
        console.print(f"  {i+1}. [dim]{sample['text'][:80]}...[/dim]")

    console.print("\n[bold green]Conversion complete![/bold green]")
    console.print("[yellow]Next: python scripts/02_preprocess.py[/yellow]\n")


if __name__ == "__main__":
    main()
