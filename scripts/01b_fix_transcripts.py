#!/usr/bin/env python3
"""
Eryx Labs - Fix Swahili TTS Transcripts
Re-process the OpenSLR data with correct transcript mapping
"""

import json
import os
import re
import shutil
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

console = Console()

Path("logs").mkdir(exist_ok=True)
logger.add("logs/fix_transcripts_{time}.log", rotation="50 MB", level="INFO")


def main():
    """Main entry point."""
    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]ERYX LABS - FIX SWAHILI TTS TRANSCRIPTS[/bold cyan]")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]\n")

    extract_dir = Path("data/downloads/extracted/data_broadcastnews_sw")
    raw_dir = Path("data/raw")

    if not extract_dir.exists():
        console.print("[red]Error: Extracted data not found![/red]")
        return

    # Load transcripts from both train and test
    transcripts = {}

    # Train text file (Kaldi format: utterance_id<tab>text)
    train_text = extract_dir / "data/train/text"
    if train_text.exists():
        console.print(f"[cyan]Loading train transcripts from {train_text}[/cyan]")
        with open(train_text, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    # Store original key
                    transcripts[utt_id] = text
                    # Extract the key part of the utterance ID for matching
                    # Format: SWH-05-20101106_16k-emission_swahili_...
                    key = utt_id.split('_', 1)[-1] if '_' in utt_id else utt_id
                    transcripts[key] = text
                    # Also try with g- or m- prefix removed
                    if key.startswith(('g-', 'm-')):
                        transcripts[key[2:]] = text
                    # Extract partXXX for flexible matching
                    part_match = re.search(r'(part\d+[a-z]?)', utt_id)
                    if part_match:
                        # Create normalized key: emission_type + part
                        norm_key = re.sub(r'_\d{8}_', '_DATE_', utt_id)  # Replace date
                        transcripts[norm_key] = text
        console.print(f"[green]Loaded {len(transcripts)} train transcripts[/green]")

    # Test transcription file (format: text (audio_id))
    test_transcription = extract_dir / "data/test/swahili_test.transcription"
    if test_transcription.exists():
        console.print(f"[cyan]Loading test transcripts from {test_transcription}[/cyan]")
        count = 0
        with open(test_transcription, 'r', encoding='utf-8') as f:
            for line in f:
                # Parse: text (audio_id)
                match = re.match(r'^(.+)\s+\(([^)]+)\)$', line.strip())
                if match:
                    text, audio_id = match.groups()
                    # Remove leading g- or m-
                    if audio_id.startswith(('g-', 'm-')):
                        key = audio_id[2:]
                    else:
                        key = audio_id
                    transcripts[key] = text.strip()
                    count += 1
        console.print(f"[green]Loaded {count} test transcripts[/green]")

    console.print(f"\n[bold]Total transcripts loaded: {len(transcripts)}[/bold]\n")

    # Now re-process the audio files with proper transcripts
    audio_dir = raw_dir / "audio"
    if not audio_dir.exists():
        console.print("[red]Error: Audio directory not found![/red]")
        return

    # Load existing metadata
    old_metadata_path = raw_dir / "metadata.json"
    with open(old_metadata_path, 'r', encoding='utf-8') as f:
        old_metadata = json.load(f)

    console.print(f"[cyan]Processing {len(old_metadata)} audio files...[/cyan]")

    new_metadata = []
    matched = 0
    unmatched = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[cyan]Matching transcripts", total=len(old_metadata))

        for sample in old_metadata:
            original_file = sample.get('original_file', '')

            # Try to find matching transcript
            text = None

            # Try different key formats
            keys_to_try = []

            # Original filename without extension
            base_name = original_file.replace('.wav', '').replace('.flac', '')
            keys_to_try.append(base_name)

            # Without 16k- prefix
            if base_name.startswith('16k-'):
                no_16k = base_name[4:]
                keys_to_try.append(no_16k)
                # Also with SWH prefix variants
                keys_to_try.append(f"SWH-05-{base_name}")

            # Extract part name (e.g., emission_swahili_...part001)
            if 'part' in base_name:
                part_match = re.search(r'(emission_swahili_.*part\d+[a-z]?)', base_name)
                if part_match:
                    keys_to_try.append(part_match.group(1))

                # Also try date-normalized version
                norm_key = re.sub(r'_\d{8}_', '_DATE_', base_name)
                if norm_key.startswith('16k-'):
                    norm_key = f"SWH-05-DATE_{norm_key[4:]}"
                else:
                    norm_key = f"SWH-05-DATE_{norm_key}"
                keys_to_try.append(norm_key)

                # Try just the part suffix matching
                # Audio: 16k-emission_swahili_05h30_-_06h00_tu_20101124_part185g
                # Trans: SWH-05-20101106_16k-emission_swahili_05h30_-_06h00_tu_20101106_part10
                part_num = re.search(r'part(\d+[a-z]?)', base_name)
                if part_num:
                    part_id = part_num.group(1)
                    # Look for any transcript with same part number
                    for k, v in transcripts.items():
                        if f'part{part_id}' in k and 'emission_swahili' in k:
                            keys_to_try.append(k)
                            break

            for key in keys_to_try:
                if key in transcripts:
                    text = transcripts[key]
                    break

            if text:
                matched += 1
            else:
                # Keep placeholder for files without transcripts
                text = f"[No transcript: {original_file[:50]}]"
                unmatched += 1

            new_metadata.append({
                'audio_file': sample['audio_file'],
                'text': text,
                'duration': sample['duration'],
                'speaker_id': sample['speaker_id'],
                'original_file': original_file
            })

            progress.update(task, advance=1)

    # Filter out samples without transcripts for TTS training
    tts_metadata = [s for s in new_metadata if not s['text'].startswith('[No transcript')]

    # Save updated metadata
    with open(raw_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(new_metadata, f, ensure_ascii=False, indent=2)

    # Save TTS-ready metadata (only with transcripts)
    with open(raw_dir / "metadata_tts.json", 'w', encoding='utf-8') as f:
        json.dump(tts_metadata, f, ensure_ascii=False, indent=2)

    # Calculate stats
    total_duration = sum(s['duration'] for s in tts_metadata)

    # Update stats
    stats = {
        'total_samples': len(new_metadata),
        'samples_with_transcripts': len(tts_metadata),
        'samples_without_transcripts': len(new_metadata) - len(tts_metadata),
        'total_duration_hours': total_duration / 3600,
        'source': 'OpenSLR 25 - Swahili Speech Corpus'
    }

    with open(raw_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Display summary
    table = Table(title="Transcript Matching Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total audio files", f"{len(new_metadata):,}")
    table.add_row("Matched transcripts", f"{matched:,}")
    table.add_row("Unmatched (placeholders)", f"{unmatched:,}")
    table.add_row("TTS-ready samples", f"{len(tts_metadata):,}")
    table.add_row("TTS duration", f"{total_duration/3600:.2f} hours")

    console.print("\n")
    console.print(table)

    # Show sample transcripts
    console.print("\n[bold]Sample Swahili Transcripts:[/bold]")
    for i, sample in enumerate(tts_metadata[:5]):
        console.print(f"  {i+1}. [dim]{sample['text'][:80]}...[/dim]")

    console.print("\n[bold green]Transcript matching complete![/bold green]")
    console.print("[yellow]Next: Re-run python scripts/02_preprocess.py[/yellow]\n")


if __name__ == "__main__":
    main()
