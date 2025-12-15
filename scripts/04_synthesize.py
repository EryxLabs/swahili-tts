#!/usr/bin/env python3
"""
Eryx Labs - Swahili TTS Synthesis
Step 4: Generate speech from text using the fine-tuned model

This script:
1. Loads the fine-tuned XTTS model
2. Synthesizes speech from Swahili text
3. Saves audio files
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel

console = Console()


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent.parent / config_path
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def synthesize_speech(
    text: str,
    output_path: str,
    model_path: str = None,
    reference_audio: str = None,
    language: str = "sw"
):
    """Synthesize speech from text."""
    try:
        from TTS.api import TTS

        # Initialize TTS
        if model_path and Path(model_path).exists():
            # Use fine-tuned model
            tts = TTS(model_path=model_path)
            console.print(f"[green]Using fine-tuned model: {model_path}[/green]")
        else:
            # Use base XTTS-v2 (multilingual)
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            console.print("[yellow]Using base XTTS-v2 (not fine-tuned)[/yellow]")

        # Synthesize with voice cloning if reference provided
        if reference_audio and Path(reference_audio).exists():
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=reference_audio,
                language=language
            )
        else:
            # Use default speaker
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                language=language
            )

        console.print(f"[green]Audio saved to: {output_path}[/green]")
        return output_path

    except Exception as e:
        console.print(f"[red]Synthesis error: {e}[/red]")
        logger.error(f"Synthesis error: {e}")
        return None


def batch_synthesize(config: dict):
    """Synthesize all test sentences."""
    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]ERYX LABS - SWAHILI TTS SYNTHESIS[/bold cyan]")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]\n")

    output_dir = Path("models/synthesized")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for fine-tuned model
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    model_path = None

    # Look for latest checkpoint
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("**/best_model.pth"))
        if checkpoints:
            model_path = str(checkpoints[0].parent)
            console.print(f"[green]Found fine-tuned model: {model_path}[/green]")

    # Reference audio for voice cloning
    reference_audio = config['evaluation'].get('reference_audio')

    # Synthesize test sentences
    test_sentences = config['evaluation']['test_sentences']

    console.print(f"\n[bold blue]Synthesizing {len(test_sentences)} test sentences...[/bold blue]\n")

    results = []
    for i, text in enumerate(test_sentences):
        console.print(f"[cyan]{i+1}. {text}[/cyan]")

        output_path = str(output_dir / f"test_{i+1:02d}.wav")
        result = synthesize_speech(
            text=text,
            output_path=output_path,
            model_path=model_path,
            reference_audio=reference_audio,
            language="sw"
        )

        if result:
            results.append({
                'text': text,
                'audio': output_path
            })
        console.print()

    # Save results
    with open(output_dir / "synthesis_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    console.print(f"\n[bold green]Synthesized {len(results)} audio files![/bold green]")
    console.print(f"[green]Output directory: {output_dir}[/green]")


def interactive_mode(config: dict):
    """Interactive synthesis mode."""
    console.print(Panel.fit(
        "[bold cyan]ERYX LABS - SWAHILI TTS[/bold cyan]\n"
        "[dim]Interactive voice synthesis[/dim]",
        border_style="cyan"
    ))

    console.print("\n[yellow]Type Swahili text to synthesize (or 'quit' to exit)[/yellow]\n")

    output_dir = Path("models/synthesized/interactive")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for fine-tuned model
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    model_path = None

    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("**/best_model.pth"))
        if checkpoints:
            model_path = str(checkpoints[0].parent)

    count = 0
    while True:
        try:
            text = console.input("[bold green]>> [/bold green]").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Kwaheri! (Goodbye!)[/yellow]")
                break

            if not text:
                continue

            count += 1
            output_path = str(output_dir / f"interactive_{count:03d}.wav")

            synthesize_speech(
                text=text,
                output_path=output_path,
                model_path=model_path,
                language="sw"
            )

            # Play audio (if available)
            try:
                import subprocess
                if os.uname().sysname == "Darwin":  # macOS
                    subprocess.run(["afplay", output_path], capture_output=True)
                elif os.name == "posix":  # Linux
                    subprocess.run(["aplay", output_path], capture_output=True)
            except:
                pass  # Audio playback not available

        except KeyboardInterrupt:
            console.print("\n[yellow]Kwaheri! (Goodbye!)[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Swahili TTS Synthesis")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--text", "-t", type=str,
                       help="Text to synthesize")
    parser.add_argument("--output", "-o", type=str, default="output.wav",
                       help="Output audio file path")
    parser.add_argument("--reference", "-r", type=str,
                       help="Reference audio for voice cloning")

    args = parser.parse_args()

    config = load_config()

    if args.interactive:
        interactive_mode(config)
    elif args.text:
        synthesize_speech(
            text=args.text,
            output_path=args.output,
            reference_audio=args.reference,
            language="sw"
        )
    else:
        batch_synthesize(config)


if __name__ == "__main__":
    main()
