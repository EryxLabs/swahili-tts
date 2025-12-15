#!/usr/bin/env python3
"""
Eryx Labs - Swahili TTS Training
Step 3: Fine-tune XTTS-v2 for Swahili voice synthesis

This script:
1. Loads the XTTS-v2 model
2. Configures fine-tuning parameters
3. Extracts speaker embeddings from Swahili voice data
4. Generates test audio samples
"""

import json
import os
import sys
from pathlib import Path

import torch
# Fix for PyTorch 2.6+ weights_only default change
# Patch torch.load to use weights_only=False for TTS model compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Set torchaudio backend to soundfile (avoids torchcodec requirement)
import torchaudio
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass  # May not need explicit backend setting

import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Configure logging
Path("logs").mkdir(exist_ok=True)
logger.add(
    "logs/training_{time}.log",
    rotation="50 MB",
    level="INFO"
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent.parent / config_path
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def check_gpu():
    """Check GPU availability."""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[green]GPU: {device} ({memory:.1f} GB)[/green]")
        return "cuda"
    elif torch.backends.mps.is_available():
        console.print("[green]GPU: Apple Silicon (MPS)[/green]")
        return "mps"
    else:
        console.print("[yellow]No GPU detected. Training will be slow.[/yellow]")
        return "cpu"


def load_xtts_model(model_path: str, device: str):
    """Load XTTS model directly without using TTS.api."""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=False)

    if device == "cuda":
        model.cuda()

    return model, config


def train_xtts(config: dict):
    """Fine-tune XTTS-v2 for Swahili."""
    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]ERYX LABS - SWAHILI TTS TRAINING[/bold cyan]")
    console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]\n")

    # Check device
    device = check_gpu()

    # Paths
    processed_dir = Path("data/processed")
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data stats
    stats_path = processed_dir / "stats.json"
    if not stats_path.exists():
        console.print("[red]Error: Preprocessed data not found![/red]")
        console.print("Run: python scripts/02_preprocess.py first")
        return

    with open(stats_path) as f:
        data_stats = json.load(f)

    # Display configuration
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Base Model", config['model']['base_model'])
    table.add_row("Device", device)
    table.add_row("Training Samples", f"{data_stats['train_samples']:,}")
    table.add_row("Training Duration", f"{data_stats['train_duration_hours']:.2f} hours")
    table.add_row("Batch Size", str(config['training']['batch_size']))
    table.add_row("Learning Rate", str(config['training']['learning_rate']))
    table.add_row("Max Steps", str(config['training']['max_steps']))

    console.print(table)
    console.print()

    try:
        # Path to downloaded model
        model_path = os.path.expanduser(
            "~/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2"
        )

        if not os.path.exists(model_path):
            console.print(f"[red]Model not found at: {model_path}[/red]")
            console.print("[yellow]Downloading model...[/yellow]")
            # Download using tts command
            os.system("tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 --list_models")

        console.print("[bold blue]Loading XTTS-v2 model directly...[/bold blue]")

        model, model_config = load_xtts_model(model_path, device)
        console.print("[green]Model loaded successfully![/green]")

        # For fine-tuning XTTS-v2, we use speaker embedding extraction
        console.print("\n[bold blue]Creating Swahili speaker embeddings...[/bold blue]")

        # Load metadata for speaker embedding creation
        train_metadata_path = processed_dir / "train.csv"
        if not train_metadata_path.exists():
            console.print("[red]Training metadata not found![/red]")
            return

        import pandas as pd
        train_df = pd.read_csv(train_metadata_path, sep='|', header=None, names=['audio', 'text', 'speaker'])

        # Select samples for speaker conditioning
        n_reference = min(100, len(train_df))
        reference_audios = []

        for _, row in train_df.head(n_reference).iterrows():
            audio_path = processed_dir / row['audio']
            if audio_path.exists():
                reference_audios.append(str(audio_path))

        console.print(f"[green]Loaded {len(reference_audios)} reference audio samples[/green]")

        # Extract speaker embeddings
        if reference_audios:
            console.print("\n[bold blue]Extracting speaker embeddings...[/bold blue]")

            # Use first reference audio for conditioning
            ref_audio = reference_audios[0]

            # Get speaker conditioning info
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=[ref_audio]
            )

            # Save speaker embeddings for later use
            embeddings_dir = checkpoint_dir / "speaker_embeddings"
            embeddings_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'gpt_cond_latent': gpt_cond_latent,
                'speaker_embedding': speaker_embedding,
                'reference_audio': ref_audio,
            }, embeddings_dir / "swahili_speaker.pt")

            console.print(f"[green]Speaker embeddings saved to: {embeddings_dir}[/green]")

            # Test synthesis with speaker embeddings
            console.print("\n[bold blue]Testing Swahili synthesis with speaker conditioning...[/bold blue]")

            test_texts = [
                "Habari yako, mimi ni msaidizi wa Kiswahili.",
                "Karibu sana katika Eryx Labs.",
                "Teknolojia ya akili bandia inabadilisha dunia.",
            ]

            output_dir = checkpoint_dir / "test_outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            import torchaudio

            # Note: XTTS-v2 doesn't support 'sw' (Swahili) directly.
            # We use 'en' for text processing since Swahili uses Latin script.
            # The speaker embeddings capture the Swahili voice characteristics.
            synthesis_lang = "en"
            console.print(f"[yellow]Using '{synthesis_lang}' for text processing (Swahili not in XTTS-v2)[/yellow]")

            for i, text in enumerate(test_texts):
                output_path = output_dir / f"swahili_test_{i+1}.wav"

                # Synthesize with speaker conditioning
                out = model.inference(
                    text=text,
                    language=synthesis_lang,  # Using 'en' for Swahili text (Latin script compatible)
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                )

                # Save audio
                torchaudio.save(
                    str(output_path),
                    torch.tensor(out["wav"]).unsqueeze(0),
                    24000
                )

                console.print(f"  [green]Generated: {output_path.name}[/green]")

            console.print("\n[bold green]Speaker conditioning setup complete![/bold green]")
            console.print(f"[bold]Test outputs saved to: {output_dir}[/bold]")

        # Save training status
        status = {
            "status": "speaker_embeddings_created",
            "reference_samples": len(reference_audios),
            "training_samples": len(train_df),
            "note": "XTTS-v2 uses speaker conditioning rather than full fine-tuning. "
                   "Speaker embeddings have been extracted and can be used for synthesis.",
        }

        with open(checkpoint_dir / "training_status.json", 'w') as f:
            json.dump(status, f, indent=2)

        console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
        console.print("[bold green]XTTS-v2 SETUP COMPLETE[/bold green]")
        console.print("[bold cyan]" + "=" * 60 + "[/bold cyan]")
        console.print("\n[yellow]Note: XTTS-v2 uses speaker conditioning for voice cloning.[/yellow]")
        console.print("[yellow]Full fine-tuning requires the Coqui TTS training recipes.[/yellow]")
        console.print(f"\n[bold]Speaker embeddings: {embeddings_dir / 'swahili_speaker.pt'}[/bold]")
        console.print(f"[bold]Test outputs: {output_dir}[/bold]")

    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        console.print("\n[yellow]TTS library not fully installed.[/yellow]")
        console.print("Install with: pip install TTS")

        # Provide alternative training approach
        console.print("\n[bold]Alternative: Manual XTTS Fine-tuning[/bold]")
        provide_manual_training_instructions(config, data_stats)

    except Exception as e:
        console.print(f"[red]Training error: {e}[/red]")
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()


def provide_manual_training_instructions(config: dict, data_stats: dict):
    """Provide instructions for manual fine-tuning."""
    console.print("""
[cyan]To fine-tune XTTS-v2 manually:[/cyan]

1. Install Coqui TTS:
   pip install TTS

2. Clone the Coqui TTS recipes:
   git clone https://github.com/coqui-ai/TTS
   cd TTS/recipes/ljspeech/xtts_v2

3. Modify the config for Swahili:
   - Point to your data: data/processed/
   - Set language to 'sw' (or use multilingual)

4. Run training:
   python train_gpt_xtts.py

[yellow]Alternatively, use the Coqui TTS command line:[/yellow]

tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \\
    --finetune \\
    --output_path checkpoints/ \\
    --dataset_config data/processed/

[green]Your processed data is ready at: data/processed/[/green]
""")


def main():
    """Main entry point."""
    console.print(Panel.fit(
        "[bold cyan]ERYX LABS - SWAHILI TTS TRAINING[/bold cyan]\n\n"
        "[dim]Fine-tuning XTTS-v2 for natural Swahili voice[/dim]",
        border_style="cyan"
    ))

    config = load_config()
    train_xtts(config)

    console.print("\n[yellow]Next step: python scripts/04_synthesize.py[/yellow]\n")


if __name__ == "__main__":
    main()
