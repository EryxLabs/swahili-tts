# Eryx Labs - Swahili TTS

Natural Swahili Text-to-Speech using XTTS-v2 fine-tuning.

## Overview

This project creates a Swahili TTS system by fine-tuning XTTS-v2 on Mozilla Common Voice Swahili data. Combined with our [Swahili LLM](../swahili-llm/), it enables complete voice-enabled conversational AI in Swahili.

## Features

- Natural Swahili voice synthesis
- Voice cloning capabilities
- ~10 hours of training data from Common Voice
- Interactive synthesis mode

## Quick Start

### Prerequisites

- Python 3.10+
- HuggingFace account (for Common Voice access)
- GPU recommended (NVIDIA or Apple Silicon)

### Installation

```bash
cd swahili-tts

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token (for Common Voice)
export HF_TOKEN='your-token-here'
```

### Run the Pipeline

```bash
# Step 1: Download Swahili voice data
python scripts/01_download_data.py

# Step 2: Preprocess audio
python scripts/02_preprocess.py

# Step 3: Fine-tune XTTS-v2
python scripts/03_train.py

# Step 4: Synthesize speech
python scripts/04_synthesize.py
```

### Interactive Mode

```bash
python scripts/04_synthesize.py --interactive
```

## Project Structure

```
swahili-tts/
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
├── README.md
├── scripts/
│   ├── 01_download_data.py  # Download Common Voice
│   ├── 02_preprocess.py     # Audio preprocessing
│   ├── 03_train.py          # Fine-tune XTTS-v2
│   └── 04_synthesize.py     # Generate speech
├── data/
│   ├── raw/                 # Downloaded audio
│   └── processed/           # Training-ready data
├── checkpoints/             # Model checkpoints
└── models/                  # Exported models
```

## Data Source

**Mozilla Common Voice Swahili**
- ~15 hours of validated Swahili speech
- ~6,000+ voice recordings
- Multiple speakers
- Natural conversational style

## Model Details

| Parameter | Value |
|-----------|-------|
| Base Model | XTTS-v2 |
| Training Data | ~10 hours |
| Sample Rate | 22050 Hz |
| Language | Swahili (sw) |

## Integration with Swahili LLM

Combine TTS with our Swahili LLM for voice-enabled AI:

```python
from swahili_llm import generate_response
from swahili_tts import synthesize_speech

# Get LLM response
text_response = generate_response("Habari, jinsi ya kuanzisha biashara?")

# Convert to speech
synthesize_speech(text_response, "response.wav")
```

## License

Apache 2.0

## About Eryx Labs

[Eryx Labs](https://eryxlabs.co.ke) builds AI solutions for African languages.

---

*Tunasema Kiswahili!*
