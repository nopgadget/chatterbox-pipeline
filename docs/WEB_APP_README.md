# Chatterbox Turbo TTS Web Application

High-performance FastAPI web application with both web UI and REST API endpoints for Chatterbox Turbo TTS.

## Features

- 🚀 **High Performance**: FastAPI with async support
- 🎨 **Modern Web UI**: Beautiful, responsive interface
- 🔌 **REST API**: Full API access with automatic documentation
- 🎭 **Paralinguistic Tags**: Support for [cough], [laugh], [chuckle], etc.
- ⚙️ **Configurable**: All generation parameters available via UI and API
- 📤 **File Upload**: Upload reference audio files via web UI or API

## Installation

1. Install web app dependencies:
```bash
pip install -r requirements_web.txt
```

2. Make sure you have all Chatterbox dependencies installed (from `pyproject.toml`)

## Running the Application

### Basic Usage
```bash
python app.py
```

The application will start on `http://localhost:8000`

### With Custom Options
```bash
python app.py --host 0.0.0.0 --port 8000 --workers 1
```

### Using Uvicorn Directly
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

## Usage

### Web UI

1. Open your browser to `http://localhost:8000`
2. Enter your text (with optional paralinguistic tags)
3. Optionally upload a reference audio file
4. Adjust parameters in "Advanced Options"
5. Click "Generate ⚡"

### API Endpoints

#### 1. Generate TTS (JSON)
```bash
POST /api/tts
Content-Type: application/json

{
  "text": "Hello world [chuckle]",
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 1000,
  "repetition_penalty": 1.2,
  "min_p": 0.0,
  "norm_loudness": true,
  "seed": 42,
  "audio_prompt_path": "/path/to/audio.wav"
}
```

#### 2. Generate TTS (File Upload)
```bash
POST /api/tts/upload
Content-Type: multipart/form-data

text: "Hello world"
audio_file: <file>
temperature: 0.8
top_p: 0.95
top_k: 1000
repetition_penalty: 1.2
min_p: 0.0
norm_loudness: true
seed: 42
```

#### 3. Health Check
```bash
GET /api/health
```

#### 4. Get Supported Tags
```bash
GET /api/tags
```

#### 5. API Documentation
Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## Example API Usage

### Python
```python
import requests

# JSON API
response = requests.post(
    "http://localhost:8000/api/tts",
    json={
        "text": "Hello [chuckle], this is a test",
        "temperature": 0.8,
        "top_p": 0.95,
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)

# File Upload API
with open("reference.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/tts/upload",
        files={"audio_file": f},
        data={
            "text": "Hello world",
            "temperature": 0.8,
        }
    )

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### cURL
```bash
# JSON API
curl -X POST "http://localhost:8000/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello [chuckle] world", "temperature": 0.8}' \
  --output output.wav

# File Upload API
curl -X POST "http://localhost:8000/api/tts/upload" \
  -F "text=Hello world" \
  -F "audio_file=@reference.wav" \
  -F "temperature=0.8" \
  --output output.wav
```

## Supported Paralinguistic Tags

- `[clear throat]`
- `[sigh]`
- `[shush]`
- `[cough]`
- `[groan]`
- `[sniff]`
- `[gasp]`
- `[chuckle]`
- `[laugh]`

## Parameters

- **text** (required): Text to synthesize
- **temperature** (0.05-2.0, default: 0.8): Sampling temperature
- **top_p** (0.0-1.0, default: 0.95): Top-p (nucleus) sampling
- **top_k** (0-1000, default: 1000): Top-k sampling
- **repetition_penalty** (1.0-2.0, default: 1.2): Repetition penalty
- **min_p** (0.0-1.0, default: 0.0): Min-p sampling (0 to disable)
- **norm_loudness** (bool, default: true): Normalize loudness to -27 LUFS
- **seed** (int, optional): Random seed for reproducibility
- **audio_prompt_path** (str, optional): Path to reference audio file

## Performance Tips

1. **Model Caching**: The model is loaded once and cached in memory for fast subsequent requests
2. **Async Support**: FastAPI provides async support for better concurrency
3. **Workers**: Use multiple workers for production (be aware of GPU memory limits)
4. **GPU**: Ensure CUDA is available for best performance

## Notes

- The Turbo model ignores `cfg_weight`, `exaggeration`, and `min_p` parameters (they're accepted but not used)
- Reference audio files are temporarily saved during processing and automatically cleaned up
- All audio output is in WAV format at the model's sample rate

