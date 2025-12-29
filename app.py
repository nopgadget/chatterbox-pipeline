"""
FastAPI Web Application for Chatterbox Turbo TTS
High-performance API and Web UI
"""
import os
import io
import random
import tempfile
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from chatterbox.tts_turbo import ChatterboxTurboTTS

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CACHE = {}
VOICE_FOLDER = Path("voice")
AVAILABLE_VOICES = {}

# Supported paralinguistic tags
EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]

def scan_voice_folder():
    """Scan voice folder for available voice files at startup.
    NOTE: This function only READS from the voice folder - it NEVER deletes or modifies files.
    """
    global AVAILABLE_VOICES
    AVAILABLE_VOICES = {}
    
    if not VOICE_FOLDER.exists():
        VOICE_FOLDER.mkdir(exist_ok=True)
        print(f"Created voice folder: {VOICE_FOLDER}")
        return
    
    # Scan for audio files (READ-ONLY - never deletes or modifies files)
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    for file_path in VOICE_FOLDER.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            # Store both with and without extension for easy lookup
            name_with_ext = file_path.name
            name_without_ext = file_path.stem
            full_path = str(file_path.resolve())
            
            AVAILABLE_VOICES[name_with_ext] = full_path
            AVAILABLE_VOICES[name_without_ext] = full_path
    
    if AVAILABLE_VOICES:
        print(f"Found {len(set(AVAILABLE_VOICES.values()))} voice file(s) in {VOICE_FOLDER}:")
        for name in sorted(set(AVAILABLE_VOICES.values())):
            print(f"  - {Path(name).name}")
    else:
        print(f"No voice files found in {VOICE_FOLDER}")

def resolve_voice_path(voice_name: Optional[str] = None, audio_prompt_path: Optional[str] = None) -> Optional[str]:
    """
    Resolve voice path from voice_name or audio_prompt_path.
    Priority: uploaded file > voice_name > audio_prompt_path
    """
    # If audio_prompt_path is provided and exists, use it (for custom uploads)
    if audio_prompt_path and os.path.exists(audio_prompt_path):
        return audio_prompt_path
    
    # If voice_name is provided, look it up in available voices
    if voice_name:
        # Try exact match first
        if voice_name in AVAILABLE_VOICES:
            return AVAILABLE_VOICES[voice_name]
        # Try with .wav extension
        if f"{voice_name}.wav" in AVAILABLE_VOICES:
            return AVAILABLE_VOICES[f"{voice_name}.wav"]
        # Try case-insensitive match
        voice_name_lower = voice_name.lower()
        for key, path in AVAILABLE_VOICES.items():
            if key.lower() == voice_name_lower or key.lower() == f"{voice_name_lower}.wav":
                return path
    
    # Fall back to audio_prompt_path if provided
    if audio_prompt_path:
        return audio_prompt_path
    
    return None

# Scan voice folder at startup
scan_voice_folder()

# Initialize FastAPI app
app = FastAPI(
    title="Chatterbox Turbo TTS API",
    description="High-performance Text-to-Speech API with Web UI",
    version="1.0.0"
)

# Request/Response models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize (supports paralinguistic tags)")
    temperature: float = Field(0.8, ge=0.05, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(1000, ge=0, le=1000, description="Top-k sampling")
    repetition_penalty: float = Field(1.2, ge=1.0, le=2.0, description="Repetition penalty")
    min_p: float = Field(0.0, ge=0.0, le=1.0, description="Min-p sampling (0 to disable)")
    norm_loudness: bool = Field(True, description="Normalize loudness to -27 LUFS")
    seed: Optional[int] = Field(None, description="Random seed (None for random)")
    voice_name: Optional[str] = Field(None, description="Name of voice file from voice/ folder (e.g., '20secondchris' or '20secondchris.wav'). Use /api/voices to list available voices.")
    audio_prompt_path: Optional[str] = Field(None, description="Path to reference audio file (if using file upload, use /api/tts/upload endpoint). Ignored if voice_name is provided.")

class TTSResponse(BaseModel):
    success: bool
    message: str
    sample_rate: Optional[int] = None
    duration: Optional[float] = None

def get_model():
    """Get or load the TTS model (cached)"""
    if "model" not in MODEL_CACHE:
        print(f"Loading Chatterbox-Turbo on {DEVICE}...")
        MODEL_CACHE["model"] = ChatterboxTurboTTS.from_pretrained(DEVICE)
    return MODEL_CACHE["model"]

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def process_audio(wav: torch.Tensor, sample_rate: int) -> np.ndarray:
    """Process audio tensor to numpy array for saving"""
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    
    # Ensure correct shape
    if wav.ndim > 1:
        if wav.shape[0] < wav.shape[1]:
            wav = wav.T
        if wav.shape[0] == 1:
            wav = wav.squeeze(0)
        elif wav.ndim == 2 and wav.shape[1] == 1:
            wav = wav.squeeze(1)
    
    # Ensure float32 and normalize
    if wav.dtype != 'float32':
        wav = wav.astype('float32')
    wav = np.clip(wav, -1.0, 1.0)
    
    return wav

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the web UI"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatterbox Turbo TTS</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            height: 100vh;
            padding: 15px;
            overflow: hidden;
        }
        .header {
            background: #ffffff;
            color: #000000;
            border: 2px solid #000000;
            padding: 15px;
            text-align: center;
            margin-bottom: 15px;
            border-radius: 8px;
        }
        .header h1 { font-size: 1.6em; }
        .grid-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 15px;
            max-width: 1400px;
            margin: 0 auto;
            height: calc(100vh - 90px);
        }
        .box {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            min-height: 0;
        }
        .box h2 {
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #333;
            border-bottom: 2px solid #000;
            padding-bottom: 6px;
            flex-shrink: 0;
        }
        .form-group {
            margin-bottom: 12px;
            flex-shrink: 0;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            color: #333;
            font-size: 0.9em;
        }
        textarea {
            width: 100%;
            flex: 1;
            min-height: 100px;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 14px;
            font-family: inherit;
            resize: none;
        }
        textarea:focus {
            outline: none;
            border-color: #000000;
        }
        .tags-container {
            display: flex;
            flex-wrap: wrap;
            gap: 3px;
            margin-top: 6px;
            max-height: 60px;
            overflow-y: auto;
            width: 100%;
            align-items: flex-start;
        }
        .tag-btn {
            padding: 2px 6px;
            background: #ffffff;
            border: 1px solid #000000;
            color: #000000;
            border-radius: 3px;
            cursor: pointer;
            font-size: 9px;
            transition: all 0.2s;
            white-space: nowrap;
            flex: 0 0 auto;
            width: auto;
            max-width: none;
            display: inline-block;
        }
        .tag-btn:hover {
            background: #000000;
            color: #ffffff;
        }
        input[type="file"] {
            width: 100%;
            padding: 8px;
            border: 2px dashed #e0e0e0;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85em;
        }
        select {
            width: 100%;
            padding: 8px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 0.9em;
            font-family: inherit;
            background: white;
            cursor: pointer;
        }
        select:focus {
            outline: none;
            border-color: #000000;
        }
        .voice-info {
            font-size: 0.75em;
            color: #666;
            margin-top: 4px;
        }
        .slider-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 8px;
        }
        .slider-item {
            display: flex;
            flex-direction: column;
        }
        .slider-item label {
            margin-bottom: 3px;
            font-size: 0.8em;
        }
        input[type="range"] {
            width: 100%;
            margin: 4px 0;
        }
        .slider-value {
            font-size: 0.8em;
            color: #666;
            margin-top: 2px;
        }
        input[type="number"] {
            width: 100%;
            padding: 6px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 0.9em;
        }
        input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
        }
        button {
            background: #ffffff;
            color: #000000;
            border: 2px solid #000000;
            padding: 12px 20px;
            font-size: 1em;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
            transition: all 0.2s;
        }
        button:hover {
            background: #f5f5f5;
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .audio-output {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .audio-output h3 {
            margin-bottom: 10px;
            font-size: 1em;
        }
        audio {
            width: 100%;
            margin-top: 8px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 15px;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #000000;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            background: #f5f5f5;
            color: #000000;
            border: 1px solid #000000;
            padding: 10px;
            border-radius: 6px;
            margin-top: 10px;
            display: none;
            font-size: 0.85em;
        }
        .error.active {
            display: block;
        }
        .scrollable {
            overflow-y: auto;
            flex: 1;
            min-height: 0;
        }
        .download-link {
            margin-top: 8px;
            color: #000;
            text-decoration: none;
            font-size: 0.9em;
            display: inline-block;
        }
        .download-link:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Chatterbox Turbo TTS</h1>
    </div>
    
    <div class="grid-container">
        <!-- Box 1: Text Input -->
        <div class="box">
            <h2>Text Input</h2>
            <form id="ttsForm">
                <div class="form-group" style="flex: 1; display: flex; flex-direction: column;">
                    <label for="text">Text to Synthesize</label>
                    <textarea id="text" name="text" placeholder="Enter your text here... You can use paralinguistic tags like [cough], [laugh], [chuckle], etc.">Hi there [clear throat]..., this is Chris... Do you have a sec? [sniff] ... I really need 400 row-bucks [cough] ... added to my row-blocks account.</textarea>
                </div>
                <div class="form-group" style="flex-shrink: 0;">
                    <label style="font-size: 0.85em;">Paralinguistic Tags</label>
                    <div class="tags-container" id="tagsContainer"></div>
                </div>
                <button type="submit" id="generateBtn">Generate</button>
            </form>
        </div>
        
        <!-- Box 2: Voice Selection -->
        <div class="box">
            <h2>Voice Selection</h2>
            <div class="scrollable">
                <div class="form-group">
                    <label for="voiceSelect">Server-Side Voice</label>
                    <select id="voiceSelect" name="voiceSelect">
                        <option value="">Loading voices...</option>
                    </select>
                    <div class="voice-info">Select a voice for faster generation</div>
                </div>
                
                <div class="form-group">
                    <label for="audioFile">Or Upload Custom Audio</label>
                    <input type="file" id="audioFile" name="audioFile" accept="audio/*">
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 8px; font-size: 0.9em;">Generating...</p>
                </div>
                
                <div class="error" id="error"></div>
            </div>
        </div>
        
        <!-- Box 3: Settings -->
        <div class="box">
            <h2>Settings</h2>
            <div class="scrollable">
                <div class="slider-group">
                    <div class="slider-item">
                        <label for="temperature">Temperature: <span class="slider-value" id="tempValue">0.8</span></label>
                        <input type="range" id="temperature" name="temperature" min="0.05" max="2.0" step="0.05" value="0.8">
                    </div>
                    <div class="slider-item">
                        <label for="top_p">Top P: <span class="slider-value" id="topPValue">0.95</span></label>
                        <input type="range" id="top_p" name="top_p" min="0.0" max="1.0" step="0.01" value="0.95">
                    </div>
                    <div class="slider-item">
                        <label for="top_k">Top K: <span class="slider-value" id="topKValue">1000</span></label>
                        <input type="range" id="top_k" name="top_k" min="0" max="1000" step="10" value="1000">
                    </div>
                    <div class="slider-item">
                        <label for="repetition_penalty">Repetition Penalty: <span class="slider-value" id="repPenValue">1.2</span></label>
                        <input type="range" id="repetition_penalty" name="repetition_penalty" min="1.0" max="2.0" step="0.05" value="1.2">
                    </div>
                    <div class="slider-item">
                        <label for="min_p">Min P: <span class="slider-value" id="minPValue">0.0</span></label>
                        <input type="range" id="min_p" name="min_p" min="0.0" max="1.0" step="0.01" value="0.0">
                    </div>
                    <div class="slider-item">
                        <label for="seed">Seed: <span class="slider-value" id="seedValue">0</span></label>
                        <input type="number" id="seed" name="seed" value="0" min="0">
                    </div>
                </div>
                <div class="checkbox-group" style="margin-top: 8px;">
                    <input type="checkbox" id="norm_loudness" name="norm_loudness" checked>
                    <label for="norm_loudness" style="margin: 0; font-size: 0.85em;">Normalize Loudness (-27 LUFS)</label>
                </div>
            </div>
        </div>
        
        <!-- Box 4: Output -->
        <div class="box">
            <h2>Output</h2>
            <div class="audio-output" id="audioOutput" style="display: none;">
                <audio id="audioPlayer" controls></audio>
                <a id="downloadLink" href="#" download="output.wav" class="download-link">Download Audio</a>
            </div>
            <div id="noOutput" style="text-align: center; color: #999; padding: 40px 0; font-size: 0.9em;">
                Generated audio will appear here
            </div>
        </div>
    </div>
    
    <script>
        // Load available voices
        async function loadVoices() {
            try {
                const response = await fetch('/api/voices');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                const voiceSelect = document.getElementById('voiceSelect');
                
                if (data.voices && data.voices.length > 0) {
                    voiceSelect.innerHTML = '<option value="">None (use uploaded file)</option>';
                    data.voices.forEach((voice, index) => {
                        const option = document.createElement('option');
                        option.value = voice;
                        option.textContent = voice;
                        // Select the first voice by default
                        if (index === 0) {
                            option.selected = true;
                        }
                        voiceSelect.appendChild(option);
                    });
                } else {
                    voiceSelect.innerHTML = '<option value="">No voices available</option>';
                }
            } catch (err) {
                console.error('Failed to load voices:', err);
                const voiceSelect = document.getElementById('voiceSelect');
                voiceSelect.innerHTML = '<option value="">Error loading voices - check console</option>';
            }
        }
        
        // Load voices on page load
        loadVoices();
        
        // Initialize paralinguistic tags
        const tags = ["[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]", "[sniff]", "[gasp]", "[chuckle]", "[laugh]"];
        const tagsContainer = document.getElementById('tagsContainer');
        tags.forEach(tag => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'tag-btn';
            btn.textContent = tag;
            btn.onclick = () => insertTag(tag);
            tagsContainer.appendChild(btn);
        });
        
        function insertTag(tag) {
            const textarea = document.getElementById('text');
            const start = textarea.selectionStart;
            const end = textarea.selectionEnd;
            const text = textarea.value;
            const prefix = (start === 0 || text[start - 1] === ' ') ? '' : ' ';
            const suffix = (end < text.length && text[end] === ' ') ? '' : ' ';
            textarea.value = text.slice(0, start) + prefix + tag + suffix + text.slice(end);
            textarea.focus();
            textarea.setSelectionRange(start + prefix.length + tag.length, start + prefix.length + tag.length);
        }
        
        // Update slider values
        document.getElementById('temperature').addEventListener('input', (e) => {
            document.getElementById('tempValue').textContent = e.target.value;
        });
        document.getElementById('top_p').addEventListener('input', (e) => {
            document.getElementById('topPValue').textContent = e.target.value;
        });
        document.getElementById('top_k').addEventListener('input', (e) => {
            document.getElementById('topKValue').textContent = e.target.value;
        });
        document.getElementById('repetition_penalty').addEventListener('input', (e) => {
            document.getElementById('repPenValue').textContent = e.target.value;
        });
        document.getElementById('min_p').addEventListener('input', (e) => {
            document.getElementById('minPValue').textContent = e.target.value;
        });
        document.getElementById('seed').addEventListener('input', (e) => {
            document.getElementById('seedValue').textContent = e.target.value;
        });
        
        
        // Form submission
        document.getElementById('ttsForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const audioFile = document.getElementById('audioFile').files[0];
            
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const audioOutput = document.getElementById('audioOutput');
            const generateBtn = document.getElementById('generateBtn');
            
            loading.classList.add('active');
            error.classList.remove('active');
            audioOutput.style.display = 'none';
            generateBtn.disabled = true;
            
            try {
                let response;
                let voiceName = document.getElementById('voiceSelect').value;
                
                // If no voice selected and no file uploaded, use first available voice
                if (!voiceName && !audioFile) {
                    const voiceSelect = document.getElementById('voiceSelect');
                    const firstVoiceOption = voiceSelect.querySelector('option[value]:not([value=""])');
                    if (firstVoiceOption) {
                        voiceName = firstVoiceOption.value;
                    }
                }
                
                if (audioFile) {
                    // Use upload endpoint (uploaded file takes priority)
                    const uploadData = new FormData();
                    uploadData.append('text', formData.get('text'));
                    uploadData.append('audio_file', audioFile);
                    uploadData.append('temperature', parseFloat(formData.get('temperature') || '0.8').toString());
                    uploadData.append('top_p', parseFloat(formData.get('top_p') || '0.95').toString());
                    uploadData.append('top_k', parseInt(formData.get('top_k') || '1000').toString());
                    uploadData.append('repetition_penalty', parseFloat(formData.get('repetition_penalty') || '1.2').toString());
                    uploadData.append('min_p', parseFloat(formData.get('min_p') || '0.0').toString());
                    const normLoudness = formData.get('norm_loudness');
                    uploadData.append('norm_loudness', normLoudness === 'on' || normLoudness === 'true' ? 'true' : 'false');
                    const seed = formData.get('seed');
                    if (seed && seed !== '0') uploadData.append('seed', parseInt(seed).toString());
                    
                    response = await fetch('/api/tts/upload', {
                        method: 'POST',
                        body: uploadData
                    });
                } else if (voiceName) {
                    // Use upload endpoint with voice_name (no file upload)
                    const uploadData = new FormData();
                    uploadData.append('text', formData.get('text'));
                    uploadData.append('voice_name', voiceName);
                    uploadData.append('temperature', parseFloat(formData.get('temperature') || '0.8').toString());
                    uploadData.append('top_p', parseFloat(formData.get('top_p') || '0.95').toString());
                    uploadData.append('top_k', parseInt(formData.get('top_k') || '1000').toString());
                    uploadData.append('repetition_penalty', parseFloat(formData.get('repetition_penalty') || '1.2').toString());
                    uploadData.append('min_p', parseFloat(formData.get('min_p') || '0.0').toString());
                    const normLoudness = formData.get('norm_loudness');
                    uploadData.append('norm_loudness', normLoudness === 'on' || normLoudness === 'true' ? 'true' : 'false');
                    const seed = formData.get('seed');
                    if (seed && seed !== '0') uploadData.append('seed', parseInt(seed).toString());
                    
                    response = await fetch('/api/tts/upload', {
                        method: 'POST',
                        body: uploadData
                    });
                } else {
                    // Use JSON endpoint (no voice specified)
                    const jsonData = {
                        text: formData.get('text'),
                        temperature: parseFloat(formData.get('temperature')),
                        top_p: parseFloat(formData.get('top_p')),
                        top_k: parseInt(formData.get('top_k')),
                        repetition_penalty: parseFloat(formData.get('repetition_penalty')),
                        min_p: parseFloat(formData.get('min_p')),
                        norm_loudness: formData.get('norm_loudness') === 'on',
                    };
                    const seed = formData.get('seed');
                    if (seed && seed !== '0') jsonData.seed = parseInt(seed);
                    if (voiceName) jsonData.voice_name = voiceName;
                    
                    response = await fetch('/api/tts', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(jsonData)
                    });
                }
                
                if (!response.ok) {
                    let errorMessage = 'Generation failed';
                    try {
                        const errorData = await response.json();
                        // FastAPI validation errors (422) return detail as array
                        if (Array.isArray(errorData.detail)) {
                            errorMessage = errorData.detail.map(e => e.msg || e.loc?.join('.') || JSON.stringify(e)).join(', ');
                        } else if (errorData.detail) {
                            errorMessage = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
                        } else if (errorData.message) {
                            errorMessage = errorData.message;
                        }
                    } catch (parseErr) {
                        errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                    }
                    throw new Error(errorMessage);
                }
                
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                document.getElementById('audioPlayer').src = url;
                document.getElementById('downloadLink').href = url;
                document.getElementById('downloadLink').download = 'output.wav';
                document.getElementById('noOutput').style.display = 'none';
                audioOutput.style.display = 'block';
            } catch (err) {
                error.textContent = 'Error: ' + (err.message || String(err));
                error.classList.add('active');
            } finally {
                loading.classList.remove('active');
                generateBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
    """
    return html_content

@app.post("/api/tts")
async def generate_tts(request: TTSRequest):
    """Generate TTS audio from text (JSON API)"""
    try:
        model = get_model()
        
        if request.seed is not None and request.seed != 0:
            set_seed(request.seed)
        
        # Resolve voice path (voice_name takes priority over audio_prompt_path)
        audio_prompt_path = resolve_voice_path(request.voice_name, request.audio_prompt_path)
        
        wav = model.generate(
            text=request.text,
            audio_prompt_path=audio_prompt_path,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            min_p=request.min_p,
            norm_loudness=request.norm_loudness,
        )
        
        # Process audio
        wav_np = process_audio(wav, model.sr)
        
        # Create in-memory WAV file
        buffer = io.BytesIO()
        sf.write(buffer, wav_np, model.sr, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="tts_output.wav"',
                "X-Sample-Rate": str(model.sr),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts/upload")
async def generate_tts_upload(
    text: str = Form(...),
    audio_file: UploadFile = File(None),
    voice_name: Optional[str] = Form(None),
    temperature: float = Form(0.8),
    top_p: float = Form(0.95),
    top_k: int = Form(1000),
    repetition_penalty: float = Form(1.2),
    min_p: float = Form(0.0),
    norm_loudness: bool = Form(True),
    seed: Optional[str] = Form(None),
):
    """Generate TTS audio with file upload support. Can use voice_name from server or upload custom audio."""
    try:
        model = get_model()
        
        # Handle seed parameter
        seed_int = None
        if seed:
            try:
                seed_int = int(seed)
                if seed_int != 0:
                    set_seed(seed_int)
            except (ValueError, TypeError):
                pass
        
        # Priority: uploaded file > voice_name > None
        audio_prompt_path = None
        if audio_file:
            # Save uploaded file temporarily if provided
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as tmp_file:
                content = await audio_file.read()
                tmp_file.write(content)
                audio_prompt_path = tmp_file.name
        elif voice_name:
            # Use server-side voice if no file uploaded
            audio_prompt_path = resolve_voice_path(voice_name, None)
            if not audio_prompt_path:
                raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found. Use /api/voices to list available voices.")
        
        try:
            wav = model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                norm_loudness=norm_loudness,
            )
            
            # Process audio
            wav_np = process_audio(wav, model.sr)
            
            # Create in-memory WAV file
            buffer = io.BytesIO()
            sf.write(buffer, wav_np, model.sr, format='WAV')
            buffer.seek(0)
            
            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f'attachment; filename="tts_output.wav"',
                    "X-Sample-Rate": str(model.sr),
                }
            )
        finally:
            # Clean up temporary file
            if audio_prompt_path and os.path.exists(audio_prompt_path):
                os.unlink(audio_prompt_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": "model" in MODEL_CACHE,
        "cuda_available": torch.cuda.is_available(),
    }

@app.get("/api/tags")
async def get_tags():
    """Get list of supported paralinguistic tags"""
    return {"tags": EVENT_TAGS}

@app.get("/api/voices")
async def get_voices():
    """Get list of available voice files from voice/ folder"""
    # Return unique voice names (without duplicates from extension variations)
    unique_voices = {}
    for key, path in AVAILABLE_VOICES.items():
        if key.endswith('.wav'):
            unique_voices[key] = path
        elif f"{key}.wav" not in AVAILABLE_VOICES:
            unique_voices[key] = path
    
    return {
        "voices": sorted(unique_voices.keys()),
        "count": len(set(AVAILABLE_VOICES.values()))
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )

