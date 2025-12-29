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

# Supported paralinguistic tags
EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]

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
    audio_prompt_path: Optional[str] = Field(None, description="Path to reference audio file (if using file upload, use /api/tts/upload endpoint)")

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
            background: #ffffff;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
            overflow: hidden;
        }
        .header {
            background: #000000;
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { opacity: 0.9; }
        .content {
            padding: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        textarea {
            width: 100%;
            min-height: 120px;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
        }
        textarea:focus {
            outline: none;
            border-color: #000000;
        }
        .tags-container {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 10px;
        }
        .tag-btn {
            padding: 4px 10px;
            background: #ffffff;
            border: 1px solid #000000;
            color: #000000;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s;
            white-space: nowrap;
            width: auto;
            display: inline-block;
        }
        .tag-btn:hover {
            background: #000000;
            color: #ffffff;
        }
        input[type="file"] {
            width: 100%;
            padding: 10px;
            border: 2px dashed #e0e0e0;
            border-radius: 8px;
            cursor: pointer;
        }
        .slider-group {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .slider-item {
            display: flex;
            flex-direction: column;
        }
        .slider-item label {
            margin-bottom: 5px;
        }
        input[type="range"] {
            width: 100%;
            margin: 10px 0;
        }
        .slider-value {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
        }
        input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        button {
            background: #000000;
            color: white;
            border: 1px solid #000000;
            padding: 15px 30px;
            font-size: 18px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
            transition: all 0.2s;
        }
        button:hover {
            background: #333333;
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .audio-output {
            margin-top: 30px;
            padding: 20px;
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        .audio-output h3 {
            margin-bottom: 15px;
        }
        audio {
            width: 100%;
            margin-top: 10px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #000000;
            border-radius: 50%;
            width: 40px;
            height: 40px;
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
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            display: none;
        }
        .error.active {
            display: block;
        }
        .api-info {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
        }
        .api-info h3 {
            margin-bottom: 10px;
        }
        .api-info code {
            background: #f5f5f5;
            border: 1px solid #e0e0e0;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
        }
        .accordion {
            margin-top: 20px;
        }
        .accordion-header {
            background: #f5f5f5;
            padding: 15px;
            cursor: pointer;
            border-radius: 8px;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid #e0e0e0;
        }
        .accordion-header:hover {
            background: #e8e8e8;
        }
        .accordion-content {
            display: none;
            padding: 20px;
            background: #ffffff;
            border-radius: 0 0 8px 8px;
            border: 1px solid #e0e0e0;
            border-top: none;
        }
        .accordion-content.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Chatterbox Turbo TTS</h1>
        </div>
        <div class="content">
            <form id="ttsForm">
                <div class="form-group">
                    <label for="text">Text to Synthesize</label>
                    <textarea id="text" name="text" placeholder="Enter your text here... You can use paralinguistic tags like [cough], [laugh], [chuckle], etc.">Hi there [clear throat]..., this is Chris... Do you have a sec? [sniff] ... I really need 400 row-bucks [cough] ... added to my row-blocks account.</textarea>
                </div>
                
                <div class="accordion">
                    <div class="accordion-header" onclick="toggleTagsAccordion()">
                        <span>Paralinguistic Tags</span>
                        <span id="tagsAccordionIcon">▼</span>
                    </div>
                    <div class="accordion-content" id="tagsAccordionContent">
                        <div class="tags-container" id="tagsContainer"></div>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="audioFile">Reference Audio File (Optional)</label>
                    <input type="file" id="audioFile" name="audioFile" accept="audio/*">
                </div>
                
                <div class="accordion">
                    <div class="accordion-header" onclick="toggleAccordion()">
                        <span>Advanced Options</span>
                        <span id="accordionIcon">▼</span>
                    </div>
                    <div class="accordion-content" id="accordionContent">
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
                                <label for="seed">Seed (0 for random): <span class="slider-value" id="seedValue">0</span></label>
                                <input type="number" id="seed" name="seed" value="0" min="0">
                            </div>
                        </div>
                        <div class="checkbox-group" style="margin-top: 15px;">
                            <input type="checkbox" id="norm_loudness" name="norm_loudness" checked>
                            <label for="norm_loudness" style="margin: 0;">Normalize Loudness (-27 LUFS)</label>
                        </div>
                    </div>
                </div>
                
                <button type="submit" id="generateBtn">Generate ⚡</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 10px;">Generating audio...</p>
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="audio-output" id="audioOutput" style="display: none;">
                <h3>Generated Audio</h3>
                <audio id="audioPlayer" controls></audio>
                <p style="margin-top: 10px; color: #666;">
                    <a id="downloadLink" href="#" download="output.wav">Download Audio</a>
                </p>
            </div>
            
            <div class="api-info">
                <h3>API Documentation</h3>
                <p>Access the API at <code>/docs</code> for interactive API documentation.</p>
                <p>POST to <code>/api/tts</code> with JSON body or use <code>/api/tts/upload</code> for file uploads.</p>
            </div>
        </div>
    </div>
    
    <script>
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
        
        function toggleTagsAccordion() {
            const content = document.getElementById('tagsAccordionContent');
            const icon = document.getElementById('tagsAccordionIcon');
            content.classList.toggle('active');
            icon.textContent = content.classList.contains('active') ? '▲' : '▼';
        }
        
        function toggleAccordion() {
            const content = document.getElementById('accordionContent');
            const icon = document.getElementById('accordionIcon');
            content.classList.toggle('active');
            icon.textContent = content.classList.contains('active') ? '▲' : '▼';
        }
        
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
                if (audioFile) {
                    // Use upload endpoint
                    const uploadData = new FormData();
                    uploadData.append('text', formData.get('text'));
                    uploadData.append('audio_file', audioFile);
                    uploadData.append('temperature', formData.get('temperature'));
                    uploadData.append('top_p', formData.get('top_p'));
                    uploadData.append('top_k', formData.get('top_k'));
                    uploadData.append('repetition_penalty', formData.get('repetition_penalty'));
                    uploadData.append('min_p', formData.get('min_p'));
                    uploadData.append('norm_loudness', formData.get('norm_loudness') ? 'true' : 'false');
                    const seed = formData.get('seed');
                    if (seed && seed !== '0') uploadData.append('seed', seed);
                    
                    response = await fetch('/api/tts/upload', {
                        method: 'POST',
                        body: uploadData
                    });
                } else {
                    // Use JSON endpoint
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
                    
                    response = await fetch('/api/tts', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(jsonData)
                    });
                }
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Generation failed');
                }
                
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                document.getElementById('audioPlayer').src = url;
                document.getElementById('downloadLink').href = url;
                document.getElementById('downloadLink').download = 'output.wav';
                audioOutput.style.display = 'block';
            } catch (err) {
                error.textContent = 'Error: ' + err.message;
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
        
        wav = model.generate(
            text=request.text,
            audio_prompt_path=request.audio_prompt_path,
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
    temperature: float = Form(0.8),
    top_p: float = Form(0.95),
    top_k: int = Form(1000),
    repetition_penalty: float = Form(1.2),
    min_p: float = Form(0.0),
    norm_loudness: bool = Form(True),
    seed: Optional[str] = Form(None),
):
    """Generate TTS audio with file upload support"""
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
        
        # Save uploaded file temporarily if provided
        audio_prompt_path = None
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as tmp_file:
                content = await audio_file.read()
                tmp_file.write(content)
                audio_prompt_path = tmp_file.name
        
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

