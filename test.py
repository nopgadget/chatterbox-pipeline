import torch
import numpy as np
import soundfile as sf
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load the Turbo model
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Generate with Paralinguistic Tags
text = "Hi there [clear throat]..., this is Chris... Do you have a sec? [sniff] ... I really need 400 row-bucks [cough] ... added to my row-blocks account."

# Generate audio (requires a reference clip for voice cloning)
wav = model.generate(text, audio_prompt_path="../20secondchris.wav", cfg_weight=0.3)

# Convert to numpy if it's a tensor
if isinstance(wav, torch.Tensor):
    wav = wav.cpu().numpy()

# Ensure correct shape: soundfile expects (samples,) for mono or (samples, channels) for stereo
if wav.ndim > 1:
    # If shape is (channels, samples), transpose to (samples, channels)
    if wav.shape[0] < wav.shape[1]:
        wav = wav.T
    # If it's mono with shape (1, samples), squeeze to (samples,)
    if wav.shape[0] == 1:
        wav = wav.squeeze(0)
    elif wav.shape[1] == 1:
        wav = wav.squeeze(1)

# Ensure float32 format
if wav.dtype != 'float32':
    wav = wav.astype('float32')

# Normalize if needed (clip to [-1, 1])
wav = np.clip(wav, -1.0, 1.0)

# Save audio file
sf.write("test-turbo.wav", wav, model.sr)