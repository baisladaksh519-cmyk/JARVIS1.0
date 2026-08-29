# Optional voice module for FRIDAY (disabled by default)

# This file provides hooks to integrate local TTS/STT engines (e.g., Coqui TTS, eSpeak, whisper.cpp)
# We keep these functions minimal and optional so FRIDAY can run without them.

import os
from typing import Optional

def tts_synthesize(text: str, voice: Optional[str] = None, out_path: Optional[str] = None) -> Optional[str]:
    """Synthesize text to speech and return path to audio file if available.

    By default this is a no-op placeholder. To enable, install a local TTS engine and
    implement the call here.
    """
    # Placeholder: user can integrate Coqui TTS or pyttsx3 here.
    return None

def stt_transcribe(audio_path: str) -> Optional[str]:
    """Transcribe a local audio file to text. Placeholder for whisper.cpp / VOSK integration."""
    return None

