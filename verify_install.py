import sys
import os

print("Verifying IndexTTS-2 Installation...")

try:
    import torch
    print(f"Torch version: {torch.__version__}")
except ImportError:
    print("Error: torch not found.")
    sys.exit(1)

try:
    from indextts2.config import Config
    print("Config module loaded.")
except ImportError as e:
    print(f"Error loading Config: {e}")
    sys.exit(1)

try:
    from indextts2.utils.audio import load_audio
    print("Audio utils loaded.")
except ImportError as e:
    print(f"Error loading Audio utils: {e}")
    sys.exit(1)

try:
    from indextts2.inference import IndexTTS2
    print("IndexTTS2 class loaded.")
except ImportError as e:
    print(f"Error loading IndexTTS2: {e}")
    sys.exit(1)

print("Verification Successful! All modules are importable.")
