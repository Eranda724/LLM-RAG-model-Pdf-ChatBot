#!/usr/bin/env python3
"""
Test script to verify Whisper model loading without meta tensor errors
"""

import torch
from transformers import pipeline
import whisper

def test_transformers_pipeline():
    """Test transformers pipeline approach"""
    print("Testing transformers pipeline...")
    try:
        # Try the simple approach first
        asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")
        print("✅ Transformers pipeline loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Transformers pipeline failed: {e}")
        return False

def test_whisper_library():
    """Test whisper library approach"""
    print("Testing whisper library...")
    try:
        whisper_model = whisper.load_model("small")
        print("✅ Whisper library loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Whisper library failed: {e}")
        return False

def test_device_info():
    """Test device information"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers not available")

if __name__ == "__main__":
    print("=== Whisper Model Loading Test ===")
    test_device_info()
    print()
    
    success1 = test_transformers_pipeline()
    print()
    success2 = test_whisper_library()
    print()
    
    if success1 or success2:
        print("✅ At least one method works!")
    else:
        print("❌ Both methods failed!") 