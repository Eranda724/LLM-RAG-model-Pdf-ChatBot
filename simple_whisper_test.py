#!/usr/bin/env python3
"""
Simple test for whisper library
"""

print("Starting whisper test...")

try:
    print("Attempting to import whisper...")
    import whisper
    print("✅ Whisper imported successfully")
    
    # Try to load a small model
    print("Loading whisper model...")
    model = whisper.load_model("tiny")
    print("✅ Whisper model loaded successfully")
    
    print("Whisper is working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed.") 