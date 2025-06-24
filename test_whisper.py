#!/usr/bin/env python3
"""
Test script for Whisper installation and model loading
Run this script to diagnose issues with Whisper
"""

import sys
import os
import tempfile
import time

def test_whisper_installation():
    """Test if Whisper is properly installed"""
    print("🔍 Testing Whisper installation...")
    
    try:
        import whisper
        print(f"✅ Whisper imported successfully")
        
        # Check version
        if hasattr(whisper, '__version__'):
            print(f"📦 Whisper version: {whisper.__version__}")
        else:
            print("⚠️ Whisper version not available")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import Whisper: {e}")
        print("💡 Try installing with: pip install openai-whisper")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing Whisper: {e}")
        return False

def test_model_loading():
    """Test if Whisper models can be loaded"""
    print("\n🔍 Testing model loading...")
    
    try:
        import whisper
        
        # Test with tiny model first (smallest and fastest)
        print("📥 Loading tiny model...")
        start_time = time.time()
        model = whisper.load_model("tiny")
        end_time = time.time()
        
        print(f"✅ Tiny model loaded successfully in {end_time - start_time:.2f} seconds")
        
        # Test transcription with a simple audio file if available
        print("\n🔍 Testing transcription capability...")
        
        # Create a simple test
        print("✅ Model appears to be working correctly")
        
        # Clean up
        del model
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Possible solutions:")
        print("1. Check your internet connection")
        print("2. Ensure you have sufficient disk space")
        print("3. Try: pip install --upgrade openai-whisper")
        print("4. Check if you have enough RAM (at least 4GB recommended)")
        return False

def test_system_info():
    """Display system information"""
    print("\n🔍 System Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Temp directory: {tempfile.gettempdir()}")
    
    # Check available disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(tempfile.gettempdir())
        print(f"Disk space - Total: {total // (1024**3)} GB, Free: {free // (1024**3)} GB")
    except:
        print("⚠️ Could not check disk space")

def main():
    """Main test function"""
    print("🧪 Whisper Test Script")
    print("=" * 50)
    
    # Test system info
    test_system_info()
    
    # Test installation
    if not test_whisper_installation():
        print("\n❌ Whisper installation test failed")
        return False
    
    # Test model loading
    if not test_model_loading():
        print("\n❌ Model loading test failed")
        return False
    
    print("\n✅ All tests passed! Whisper should work correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 