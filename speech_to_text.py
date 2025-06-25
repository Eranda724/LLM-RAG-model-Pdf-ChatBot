import whisper
import streamlit as st
import tempfile
import os
import time
from typing import Tuple, Optional, Dict, Any, List
import logging
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue
import threading
import numpy as np
import av
import io
import wave

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToText:
    def __init__(self):
        """Initialize the Speech to Text model"""
        self.model = None
        self.model_loaded = False
        self.model_size = None
        self.audio_queue = queue.Queue()
        self.audio_data: List[float] = []
        
    def load_model(self, model_size: str = "base") -> bool:
        """Load the Whisper model"""
        try:
            with st.spinner(f"🔄 Loading Whisper {model_size} model..."):
                # Clear any existing model to free memory
                if self.model is not None:
                    del self.model
                    self.model = None
                
                # Load the new model
                self.model = whisper.load_model(model_size)
                self.model_loaded = True
                self.model_size = model_size
                logger.info(f"Successfully loaded Whisper {model_size} model")
                return True
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            st.error(f"❌ Error loading Whisper model: {str(e)}")
            self.model_loaded = False
            return False

    def clear_audio_data(self):
        """Clear recorded audio data"""
        self.audio_data = []

    def create_temp_audio_file(self, audio_data: bytes, file_extension: str = '.wav') -> str:
        """Create a temporary audio file with proper error handling"""
        try:
            # Use tempfile.NamedTemporaryFile for better Windows compatibility
            temp_file = tempfile.NamedTemporaryFile(
                suffix=file_extension,
                prefix="whisper_audio_",
                delete=False  # Don't auto-delete, we'll handle cleanup
            )
            
            temp_path = temp_file.name
            
            # Write audio data to file
            temp_file.write(audio_data)
            temp_file.flush()  # Ensure data is written to disk
            temp_file.close()  # Close file handle before Whisper uses it
            
            # Verify file was created and has content
            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"Temporary file was not created: {temp_path}")
            
            if os.path.getsize(temp_path) == 0:
                raise ValueError(f"Temporary file is empty: {temp_path}")
            
            logger.info(f"Created temporary audio file: {temp_path} ({len(audio_data)} bytes)")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating temporary audio file: {str(e)}")
            raise

    def create_wav_from_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert numpy array to WAV bytes"""
        try:
            # Ensure audio is in the right format
            if audio_array.dtype != np.int16:
                # Convert float to int16
                if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                    audio_array = (audio_array * 32767).astype(np.int16)
                else:
                    audio_array = audio_array.astype(np.int16)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_array.tobytes())
            
            wav_buffer.seek(0)
            return wav_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error creating WAV from array: {str(e)}")
            raise

    def transcribe_audio(self, audio_file=None, mic_audio=None) -> Tuple[bool, str, float]:
        """Transcribe audio from either file upload or microphone"""
        if not self.model_loaded or self.model is None:
            return False, "Model not loaded. Please load the model first.", 0.0

        temp_path = None
        try:
            # Determine which audio source to use and prepare audio data
            if mic_audio is not None:
                # Handle microphone audio (numpy array or raw bytes)
                if isinstance(mic_audio, np.ndarray):
                    audio_data = self.create_wav_from_array(mic_audio)
                else:
                    # Assume it's already audio bytes, convert to proper WAV format
                    try:
                        # Try to interpret as int16 array
                        audio_array = np.frombuffer(mic_audio, dtype=np.int16)
                        audio_data = self.create_wav_from_array(audio_array)
                    except:
                        # If that fails, use the raw bytes
                        audio_data = mic_audio
                file_extension = '.wav'
                
            elif audio_file is not None:
                # Handle uploaded file
                audio_file.seek(0)  # Reset file pointer
                audio_data = audio_file.read()
                file_extension = os.path.splitext(audio_file.name)[1].lower() or '.wav'
                
            else:
                return False, "No audio provided", 0.0

            # Validate audio data
            if not audio_data or len(audio_data) == 0:
                return False, "Audio data is empty", 0.0

            # Create temporary file
            temp_path = self.create_temp_audio_file(audio_data, file_extension)

            # Add a small delay to ensure file is ready
            time.sleep(0.1)
            
            # Double-check file exists and is accessible
            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"Temporary file disappeared: {temp_path}")
            
            # Transcribe with error handling
            start_time = time.time()
            logger.info(f"Starting transcription of file: {temp_path}")
            
            try:
                result: Dict[str, Any] = self.model.transcribe(temp_path)
            except Exception as whisper_error:
                # If Whisper fails, try to give more specific error info
                logger.error(f"Whisper transcription failed: {str(whisper_error)}")
                
                # Check if file is still accessible
                try:
                    with open(temp_path, 'rb') as test_file:
                        test_data = test_file.read(100)  # Read first 100 bytes
                    logger.info(f"File is accessible, first 100 bytes length: {len(test_data)}")
                except Exception as file_error:
                    logger.error(f"File access error: {str(file_error)}")
                    raise FileNotFoundError(f"Cannot access temporary file: {temp_path}")
                
                # Re-raise the original Whisper error
                raise whisper_error
            
            end_time = time.time()
            
            processing_time = round(end_time - start_time, 2)
            transcription = str(result.get("text", "")).strip()
            
            logger.info(f"Transcription completed in {processing_time} seconds")

            if not transcription:
                return False, "No text was transcribed. Please check your audio quality and try again.", processing_time

            return True, transcription, processing_time

        except Exception as e:
            error_msg = f"Error transcribing audio: {str(e)}"
            logger.error(error_msg)
            
            # Additional debugging info
            if temp_path:
                logger.error(f"Temp file path: {temp_path}")
                logger.error(f"Temp file exists: {os.path.exists(temp_path) if temp_path else 'N/A'}")
                if os.path.exists(temp_path):
                    logger.error(f"Temp file size: {os.path.getsize(temp_path)}")
            
            return False, error_msg, 0.0
            
        finally:
            # Clean up temporary file with retry logic
            if temp_path and os.path.exists(temp_path):
                for attempt in range(3):  # Try up to 3 times
                    try:
                        time.sleep(0.1)  # Small delay before cleanup
                        os.unlink(temp_path)
                        logger.info(f"Cleaned up temporary file: {temp_path}")
                        break
                    except Exception as e:
                        if attempt < 2:  # If not the last attempt
                            logger.warning(f"Cleanup attempt {attempt + 1} failed: {str(e)}")
                            time.sleep(0.5)  # Wait before retry
                        else:
                            logger.warning(f"Failed to clean up temporary file {temp_path}: {str(e)}")

def create_speech_to_text_interface():
    """Create the Speech to Text interface"""
    st.markdown("## 🎤 Speech to Text")
    st.markdown("Convert your speech to text using AI-powered transcription.")
    
    # Initialize session state
    if 'stt_model' not in st.session_state:
        st.session_state.stt_model = SpeechToText()
    
    if 'stt_transcriptions' not in st.session_state:
        st.session_state.stt_transcriptions = []
    
    # Model selection and loading
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_size = st.selectbox(
            "Select Whisper Model Size:",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger models are more accurate but slower. Start with 'base' for good balance."
        )
    
    with col2:
        if st.button("🔄 Load Model", type="primary"):
            success = st.session_state.stt_model.load_model(model_size)
            if success:
                st.success("✅ Model loaded successfully!")
    
    # Show model status
    if st.session_state.stt_model.model_loaded:
        st.success(f"✅ Whisper {st.session_state.stt_model.model_size} model is ready!")
    else:
        st.warning("⚠️ Please load the Whisper model first")
        st.stop()
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record Audio"])
    
    with tab1:
        st.markdown("### 📁 Upload Audio File")
        st.info("Supported formats: WAV, MP3, M4A, FLAC, OGG")
        
        uploaded_audio = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
            key="stt_file_upload"
        )
        
        if uploaded_audio:
            file_size = len(uploaded_audio.getvalue())
            st.info(f"📁 File: {uploaded_audio.name} ({file_size:,} bytes)")
            
            if file_size > 25 * 1024 * 1024:  # 25MB limit
                st.warning("⚠️ File is quite large. Processing may take longer.")
            
            if st.button("🎵 Transcribe Audio File", type="primary"):
                with st.spinner("🎵 Transcribing audio... This may take a few moments."):
                    success, transcription, processing_time = st.session_state.stt_model.transcribe_audio(audio_file=uploaded_audio)
                    
                    if success:
                        st.success(f"✅ Transcription completed in {processing_time} seconds!")
                        
                        # Add to history
                        st.session_state.stt_transcriptions.append({
                            'type': 'file',
                            'filename': uploaded_audio.name,
                            'transcription': transcription,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'processing_time': processing_time
                        })
                        
                        st.markdown("### 📝 Transcription Result:")
                        st.text_area(
                            "Transcribed Text:",
                            transcription,
                            height=200,
                            key=f"transcription_result_{len(st.session_state.stt_transcriptions)}"
                        )
                        
                        # Add copy button
                        if st.button("📋 Copy to Clipboard"):
                            st.code(transcription)
                    else:
                        st.error(f"❌ {transcription}")
    
    with tab2:
        st.markdown("### 🎤 Record Audio")
        st.info("Click 'START' to begin recording. Speak clearly into your microphone. Click 'STOP' when finished.")
        
        def audio_frames_callback(frame: av.AudioFrame) -> av.AudioFrame:
            """Process audio frames from WebRTC stream"""
            try:
                # Convert audio frame to numpy array
                audio_data = frame.to_ndarray().flatten().astype(np.float32)
                st.session_state.stt_model.audio_data.extend(audio_data.tolist())
            except Exception as e:
                logger.error(f"Error processing audio frame: {str(e)}")
            return frame
        
        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            audio_frame_callback=audio_frames_callback,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": False, "audio": True},
        )
        
        # Show recording status
        if webrtc_ctx.state.playing:
            st.success("🔴 Recording... Speak now!")
            st.info(f"📊 Audio samples collected: {len(st.session_state.stt_model.audio_data):,}")
        else:
            st.info("⏹️ Recording stopped. Click 'START' to begin recording.")
        
        # Transcribe button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎵 Transcribe Recording", type="primary"):
                if len(st.session_state.stt_model.audio_data) > 1000:  # Minimum samples
                    with st.spinner("🎵 Transcribing recorded audio..."):
                        # Convert audio data to numpy array
                        audio_array = np.array(st.session_state.stt_model.audio_data, dtype=np.float32)
                        
                        success, transcription, processing_time = st.session_state.stt_model.transcribe_audio(mic_audio=audio_array)
                        
                        if success:
                            st.success(f"✅ Transcription completed in {processing_time} seconds!")
                            
                            # Add to history
                            st.session_state.stt_transcriptions.append({
                                'type': 'recording',
                                'filename': 'Microphone Recording',
                                'transcription': transcription,
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                                'processing_time': processing_time
                            })
                            
                            st.markdown("### 📝 Transcription Result:")
                            st.text_area(
                                "Transcribed Text:",
                                transcription,
                                height=200,
                                key=f"transcription_result_{len(st.session_state.stt_transcriptions)}"
                            )
                            
                            # Clear audio data
                            st.session_state.stt_model.clear_audio_data()
                        else:
                            st.error(f"❌ {transcription}")
                else:
                    st.warning("⚠️ Not enough audio recorded. Please record for at least 2-3 seconds.")
        
        with col2:
            if st.button("🗑️ Clear Recording"):
                st.session_state.stt_model.clear_audio_data()
                st.success("✅ Recording cleared!")
    
    # Display transcription history
    if st.session_state.stt_transcriptions:
        st.markdown("### 📚 Transcription History")
        
        # Add clear all button
        if st.button("🗑️ Clear All History"):
            st.session_state.stt_transcriptions = []
            st.rerun()
        
        for i, trans in enumerate(reversed(st.session_state.stt_transcriptions)):
            actual_index = len(st.session_state.stt_transcriptions) - 1 - i
            with st.expander(f"📝 {trans['filename']} - {trans['timestamp']}"):
                st.text_area(
                    "Transcription:",
                    trans['transcription'],
                    height=150,
                    key=f"history_transcription_{actual_index}",
                    disabled=True
                )
                st.caption(f"⏱️ Processing time: {trans['processing_time']} seconds | Type: {trans['type']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📋 Copy", key=f"copy_{actual_index}"):
                        st.code(trans['transcription'])
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_transcription_{actual_index}"):
                        st.session_state.stt_transcriptions.pop(actual_index)
                        st.rerun()

if __name__ == "__main__":
    create_speech_to_text_interface()