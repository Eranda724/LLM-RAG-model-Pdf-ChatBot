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

    def transcribe_audio(self, audio_file=None, mic_audio=None) -> Tuple[bool, str, float]:
        """Transcribe audio from either file upload or microphone"""
        if not self.model_loaded or self.model is None:
            return False, "Model not loaded. Please load the model first.", 0.0

        try:
            # Determine which audio source to use
            if mic_audio is not None:
                audio_data = mic_audio
                file_extension = '.wav'
            elif audio_file is not None:
                audio_data = audio_file.read()
                file_extension = os.path.splitext(audio_file.name)[1].lower() or '.wav'
            else:
                return False, "No audio provided", 0.0

            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(tempfile.gettempdir(), 'whisper_audio')
            os.makedirs(temp_dir, exist_ok=True)

            # Save to temporary file with unique name
            temp_filename = f"whisper_{time.time()}_{os.getpid()}{file_extension}"
            temp_path = os.path.join(temp_dir, temp_filename)

            try:
                with open(temp_path, 'wb') as f:
                    f.write(audio_data)

                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                    return False, "Failed to create temporary audio file.", 0.0

                # Transcribe
                start_time = time.time()
                result: Dict[str, Any] = self.model.transcribe(temp_path)
                end_time = time.time()
                
                processing_time = round(end_time - start_time, 2)
                transcription = str(result.get("text", "")).strip()

            finally:
                # Clean up temp file
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_path}: {str(e)}")

            if not transcription:
                return False, "No text was transcribed. Please check your audio.", processing_time

            return True, transcription, processing_time

        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return False, f"Error transcribing audio: {str(e)}", 0.0

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
            help="Larger models are more accurate but slower"
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
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record Audio"])
    
    with tab1:
        st.markdown("### 📁 Upload Audio File")
        uploaded_audio = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
            key="stt_file_upload"
        )
        
        if uploaded_audio:
            st.info(f"📁 File: {uploaded_audio.name} ({len(uploaded_audio.getvalue()):,} bytes)")
            
            if st.session_state.stt_model.model_loaded:
                if st.button("🎵 Transcribe Audio File", type="primary"):
                    with st.spinner("🎵 Transcribing audio..."):
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
                        else:
                            st.error(transcription)
    
    with tab2:
        st.markdown("### 🎤 Record Audio")
        st.info("Click 'START' to begin recording. Click 'STOP' when finished.")
        
        def audio_frames_callback(frame: av.AudioFrame) -> av.AudioFrame:
            """Process audio frames from WebRTC stream"""
            # Convert audio frame to numpy array
            audio_data = frame.to_ndarray().flatten().astype(np.float32)
            st.session_state.stt_model.audio_data.extend(audio_data.tolist())
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
        
        if webrtc_ctx.state.playing and st.session_state.stt_model.model_loaded:
            if st.button("🎵 Transcribe Recording", type="primary"):
                if len(st.session_state.stt_model.audio_data) > 0:
                    with st.spinner("🎵 Transcribing audio..."):
                        # Convert audio data to bytes
                        audio_array = np.array(st.session_state.stt_model.audio_data, dtype=np.float32)
                        audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
                        
                        success, transcription, processing_time = st.session_state.stt_model.transcribe_audio(mic_audio=audio_bytes)
                        
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
                            st.error(transcription)
                else:
                    st.warning("No audio recorded yet. Click 'START' to begin recording.")
    
    # Display transcription history
    if st.session_state.stt_transcriptions:
        st.markdown("### 📚 Transcription History")
        
        for i, trans in enumerate(st.session_state.stt_transcriptions):
            with st.expander(f"📝 {trans['filename']} - {trans['timestamp']}"):
                st.text_area(
                    "Transcription:",
                    trans['transcription'],
                    height=150,
                    key=f"history_transcription_{i}",
                    disabled=True
                )
                st.caption(f"⏱️ Processing time: {trans['processing_time']} seconds")
                if st.button("🗑️ Remove", key=f"remove_transcription_{i}"):
                    st.session_state.stt_transcriptions.pop(i)
                    st.rerun()

if __name__ == "__main__":
    create_speech_to_text_interface() 