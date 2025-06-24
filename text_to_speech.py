import streamlit as st
from gtts import gTTS
import tempfile
import os
import time
import uuid
from typing import Tuple, Optional

class TextToSpeech:
    def __init__(self):
        """Initialize the Text to Speech converter"""
        self.supported_languages = {
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Russian': 'ru',
            'Japanese': 'ja',
            'Korean': 'ko',
            'Chinese (Simplified)': 'zh-cn',
            'Chinese (Traditional)': 'zh-tw',
            'Arabic': 'ar',
            'Hindi': 'hi',
            'Dutch': 'nl',
            'Swedish': 'sv',
            'Norwegian': 'no',
            'Danish': 'da',
            'Finnish': 'fi',
            'Polish': 'pl',
            'Turkish': 'tr'
        }
    
    def text_to_speech(self, text: str, language: str = 'en', slow: bool = False) -> Tuple[bool, str, float]:
        """Convert text to speech"""
        if not text.strip():
            return False, "Please enter some text to convert to speech.", 0.0
        
        try:
            # Generate a unique filename
            filename = f"tts_{uuid.uuid4()}.mp3"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            
            # Convert text to speech
            start_time = time.time()
            tts = gTTS(text=text, lang=language, slow=slow)
            tts.save(filepath)
            end_time = time.time()
            
            processing_time = round(end_time - start_time, 2)
            return True, filepath, processing_time
            
        except Exception as e:
            return False, f"❌ Error generating speech: {str(e)}", 0.0
    
    def get_language_code(self, language_name: str) -> str:
        """Get language code from language name"""
        return self.supported_languages.get(language_name, 'en')
    
    def get_language_names(self) -> list:
        """Get list of supported language names"""
        return list(self.supported_languages.keys())

def create_text_to_speech_interface():
    """Create the Text to Speech interface"""
    st.markdown("## 🔊 Text to Speech")
    st.markdown("Convert your text into natural-sounding speech using AI.")
    
    # Initialize session state
    if 'tts_converter' not in st.session_state:
        st.session_state.tts_converter = TextToSpeech()
    
    if 'tts_generations' not in st.session_state:
        st.session_state.tts_generations = []
    
    # Text input section
    st.markdown("### 📝 Text Input")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to convert to speech:",
            height=150,
            placeholder="Type or paste your text here...",
            help="Enter the text you want to convert to speech"
        )
    
    with col2:
        st.markdown("### 🎛️ Settings")
        
        # Language selection
        language_names = st.session_state.tts_converter.get_language_names()
        selected_language = st.selectbox(
            "Language:",
            language_names,
            index=0,
            help="Select the language for speech synthesis"
        )
        
        # Speed control
        slow_speech = st.checkbox(
            "Slow Speech",
            value=False,
            help="Enable for slower, more pronounced speech"
        )
        
        # Voice options (placeholder for future enhancements)
        st.markdown("**Voice Options:**")
        st.info("🎤 Additional voice options coming soon!")
    
    # Generate speech button
    if st.button("🔊 Generate Speech", type="primary", disabled=not text_input.strip()):
        if text_input.strip():
            with st.spinner("🔊 Generating speech..."):
                language_code = st.session_state.tts_converter.get_language_code(selected_language)
                success, audio_path, processing_time = st.session_state.tts_converter.text_to_speech(
                    text_input, language_code, slow_speech
                )
                
                if success:
                    st.success(f"✅ Speech generated in {processing_time} seconds!")
                    
                    # Add to history
                    st.session_state.tts_generations.append({
                        'text': text_input,
                        'language': selected_language,
                        'audio_path': audio_path,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'processing_time': processing_time,
                        'slow': slow_speech
                    })
                    
                    # Display audio player
                    st.markdown("### 🎵 Generated Audio")
                    
                    # Read and display the audio file
                    with open(audio_path, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                    
                    st.audio(audio_bytes, format='audio/mp3')
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Audio",
                        data=audio_bytes,
                        file_name=f"speech_{uuid.uuid4().hex[:8]}.mp3",
                        mime="audio/mp3"
                    )
                    
                    # Display text preview
                    with st.expander("📝 Text Preview"):
                        st.text_area(
                            "Generated Text:",
                            text_input,
                            height=100,
                            disabled=True
                        )
                        st.caption(f"Language: {selected_language} | Speed: {'Slow' if slow_speech else 'Normal'}")
                else:
                    st.error(audio_path)
        else:
            st.warning("⚠️ Please enter some text before generating speech.")
    
    # Display generation history
    if st.session_state.tts_generations:
        st.markdown("### 📚 Generation History")
        
        for i, gen in enumerate(st.session_state.tts_generations):
            with st.expander(f"🔊 {gen['language']} - {gen['timestamp']}"):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Display text preview
                    st.text_area(
                        "Text:",
                        gen['text'][:200] + "..." if len(gen['text']) > 200 else gen['text'],
                        height=100,
                        key=f"history_text_{i}",
                        disabled=True
                    )
                    
                    # Display audio player
                    try:
                        with open(gen['audio_path'], 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/mp3')
                    except FileNotFoundError:
                        st.warning("⚠️ Audio file not found")
                    
                    st.caption(f"⏱️ Processing time: {gen['processing_time']} seconds | Speed: {'Slow' if gen['slow'] else 'Normal'}")
                
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_generation_{i}"):
                        # Remove audio file
                        try:
                            os.remove(gen['audio_path'])
                        except:
                            pass
                        # Remove from history
                        st.session_state.tts_generations.pop(i)
                        st.rerun()
    
    # Tips and information
    st.markdown("---")
    st.markdown("### 💡 Tips for Better Speech Generation")
    st.markdown("""
    - **Clear Text**: Use proper punctuation and grammar for better pronunciation
    - **Language Selection**: Choose the correct language for accurate pronunciation
    - **Text Length**: Longer texts take more time to process
    - **Special Characters**: Avoid excessive use of symbols or special characters
    - **Speed Control**: Use slow speech for important announcements or learning purposes
    """)
    
    st.markdown("### 🔧 Technical Information")
    st.markdown("""
    - **Engine**: Google Text-to-Speech (gTTS)
    - **Supported Languages**: 20+ languages including major world languages
    - **Audio Format**: MP3
    - **Processing**: Cloud-based processing via Google's TTS service
    - **Quality**: High-quality, natural-sounding speech synthesis
    """)
    
    st.markdown("### 🌐 Supported Languages")
    st.markdown("""
    The system supports multiple languages including:
    - **European**: English, Spanish, French, German, Italian, Portuguese, Dutch, Swedish, Norwegian, Danish, Finnish, Polish, Turkish
    - **Asian**: Japanese, Korean, Chinese (Simplified & Traditional), Hindi
    - **Other**: Russian, Arabic
    """) 