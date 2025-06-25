import streamlit as st
import librosa
import numpy as np
import uuid
import tempfile
import os
import time
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import whisper

class SpeechTranslator:
    def __init__(self):
        """Initialize the Speech Translator"""
        self.whisper_model = None
        self.sinhala_tokenizer = None
        self.sinhala_model = None
        self.singlish_pipe = None
        self.models_loaded = False
    
    def load_models(self):
        """Load all required models"""
        try:
            with st.spinner("🔄 Loading models..."):
                progress_bar = st.progress(0)
                
                # Load Whisper model directly
                st.write("Loading Whisper ASR model...")
                try:
                    self.whisper_model = whisper.load_model("small")
                except Exception as e:
                    st.error(f"Whisper model failed to load: {e}")
                    return False, f"❌ Error loading Whisper model: {str(e)}"
                progress_bar.progress(33)
                
                # Load English-Sinhala model
                st.write("Loading English-Sinhala translation model...")
                try:
                    self.sinhala_tokenizer = AutoTokenizer.from_pretrained("thilina/mt5-sinhalese-english")
                    self.sinhala_model = AutoModelForSeq2SeqLM.from_pretrained("thilina/mt5-sinhalese-english")
                except Exception as e:
                    st.warning(f"Could not load Sinhala model: {e}")
                    st.info("Using fallback translation method for Sinhala")
                    self.sinhala_tokenizer = None
                    self.sinhala_model = None
                progress_bar.progress(66)
                
                # Load Singlish-English model
                st.write("Loading Singlish-English translation model...")
                try:
                    from transformers import pipeline
                    self.singlish_pipe = pipeline("text2text-generation", model="raqdo09/singlish-to-english-synthetic")
                except Exception as e:
                    st.warning(f"Could not load Singlish model: {e}")
                    st.info("Using fallback translation method for Singlish")
                    self.singlish_pipe = None
                progress_bar.progress(100)
                
                self.models_loaded = True
                return True, "✅ Models loaded successfully!"
        except Exception as e:
            self.models_loaded = False
            return False, f"❌ Error loading models: {str(e)}"
    
    def translate_english_to_sinhala(self, english_text):
        """Translate English text to Sinhala using mt5 model or fallback"""
        try:
            if self.sinhala_tokenizer and self.sinhala_model:
                # Use the loaded model
                input_text = f"translate English to Sinhala: {english_text}"
                inputs = self.sinhala_tokenizer(
                    input_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                
                # Generate translation
                with torch.no_grad():
                    outputs = self.sinhala_model.generate(
                        **inputs,
                        max_length=256,
                        num_beams=4,
                        do_sample=True,
                        temperature=0.7,
                        early_stopping=True,
                        pad_token_id=self.sinhala_tokenizer.pad_token_id
                    )
                
                sinhala_text = self.sinhala_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return True, sinhala_text
            else:
                # Fallback: Simple word replacement (demo purposes)
                fallback_dict = {
                    "hello": "ආයුබෝවන්",
                    "good morning": "සුභ උදෑසනක්",
                    "thank you": "ස්තූතියි",
                    "yes": "ඔව්",
                    "no": "නැහැ",
                    "how are you": "ඔයා කොහොමද",
                    "what is your name": "ඔයාගේ නම මොකක්ද"
                }
                
                lower_text = english_text.lower()
                for eng, sin in fallback_dict.items():
                    if eng in lower_text:
                        return True, sin
                
                return True, f"[Fallback] {english_text} (Sinhala translation not available)"
                
        except Exception as e:
            return False, f"Translation error: {str(e)}"
    
    def translate_singlish_to_english(self, singlish_text):
        """Translate Singlish text to English using pipeline or fallback"""
        try:
            if self.singlish_pipe:
                result = self.singlish_pipe(
                    singlish_text, 
                    max_length=256, 
                    do_sample=True, 
                    temperature=0.7
                )
                return True, result[0]['generated_text']
            else:
                # Fallback: Simple corrections
                corrections = {
                    "lah": "",
                    "lor": "",
                    "meh": "",
                    "can": "yes",
                    "cannot": "no",
                    "confirm": "sure",
                    "steady": "good"
                }
                
                corrected = singlish_text
                for singlish, english in corrections.items():
                    corrected = corrected.replace(singlish, english)
                
                return True, f"[Fallback] {corrected.strip()}"
                
        except Exception as e:
            return False, f"Translation error: {str(e)}"
    
    def save_uploaded_file(self, uploaded_file):
        """Save uploaded file to temporary location"""
        try:
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, f"audio_{uuid.uuid4().hex}_{uploaded_file.name}")
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            return file_path
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return None
    
    def process_english_to_sinhala_audio(self, uploaded_file):
        """Process English audio and translate to Sinhala"""
        try:
            if uploaded_file is None:
                return None, None, None, "Please upload an audio file"
            
            if not self.whisper_model:
                return None, None, None, "Speech recognition model not loaded. Please load models first."
            
            # Save uploaded file
            file_path = self.save_uploaded_file(uploaded_file)
            if not file_path:
                return None, None, None, "Error saving uploaded file"
            
            try:
                # Speech recognition using Whisper (Audio to English text)
                result = self.whisper_model.transcribe(file_path)
                english_text = result["text"]
                
                if not english_text.strip():
                    return english_text, None, None, "No speech detected in audio"
                
                # Translation (English to Sinhala)
                success, sinhala_text = self.translate_english_to_sinhala(english_text)
                if not success:
                    return english_text, None, None, sinhala_text
                
                # Text-to-speech (Sinhala)
                output_filename = None
                try:
                    output_filename = f"sinhala_audio_{uuid.uuid4().hex}.mp3"
                    tts = gTTS(text=sinhala_text, lang='si')  # 'si' for Sinhala
                    tts.save(output_filename)
                    status = "Translation complete!"
                except Exception as tts_error:
                    st.warning(f"Sinhala TTS not available: {tts_error}")
                    status = "Translation complete! (Audio generation not supported for Sinhala)"
                
                return english_text, sinhala_text, output_filename, status
                
            finally:
                # Clean up temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)
            
        except Exception as e:
            return None, None, None, f"Error: {str(e)}"
    
    def process_singlish_to_english_audio(self, uploaded_file):
        """Process Singlish audio and translate to English"""
        try:
            if uploaded_file is None:
                return None, None, None, "Please upload an audio file"
            
            if not self.whisper_model:
                return None, None, None, "Speech recognition model not loaded. Please load models first."
            
            # Save uploaded file
            file_path = self.save_uploaded_file(uploaded_file)
            if not file_path:
                return None, None, None, "Error saving uploaded file"
            
            try:
                # Speech recognition using Whisper (Audio to Singlish text)
                result = self.whisper_model.transcribe(file_path)
                singlish_text = result["text"]
                
                if not singlish_text.strip():
                    return singlish_text, None, None, "No speech detected in audio"
                
                # Translation (Singlish to English)
                success, english_text = self.translate_singlish_to_english(singlish_text)
                if not success:
                    return singlish_text, None, None, english_text
                
                # Text-to-speech (English)
                output_filename = f"english_audio_{uuid.uuid4().hex}.mp3"
                try:
                    tts = gTTS(text=english_text, lang='en')
                    tts.save(output_filename)
                    status = "Translation complete!"
                except Exception as tts_error:
                    output_filename = None
                    status = f"Translation complete! (Audio generation failed: {tts_error})"
                
                return singlish_text, english_text, output_filename, status
                
            finally:
                # Clean up temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)
            
        except Exception as e:
            return None, None, None, f"Error: {str(e)}"

def get_current_time():
    """Get current timestamp"""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def create_speech_translator_interface():
    """Create the Speech Translator interface"""
    st.markdown("## 🌐 Speech Translator")
    st.markdown("Translate speech between English, Sinhala, and Singlish.")
    
    # Initialize session state
    if 'speech_translator' not in st.session_state:
        st.session_state.speech_translator = SpeechTranslator()
    
    if 'translation_history' not in st.session_state:
        st.session_state.translation_history = []
    
    # Model loading section
    if not st.session_state.speech_translator.models_loaded:
        st.info("Click below to load translation models (this may take a few minutes)")
        if st.button("🔄 Load Translation Models", type="primary"):
            success, message = st.session_state.speech_translator.load_models()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
                st.info("💡 If models fail to load, you can still use the interface with limited functionality.")
        return
    else:
        st.success("✅ Translation models are ready!")
    
    # Check if ASR is available
    if st.session_state.speech_translator.whisper_model is None:
        st.error("❌ Speech recognition model is not available. Please reload the models.")
        if st.button("🔄 Reload Models", type="primary"):
            success, message = st.session_state.speech_translator.load_models()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        return
    
    # Create tabs for different translation directions
    tab1, tab2 = st.tabs(["🇬🇧 English → සිංහල", "Singlish → 🇬🇧 English"])
    
    with tab1:
        st.markdown("### Upload English audio to get Sinhala translation")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'ogg'],
            key="english_audio_upload"
        )
        
        if uploaded_file:
            st.info(f"📁 File: {uploaded_file.name} ({len(uploaded_file.getvalue()):,} bytes)")
            
            if st.button("🔄 Translate to Sinhala", type="primary", key="en_translate"):
                with st.spinner("🎵 Processing audio and translating..."):
                    english_text, sinhala_text, audio_path, status = st.session_state.speech_translator.process_english_to_sinhala_audio(uploaded_file)
                    
                    if english_text is not None:
                        st.success(status)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**English Transcription:**")
                            st.text_area("", english_text, height=150, key="en_transcription")
                        with col2:
                            st.markdown("**Sinhala Translation:**")
                            st.text_area("", sinhala_text, height=150, key="si_translation")
                        
                        if audio_path and os.path.exists(audio_path):
                            st.audio(audio_path)
                        
                        # Add to history
                        st.session_state.translation_history.append({
                            'type': 'en_to_si',
                            'source_text': english_text,
                            'translated_text': sinhala_text,
                            'audio_path': audio_path,
                            'timestamp': get_current_time()
                        })
                    else:
                        st.error(status)
    
    with tab2:
        st.markdown("### Upload Singlish audio to get proper English translation")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'ogg'],
            key="singlish_audio_upload"
        )
        
        if uploaded_file:
            st.info(f"📁 File: {uploaded_file.name} ({len(uploaded_file.getvalue()):,} bytes)")
            
            if st.button("🔄 Translate to English", type="primary", key="sg_translate"):
                with st.spinner("🎵 Processing audio and translating..."):
                    singlish_text, english_text, audio_path, status = st.session_state.speech_translator.process_singlish_to_english_audio(uploaded_file)
                    
                    if singlish_text is not None:
                        st.success(status)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Singlish Transcription:**")
                            st.text_area("", singlish_text, height=150, key="sg_transcription")
                        with col2:
                            st.markdown("**English Translation:**")
                            st.text_area("", english_text, height=150, key="en_translation")
                        
                        if audio_path and os.path.exists(audio_path):
                            st.audio(audio_path)
                        
                        # Add to history
                        st.session_state.translation_history.append({
                            'type': 'sg_to_en',
                            'source_text': singlish_text,
                            'translated_text': english_text,
                            'audio_path': audio_path,
                            'timestamp': get_current_time()
                        })
                    else:
                        st.error(status)
    
    # Display translation history
    if st.session_state.translation_history:
        st.markdown("### 📚 Translation History")
        
        for i, trans in enumerate(st.session_state.translation_history):
            with st.expander(
                f"{'🇬🇧 → සිංහල' if trans['type'] == 'en_to_si' else 'Singlish → 🇬🇧'} - {trans['timestamp']}"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Text:**")
                    st.text_area(
                        "",
                        trans['source_text'],
                        height=100,
                        key=f"history_source_{i}"
                    )
                with col2:
                    st.markdown("**Translated Text:**")
                    st.text_area(
                        "",
                        trans['translated_text'],
                        height=100,
                        key=f"history_translated_{i}"
                    )
                
                if trans['audio_path'] and os.path.exists(trans['audio_path']):
                    st.audio(trans['audio_path'])
                
                if st.button("🗑️ Remove", key=f"remove_translation_{i}"):
                    st.session_state.translation_history.pop(i)
                    st.rerun()
    
    # Tips section
    with st.expander("💡 Tips & Information"):
        st.markdown("""
        - **Supported Audio Formats**: WAV, MP3, M4A, OGG
        - **Best Practices**:
            - Use clear audio recordings with minimal background noise
            - Speak clearly and at a moderate pace
            - Keep audio files under 30 seconds for better processing
        - **Model Information**:
            - Speech Recognition: Whisper (Small)
            - English-Sinhala: thilina/mt5-sinhalese-english
            - Singlish-English: raqdo09/singlish-to-english-synthetic
        - **Note**: Some models may use fallback methods if the primary model fails to load
        """)

# Main execution
if __name__ == "__main__":
    create_speech_translator_interface()