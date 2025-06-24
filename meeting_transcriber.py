import streamlit as st
import librosa
import numpy as np
from transformers.pipelines import pipeline
import requests
import json
import textwrap
import tempfile
import os
import time
import uuid

def initialize_asr():
    """Initialize the ASR pipeline"""
    return pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcribe_audio(audio_file, asr_pipeline):
    """Transcribe audio using Whisper"""
    audio, sr = librosa.load(audio_file, sr=16000)
    if len(audio) > 480000:
        result = asr_pipeline(audio.astype(np.float32), return_timestamps=True)
    else:
        result = asr_pipeline(audio.astype(np.float32))
    return result["text"]

def summarize_with_local_model(transcript):
    """Summarize using a simple extractive summarization approach"""
    try:
        # Split into sentences
        sentences = transcript.split('. ')
        if len(sentences) <= 3:
            return transcript
        
        # Simple extractive summarization
        # Take first sentence (usually contains context)
        # Take a sentence from middle (usually contains key points)
        # Take last sentence (usually contains conclusions)
        summary_sentences = [
            sentences[0],
            sentences[len(sentences)//2],
            sentences[-1]
        ]
        
        summary = '. '.join(summary_sentences) + '.'
        return f"📝 Basic Summary (Local Processing):\n\n{summary}"
        
    except Exception as e:
        return f"Error in local summarization: {str(e)}"

def summarize_with_openrouter(transcript):
    """Summarize transcript using OpenRouter API"""
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets.get("OPENROUTER_API_KEY")
        if not api_key:
            return summarize_with_local_model(transcript)

        prompt = (
            f"Please summarize the following meeting transcript in a detailed yet concise format."
            f"Highlight key decisions, action items, important discussions, and next steps.\n\n"
            f"Transcript:\n{transcript}"
        )
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://yourdomain.com",
                "X-Title": "AI Meeting Summarizer",
            },
            data=json.dumps({
                "model": "deepseek/deepseek-chat:free",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            })
        )
        if response.status_code == 200:
            summary = response.json()['choices'][0]['message']['content']
            return textwrap.fill(summary, width=100)
        else:
            # Fallback to local summarization if API fails
            return summarize_with_local_model(transcript)
    except Exception as e:
        # Fallback to local summarization on any error
        return summarize_with_local_model(transcript)

def create_meeting_transcriber_interface():
    """Create the meeting transcriber interface using Streamlit"""
    st.markdown("## 🎙️ Meeting Transcriber & Summarizer")
    st.markdown("Upload a meeting recording to get both transcription and AI-generated summary.")
    
    # Initialize session state
    if 'meeting_asr' not in st.session_state:
        with st.spinner("🔄 Loading Whisper model..."):
            st.session_state.meeting_asr = initialize_asr()
            st.success("✅ Whisper model loaded successfully!")
    
    if 'meeting_transcriptions' not in st.session_state:
        st.session_state.meeting_transcriptions = []
    
    # File upload section
    uploaded_audio = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
        key="meeting_audio_upload"
    )
    
    if uploaded_audio:
        st.info(f"📁 File: {uploaded_audio.name} ({len(uploaded_audio.getvalue()):,} bytes)")
        
        if st.button("🎵 Transcribe & Summarize", type="primary"):
            try:
                # Save uploaded file temporarily
                temp_dir = os.path.join(tempfile.gettempdir(), 'meeting_audio')
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"meeting_{time.time()}_{uploaded_audio.name}")
                
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_audio.getvalue())
                
                # Process the audio
                with st.spinner("🎵 Transcribing audio..."):
                    transcript = transcribe_audio(temp_path, st.session_state.meeting_asr)
                    st.success("✅ Transcription complete!")
                
                with st.spinner("🤔 Generating summary..."):
                    summary = summarize_with_openrouter(transcript)
                    st.success("✅ Summary generated!")
                
                # Add to history
                st.session_state.meeting_transcriptions.append({
                    'filename': uploaded_audio.name,
                    'transcript': transcript,
                    'summary': summary,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Display results
                st.markdown("### 📝 Transcription")
                st.text_area("Full Transcript:", transcript, height=200)
                
                st.markdown("### 📋 Summary")
                st.text_area("Generated Summary:", summary, height=150)
                
                # Cleanup
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
            except Exception as e:
                st.error(f"❌ Error processing audio: {str(e)}")
    
    # Display history
    if st.session_state.meeting_transcriptions:
        st.markdown("### 📚 Previous Transcriptions")
        
        for i, trans in enumerate(st.session_state.meeting_transcriptions):
            with st.expander(f"📝 {trans['filename']} - {trans['timestamp']}"):
                st.text_area(
                    "Transcript:",
                    trans['transcript'],
                    height=150,
                    key=f"history_transcript_{i}"
                )
                st.text_area(
                    "Summary:",
                    trans['summary'],
                    height=100,
                    key=f"history_summary_{i}"
                )
                if st.button("🗑️ Remove", key=f"remove_transcription_{i}"):
                    st.session_state.meeting_transcriptions.pop(i)
                    st.rerun()
    
    # Tips section
    with st.expander("💡 Tips & Information"):
        st.markdown("""
        - **Supported Formats**: WAV, MP3, M4A, FLAC, OGG
        - **Best Practices**:
            - Use clear audio recordings with minimal background noise
            - Keep file sizes reasonable for faster processing
            - For long meetings, consider breaking into smaller segments
        - **Features**:
            - Automatic speech recognition using Whisper
            - AI-powered summarization highlighting key points
            - Persistent history of transcriptions and summaries
        """)

# Run the interface
if __name__ == "__main__":
    create_meeting_transcriber_interface()