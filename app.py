import streamlit as st
import warnings
import logging
from model import PDFChatModel
import base64

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Set page config
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .stChatMessage [data-testid="stChatMessageContent"] {
        padding: 1rem;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .stStatus {
        border-radius: 0.5rem;
    }
    .stSpinner {
        border-radius: 0.5rem;
    }
    .pdf-container {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def display_pdf(pdf_base64, pdf_name):
    """Display PDF in the browser"""
    pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = PDFChatModel()
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Create two columns
col1, col2 = st.columns([1, 1])

# Left column - PDF Processing
with col1:
    st.markdown("## 📄 PDF Processing")
    uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])
    
    if uploaded_file is not None:
        # Display PDF preview
        preview_result = st.session_state.model.display_pdf_preview(uploaded_file)
        if preview_result["success"]:
            st.markdown("### PDF Preview")
            with st.container():
                st.markdown('<div class="pdf-container">', unsafe_allow_html=True)
                display_pdf(preview_result["pdf_base64"], preview_result["pdf_name"])
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f"**Total Pages:** {preview_result['total_pages']}")
        else:
            st.error(preview_result["error"])
        
        # Process PDF
        if not st.session_state.model.processing_complete:
            with st.spinner("Processing PDF..."):
                success, message = st.session_state.model.process_pdf(uploaded_file)
                if success:
                    success, message = st.session_state.model.setup_rag_chain()
                    if success:
                        st.success("PDF processed successfully! You can now ask questions.")
                    else:
                        st.error(message)
                else:
                    st.error(message)

# Right column - Chat Interface
with col2:
    st.markdown("## 💬 Chat Interface")
    
    if st.session_state.model.is_ready():
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get AI response
            with st.chat_message("assistant"):
                with st.status("Thinking...", expanded=True) as status:
                    success, response, processing_time = st.session_state.model.get_response(prompt)
                    if success:
                        status.update(label=f"Response generated in {processing_time} seconds", state="complete")
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error(response)
    else:
        st.info("Please upload a PDF file to start chatting.") 