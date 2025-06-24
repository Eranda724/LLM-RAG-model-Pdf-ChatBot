import streamlit as st
import warnings
import logging
from model import PDFChatModel
from vector import process_csv_file
from gpt_chatbot import gpt_chatbot
import base64
import uuid
from typing import Dict, List, Any
import time

# Import speech functionality modules
from speech_to_text import create_speech_to_text_interface
from text_to_speech import create_text_to_speech_interface
from meeting_transcriber import create_meeting_transcriber_interface

# LangChain imports
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Set page config
st.set_page_config(
    page_title="Multi-Document Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and alignments
st.markdown("""
<style>
    /* Main content styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f0f2f6;
        border-right: 2px solid #e0e0e0;
    }
    
    /* Navigation button styling */
    .nav-button {
        width: 100%;
        padding: 15px;
        margin: 10px 0;
        border: none;
        border-radius: 10px;
        background-color: #ffffff;
        color: #333333;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: left;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .nav-button:hover {
        background-color: #e3f2fd;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    
    .nav-button.active {
        background-color: #2196f3;
        color: white;
        border-left: 4px solid #1976d2;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .stChatMessage [data-testid="stChatMessageContent"] {
        padding: 1rem;
    }
    
    .stStatus {
        border-radius: 0.5rem;
    }
    
    .stSpinner {
        border-radius: 0.5rem;
    }
    
    /* PDF container styling */
    .pdf-container {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 0;
        margin: 0;
        height: calc(100vh - 200px);
        overflow-y: auto;
    }
    
    /* Chat container styling */
    [data-testid="stVerticalBlock"] > div:has(> div.stChatMessage) {
        max-height: calc(100vh - 200px);
        overflow-y: auto;
        padding-right: 10px;
    }
    
    .chat-container {
        padding-bottom: 80px;
    }
    
    /* File selection styling */
    .file-selector {
        border: 2px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    
    .selected-file {
        border-color: #007bff;
        background-color: #e7f3ff;
    }
    
    /* File button styling */
    .file-button {
        margin: 0.25rem;
        padding: 0.5rem 1rem;
        border: 2px solid #007bff;
        border-radius: 0.25rem;
        background-color: white;
        color: #007bff;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .file-button:hover {
        background-color: #007bff;
        color: white;
    }
    
    .file-button.selected {
        background-color: #007bff;
        color: white;
    }
    
    /* Page header styling */
    .page-header {
        background: linear-gradient(90deg, #2196f3, #21cbf3);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .page-header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    
    .page-header p {
        margin: 10px 0 0 0;
        font-size: 1.2em;
        opacity: 0.9;
    }
    
    /* Custom text input styling */
    .custom-text-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        background-color: #f8f9fa;
    }
    
    .text-input-area {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 20px 0;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
    }
    
    /* File upload area styling */
    .upload-area {
        border: 2px dashed #007bff;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background-color: #f8f9fa;
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #0056b3;
        background-color: #e7f3ff;
    }
    
    /* Status indicators */
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .status-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Column alignment improvements */
    .stColumn {
        display: flex;
        align-items: stretch;
    }
    
    /* Button group styling */
    .button-group {
        display: flex;
        gap: 10px;
        margin: 10px 0;
        flex-wrap: wrap;
    }
    
    /* Tab styling improvements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0px 0px;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }
    
    /* Custom text expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 15px;
        margin: 5px 0;
        background-color: white;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e0e0e0;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #007bff;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f8f9fa;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* Input styling */
    .stTextInput input {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 8px 12px;
    }
    
    /* Button styling improvements */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 2px solid #dee2e6;
    }
    
    /* Main content improvements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Responsive improvements */
    @media (max-width: 768px) {
        .stColumn {
            flex-direction: column;
        }
        
        .page-header h1 {
            font-size: 2em;
        }
        
        .section-header {
            font-size: 1.2em;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

if 'pdf_files' not in st.session_state:
    st.session_state.pdf_files = {}
if 'csv_files' not in st.session_state:
    st.session_state.csv_files = {}
if 'text_files' not in st.session_state:
    st.session_state.text_files = {}
if 'custom_texts' not in st.session_state:
    st.session_state.custom_texts = {}
if 'gpt_messages' not in st.session_state:
    st.session_state.gpt_messages = []

def display_pdf(pdf_base64, pdf_name):
    """Display PDF in the interface"""
    pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def setup_rag_chain(text_content: str, collection_name: str):
    """Setup RAG chain for text processing"""
    try:
        doc = Document(
            page_content=text_content,
            metadata={"source": "direct_text_input"}
        )
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
        except Exception as e:
            st.error("Ollama Embedding model failed to load.")
            st.stop()
        
        vector_store = Chroma.from_documents(
            documents=[doc],
            embedding=embeddings,
            collection_name=collection_name
        )
        
        llm = ChatOllama(model="llama3:8b", temperature=0.1)
        
        QUERY_PROMPT = ChatPromptTemplate.from_template("""You are an AI language model assistant. Your task is to generate 2-3
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.

        Original question: {question}

        Generate variations that:
        1. Use different keywords or synonyms
        2. Approach the topic from different angles
        3. Consider broader or more specific contexts
        """)
        
        retriever = MultiQueryRetriever.from_llm(
            vector_store.as_retriever(search_kwargs={"k": 5}),
            llm,
            prompt=QUERY_PROMPT
        )
        
        template = """You are a helpful AI assistant that answers questions based on the provided context.

        Instructions:
        1. Answer the question based ONLY on the following context
        2. If the context doesn't contain enough information to answer the question, say so clearly
        3. Provide specific details and examples when available in the context
        4. Be concise but comprehensive in your response
        5. If you find relevant information, cite it appropriately

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain, vector_store, True, "RAG chain setup successfully"
    except Exception as e:
        logging.error(f"Error setting up RAG chain: {str(e)}")
        return None, None, False, str(e)

def create_file_chat_interface(file_info: Dict, messages_key: str, file_key: str):
    """Create chat interface for individual files"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📄 File Preview")
        preview_key = f"{file_key}_preview"
        
        if st.button(f"👁️ Show {file_info['type'].upper()} Preview", key=f"{file_key}_preview_button"):
            if preview_key not in st.session_state:
                st.session_state[preview_key] = False
            st.session_state[preview_key] = not st.session_state[preview_key]
        
        if st.session_state.get(preview_key, False):
            if file_info['type'] == 'pdf':
                if file_info['processor'].processing_complete:
                    preview_result = file_info['processor'].display_pdf_preview(file_info['file'])
                    if preview_result["success"]:
                        display_pdf(preview_result["pdf_base64"], preview_result["pdf_name"])
                        st.markdown(f"**Total Pages:** {preview_result['total_pages']}")
                    else:
                        st.error(preview_result["error"])
            elif file_info['type'] == 'csv':
                try:
                    df = pd.read_csv(file_info['file'])
                    st.dataframe(df, height=400)
                except Exception as e:
                    st.error(f"Error displaying CSV: {str(e)}")
            elif file_info['type'] == 'text':
                st.text_area(
                    "Text Content",
                    file_info['content'],
                    height=400,
                    disabled=True,
                    key=f"text_preview_{file_key}"
                )
    
    with col2:
        st.markdown("### 💬 Chat Interface")
        
        # Process file if not already processed
        if not file_info.get('processed', False):
            with st.spinner(f"🔄 Processing {file_info['type'].upper()}..."):
                if file_info['type'] == 'pdf':
                    success, message = file_info['processor'].process_pdf(file_info['file'])
                    if success:
                        success, message = file_info['processor'].setup_rag_chain()
                        if success:
                            st.success(f"✅ {file_info['type'].upper()} processed successfully!")
                            file_info['processed'] = True
                        else:
                            st.error(f"❌ {message}")
                    else:
                        st.error(f"❌ {message}")
                elif file_info['type'] == 'csv':
                    retriever, success, message = process_csv_file(file_info['file'])
                    if success:
                        file_info['retriever'] = retriever
                        file_info['processed'] = True
                        st.success(f"✅ {file_info['type'].upper()} processed successfully!")
                    else:
                        st.error(f"❌ {message}")
                elif file_info['type'] == 'text':
                    st.success(f"✅ {file_info['type'].upper()} processed successfully!")
                    file_info['processed'] = True
        
        # Chat interface
        if file_info.get('processed', False):
            if messages_key not in st.session_state:
                st.session_state[messages_key] = []
            
            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message in st.session_state[messages_key]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input(f"Ask about your {file_info['type'].upper()}", key=f"{file_key}_chat_input"):
                st.session_state[messages_key].append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.status("🤔 Thinking...", expanded=True) as status:
                        if file_info['type'] == 'pdf':
                            success, response, processing_time = file_info['processor'].get_response(prompt)
                            if success:
                                status.update(label=f"✅ Response generated in {processing_time} seconds", state="complete")
                                st.markdown(response)
                                st.session_state[messages_key].append({"role": "assistant", "content": response})
                            else:
                                st.error(response)
                        elif file_info['type'] == 'text':
                            try:
                                start_time = time.time()
                                response = file_info['chain'].invoke(prompt)
                                end_time = time.time()
                                processing_time = round(end_time - start_time, 2)
                                status.update(label=f"✅ Response generated in {processing_time} seconds", state="complete")
                                st.markdown(response)
                                st.session_state[messages_key].append({"role": "assistant", "content": response})
                            except Exception as e:
                                st.error(f"❌ Error generating response: {str(e)}")
                        elif file_info['type'] == 'csv':
                            try:
                                retriever = file_info.get('retriever')
                                if retriever is None:
                                    response = "❌ CSV retriever not initialized. Please try processing the file again."
                                    st.error(response)
                                    st.session_state[messages_key].append({"role": "assistant", "content": response})
                                else:
                                    docs = retriever.get_relevant_documents(prompt)
                                    response = "Based on the CSV data:\n\n"
                                    for doc in docs:
                                        response += f"- {doc.page_content}\n\n"
                                    status.update(label="✅ Response generated", state="complete")
                                    st.markdown(response)
                                    st.session_state[messages_key].append({"role": "assistant", "content": response})
                            except Exception as e:
                                error_msg = f"❌ Error generating response: {str(e)}"
                                st.error(error_msg)
                                st.session_state[messages_key].append({"role": "assistant", "content": error_msg})
        else:
            st.info(f"⏳ Please wait while the {file_info['type'].upper()} is being processed...")

def create_mix_analysis_interface(file_type: str, files_list: List):
    """Create mix analysis interface for multiple files"""
    st.markdown(f"## 🔍 {file_type.upper()} Mix Analysis")
    st.markdown(f"### Select {file_type.upper()} files for analysis")
    
    selected_key = f"selected_{file_type}_files"
    if selected_key not in st.session_state:
        st.session_state[selected_key] = []
    
    st.markdown(f"**Available {file_type.upper()} Files:**")
    
    # File selection grid
    with st.container():
        cols = st.columns(min(len(files_list), 4))
        for idx, (file_key, file_info) in enumerate(files_list):
            col_idx = idx % 4
            with cols[col_idx]:
                is_selected = file_key in st.session_state[selected_key]
                button_label = f"{file_info['tab_name']}\n({file_info['name'][:20]}...)" if len(file_info['name']) > 20 else f"{file_info['tab_name']}\n({file_info['name']})"
                
                if st.button(
                    button_label,
                    key=f"select_{file_key}_mix",
                    help=f"Click to {'deselect' if is_selected else 'select'} {file_info['name']}",
                    type="primary" if is_selected else "secondary"
                ):
                    if file_key in st.session_state[selected_key]:
                        st.session_state[selected_key].remove(file_key)
                    else:
                        st.session_state[selected_key].append(file_key)
    
    # Show selected files
    if st.session_state[selected_key]:
        st.markdown("### Selected Files:")
        selected_info = []
        for file_key in st.session_state[selected_key]:
            file_info = next(info for key, info in files_list if key == file_key)
            selected_info.append(f"📄 **{file_info['tab_name']}**: {file_info['name']}")
        st.markdown("\n".join(selected_info))
        
        # Start analysis button
        if st.button(f"🚀 Start {file_type.upper()} Analysis", type="primary", key=f"analyze_{file_type}"):
            all_processed = True
            with st.spinner(f"🔄 Processing selected {file_type.upper()} files..."):
                for file_key in st.session_state[selected_key]:
                    file_info = next(info for key, info in files_list if key == file_key)
                    if not file_info.get('processed', False):
                        if file_info['type'] == 'pdf':
                            success, message = file_info['processor'].process_pdf(file_info['file'])
                            if success:
                                success, message = file_info['processor'].setup_rag_chain()
                                if success:
                                    file_info['processed'] = True
                                else:
                                    st.error(f"❌ Error processing {file_info['name']}: {message}")
                                    all_processed = False
                            else:
                                st.error(f"❌ Error processing {file_info['name']}: {message}")
                                all_processed = False
                        elif file_info['type'] == 'csv':
                            retriever, success, message = process_csv_file(file_info['file'])
                            if success:
                                file_info['retriever'] = retriever
                                file_info['processed'] = True
                            else:
                                st.error(f"❌ Error processing {file_info['name']}: {message}")
                                all_processed = False
            
            if all_processed:
                st.success(f"✅ All selected {file_type.upper()} files processed successfully!")
        
        # Mix analysis chat
        if st.session_state[selected_key] and all(
            next(info for key, info in files_list if key == file_key).get('processed', False)
            for file_key in st.session_state[selected_key]
        ):
            st.markdown(f"### {file_type.upper()} Mix Analysis Chat")
            mix_messages_key = f"mix_messages_{file_type}"
            if mix_messages_key not in st.session_state:
                st.session_state[mix_messages_key] = []
            
            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message in st.session_state[mix_messages_key]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input(f"Ask questions about your selected {file_type.upper()} files", key=f"mix_{file_type}_chat_input"):
                st.session_state[mix_messages_key].append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.status(f"🔍 Analyzing across selected {file_type.upper()} files...", expanded=True) as status:
                        try:
                            response = f"## {file_type.upper()} Mix Analysis Results:\n\n"
                            for file_key in st.session_state[selected_key]:
                                file_info = next(info for key, info in files_list if key == file_key)
                                response += f"### From {file_info['tab_name']} ({file_info['name']}):\n"
                                
                                if file_info['type'] == 'pdf':
                                    success, pdf_response, _ = file_info['processor'].get_response(prompt)
                                    if success:
                                        response += f"{pdf_response}\n\n"
                                    else:
                                        response += f"❌ Error getting response: {pdf_response}\n\n"
                                elif file_info['type'] == 'csv':
                                    try:
                                        retriever = file_info.get('retriever')
                                        if retriever is None:
                                            response += "❌ CSV retriever not initialized. Please try processing the file again.\n\n"
                                        else:
                                            docs = retriever.get_relevant_documents(prompt)
                                            if docs:
                                                for doc in docs:
                                                    response += f"- {doc.page_content}\n"
                                                response += "\n"
                                            else:
                                                response += "No relevant data found.\n\n"
                                    except Exception as e:
                                        response += f"❌ Error getting response: {str(e)}\n\n"
                            
                            status.update(label="✅ Analysis complete", state="complete")
                            st.markdown(response)
                            st.session_state[mix_messages_key].append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
    else:
        st.info(f"📋 Please select at least one {file_type.upper()} file to start analysis.")

def create_gpt_chat_interface():
    """Create GPT chat interface"""
    st.markdown("## 🤖 GPT Chat Assistant")
    st.markdown("Ask me anything! I'm here to help with general questions and conversations.")
    
    # Status and controls
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.gpt_messages = []
            gpt_chatbot.clear_history()
    with col2:
        if gpt_chatbot.is_ready():
            st.success("✅ GPT ChatBot is ready!")
        else:
            st.error("❌ GPT ChatBot is not ready. Please check the model installation.")
    
    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.gpt_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "processing_time" in message:
                    st.caption(f"⏱️ Response time: {message['processing_time']} seconds")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything...", key="gpt_chat_input"):
        st.session_state.gpt_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.status("🤔 Thinking...", expanded=True) as status:
                try:
                    response, processing_time = gpt_chatbot.invoke(prompt)
                    if response.startswith("Error:"):
                        status.update(label="❌ Error occurred", state="error")
                        st.error(response)
                    else:
                        status.update(label=f"✅ Response generated in {processing_time} seconds", state="complete")
                        st.markdown(response)
                        st.session_state.gpt_messages.append({
                            "role": "assistant",
                            "content": response,
                            "processing_time": processing_time
                        })
                except Exception as e:
                    status.update(label="❌ Error occurred", state="error")
                    st.error(f"❌ Error generating response: {str(e)}")

def create_custom_text_interface():
    """Create custom text input and chat interface"""
    st.markdown("## 📝 Custom Text Chat")
    st.markdown("Add your own text content and chat with it!")
    
    # Custom text input section
    st.markdown("### ✍️ Add Custom Text")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        custom_text = st.text_area(
            "Enter your text content here:",
            height=200,
            placeholder="Paste or type your text content here...",
            key="custom_text_input"
        )
    with col2:
        st.markdown("### 📋 Text Options")
        text_title = st.text_input("Text Title:", placeholder="Enter a title for your text")
        
        if st.button("➕ Add Text", type="primary"):
            if custom_text.strip():
                text_id = str(uuid.uuid4())
                text_title = text_title if text_title else f"Custom Text {len(st.session_state.custom_texts) + 1}"
                
                # Setup RAG chain for the custom text
                chain, vector_store, success, message = setup_rag_chain(custom_text, f"custom_text_{text_id}")
                
                if success:
                    st.session_state.custom_texts[text_id] = {
                        'content': custom_text,
                        'title': text_title,
                        'chain': chain,
                        'vector_store': vector_store,
                        'processed': True,
                        'type': 'custom_text'
                    }
                    st.success(f"✅ Text '{text_title}' added successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Error processing text: {message}")
            else:
                st.error("❌ Please enter some text content.")
    
    # Display existing custom texts
    if st.session_state.custom_texts:
        st.markdown("### 📚 Your Custom Texts")
        
        for text_id, text_info in list(st.session_state.custom_texts.items()):
            with st.expander(f"📝 {text_info['title']}"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**Content Preview:** {text_info['content'][:100]}...")
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_custom_{text_id}"):
                        del st.session_state.custom_texts[text_id]
                        st.rerun()
                
                # Chat interface for this custom text
                if text_info.get('processed', False):
                    messages_key = f"custom_text_messages_{text_id}"
                    if messages_key not in st.session_state:
                        st.session_state[messages_key] = []
                    
                    # Display chat messages
                    for message in st.session_state[messages_key]:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                    
                    # Chat input
                    if prompt := st.chat_input(f"Ask about '{text_info['title']}'", key=f"custom_text_chat_{text_id}"):
                        st.session_state[messages_key].append({"role": "user", "content": prompt})
                        
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        
                        with st.chat_message("assistant"):
                            with st.status("🤔 Thinking...", expanded=True) as status:
                                try:
                                    start_time = time.time()
                                    response = text_info['chain'].invoke(prompt)
                                    end_time = time.time()
                                    processing_time = round(end_time - start_time, 2)
                                    status.update(label=f"✅ Response generated in {processing_time} seconds", state="complete")
                                    st.markdown(response)
                                    st.session_state[messages_key].append({"role": "assistant", "content": response})
                                except Exception as e:
                                    st.error(f"❌ Error generating response: {str(e)}")

def create_tabs_for_chat():
    """Create tabs for chat interfaces"""
    tab_names = ["🤖 GPT Chat Assistant"]
    tab_content = []
    
    # Always add GPT chat as first tab
    tab_content.append({
        'type': 'gpt_chat',
        'key': None,
        'info': None
    })
    
    if st.session_state.get('pdf_files') or st.session_state.get('csv_files') or st.session_state.get('text_files'):
        # Individual file tabs
        for pdf_key, pdf_info in st.session_state.get('pdf_files', {}).items():
            tab_names.append(f"📄 {pdf_info['tab_name']}")
            tab_content.append({
                'type': 'pdf_individual',
                'key': pdf_key,
                'info': pdf_info
            })
        
        for csv_key, csv_info in st.session_state.get('csv_files', {}).items():
            tab_names.append(f"📊 {csv_info['tab_name']}")
            tab_content.append({
                'type': 'csv_individual',
                'key': csv_key,
                'info': csv_info
            })
        
        for text_key, text_info in st.session_state.get('text_files', {}).items():
            tab_names.append(f"📝 {text_info['tab_name']}")
            tab_content.append({
                'type': 'text_individual',
                'key': text_key,
                'info': text_info
            })
        
        # Mix analysis tabs
        if len(st.session_state.get('pdf_files', {})) > 1:
            tab_names.append("🔍 PDF Mix Analysis")
            tab_content.append({
                'type': 'pdf_mix',
                'key': None,
                'info': None
            })
        
        if len(st.session_state.get('csv_files', {})) > 1:
            tab_names.append("🔍 CSV Mix Analysis")
            tab_content.append({
                'type': 'csv_mix',
                'key': None,
                'info': None
            })
        
        if len(st.session_state.get('text_files', {})) > 1:
            tab_names.append("🔍 Text Mix Analysis")
            tab_content.append({
                'type': 'text_mix',
                'key': None,
                'info': None
            })
    
    return tab_names, tab_content

def show_home_page():
    """Show the home page with welcome message and navigation"""
    st.markdown('<div class="page-header"><h1>🤖 Multi-Document Chat Assistant</h1><p>Your AI-powered document analysis and chat companion</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📚 Document Chat")
        st.markdown("Upload PDF, CSV, or text files and chat with them using AI.")
        if st.button("📄 Go to Document Chat", type="primary", key="home_docs"):
            st.session_state.current_page = 'chat_docs'
            st.rerun()
    
    with col2:
        st.markdown("### 📝 Custom Text")
        st.markdown("Add your own text content and have conversations with it.")
        if st.button("✍️ Go to Custom Text", type="primary", key="home_custom"):
            st.session_state.current_page = 'custom_text'
            st.rerun()
    
    with col3:
        st.markdown("### 🤖 GPT Assistant")
        st.markdown("Chat with our general-purpose AI assistant for any questions.")
        if st.button("💬 Go to GPT Chat", type="primary", key="home_gpt"):
            st.session_state.current_page = 'gpt_chat'
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. **📚 Document Chat**: Upload files and ask questions about their content
    2. **📝 Custom Text**: Add your own text and chat with it
    3. **🤖 GPT Assistant**: General AI conversations and help
    4. **🔍 Mix Analysis**: Compare multiple files of the same type
    """)

def show_document_chat_interface():
    """Show the document chat interface"""
    st.markdown('<h2 class="section-header">📄 Document Chat</h2>', unsafe_allow_html=True)
    
    # File summary
    st.markdown("### 📊 File Summary")
    pdf_count = len(st.session_state.get('pdf_files', {}))
    csv_count = len(st.session_state.get('csv_files', {}))
    text_count = len(st.session_state.get('text_files', {}))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 PDF Files", pdf_count)
    with col2:
        st.metric("📊 CSV Files", csv_count)
    with col3:
        st.metric("📝 Text Files", text_count)
    
    if pdf_count == 0 and csv_count == 0 and text_count == 0:
        st.info("📁 No files uploaded yet. Upload files below to get started!")
    
    # File upload section
    st.markdown("### 📁 Upload Files")
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a file to upload",
        type=["pdf", "csv", "txt"],
        key="main_file_uploader",
        help="Supported formats: PDF, CSV, TXT"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    def detect_file_type(file):
        """Detect file type based on extension"""
        if file is None:
            return None
        file_extension = file.name.lower().split('.')[-1]
        if file_extension == 'pdf':
            return 'pdf'
        elif file_extension == 'csv':
            return 'csv'
        elif file_extension == 'txt':
            return 'text'
        else:
            return None
    
    # Process uploaded file
    if uploaded_file is not None:
        file_type = detect_file_type(uploaded_file)
        if file_type is None:
            st.error("❌ Unsupported file type. Please upload a PDF, CSV, or TXT file.")
        else:
            file_key = f"{file_type}_{len(st.session_state.get(f'{file_type}_files', {})) + 1}"
            if f'{file_type}_files' not in st.session_state:
                st.session_state[f'{file_type}_files'] = {}
            
            # Check if file already exists
            file_exists = False
            for key, info in st.session_state[f'{file_type}_files'].items():
                if info['name'] == uploaded_file.name and info['size'] == uploaded_file.size:
                    file_exists = True
                    break
            
            if not file_exists:
                # Generate tab name
                existing_tabs = [info['tab_name'] for info in st.session_state[f'{file_type}_files'].values()]
                if file_type.upper() not in existing_tabs:
                    tab_name = file_type.upper()
                else:
                    numbers = [int(name.split("(")[1].split(")")[0]) for name in existing_tabs if "(" in name]
                    next_num = max(numbers) + 1 if numbers else 2
                    tab_name = f"{file_type.upper()} ({next_num})"
                
                # Create file info
                file_info = {
                    'file': uploaded_file,
                    'type': file_type,
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'tab_name': tab_name,
                    'processed': False
                }
                
                if file_type == 'pdf':
                    file_info['processor'] = PDFChatModel()
                elif file_type == 'csv':
                    file_info['retriever'] = None
                elif file_type == 'text':
                    file_info['content'] = uploaded_file.read().decode('utf-8')
                    uploaded_file.seek(0)  # Reset file pointer
                
                st.session_state[f'{file_type}_files'][file_key] = file_info
                st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                st.rerun()
    
    # Display uploaded files
    if st.session_state.get('pdf_files') or st.session_state.get('csv_files') or st.session_state.get('text_files'):
        st.markdown("### 📋 Uploaded Files")
        
        # Text files
        if st.session_state.get('text_files'):
            st.markdown("#### 📝 Text Files")
            for file_key, file_info in list(st.session_state['text_files'].items()):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"📝 **{file_info['tab_name']}**: {file_info['name']}")
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_{file_key}"):
                        del st.session_state['text_files'][file_key]
                        st.rerun()
        
        # PDF files
        if st.session_state.get('pdf_files'):
            st.markdown("#### 📄 PDF Files")
            for file_key, file_info in list(st.session_state['pdf_files'].items()):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"📄 **{file_info['tab_name']}**: {file_info['name']}")
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_{file_key}"):
                        del st.session_state['pdf_files'][file_key]
                        st.rerun()
        
        # CSV files
        if st.session_state.get('csv_files'):
            st.markdown("#### 📊 CSV Files")
            for file_key, file_info in list(st.session_state['csv_files'].items()):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"📊 **{file_info['tab_name']}**: {file_info['name']}")
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_{file_key}"):
                        del st.session_state['csv_files'][file_key]
                        st.rerun()
    
    # Create tabs
    tab_names, tab_content = create_tabs_for_chat()
    tabs = st.tabs(tab_names)
    
    for i, tab_info in enumerate(tab_content):
        with tabs[i]:
            tab_type = tab_info['type']
            file_key = tab_info['key']
            file_info = tab_info['info']
            
            if tab_type == 'gpt_chat':
                create_gpt_chat_interface()
            elif tab_type == 'pdf_individual' and file_info is not None:
                st.markdown(f"## 📄 {file_info['tab_name']} - {file_info['name']}")
                messages_key = f"messages_pdf_{file_key}"
                create_file_chat_interface(file_info, messages_key, file_key)
            elif tab_type == 'csv_individual' and file_info is not None:
                st.markdown(f"## 📊 {file_info['tab_name']} - {file_info['name']}")
                messages_key = f"messages_csv_{file_key}"
                create_file_chat_interface(file_info, messages_key, file_key)
            elif tab_type == 'text_individual' and file_info is not None:
                st.markdown(f"## 📝 {file_info['tab_name']} - {file_info['name']}")
                messages_key = f"messages_text_{file_key}"
                create_file_chat_interface(file_info, messages_key, file_key)
            elif tab_type == 'pdf_mix':
                pdf_files_list = list(st.session_state.get('pdf_files', {}).items())
                create_mix_analysis_interface('pdf', pdf_files_list)
            elif tab_type == 'csv_mix':
                csv_files_list = list(st.session_state.get('csv_files', {}).items())
                create_mix_analysis_interface('csv', csv_files_list)
            elif tab_type == 'text_mix':
                text_files_list = list(st.session_state.get('text_files', {}).items())
                create_mix_analysis_interface('text', text_files_list)
    
    if not (st.session_state.get('pdf_files') or st.session_state.get('csv_files') or st.session_state.get('text_files')):
        st.info("👆 Upload files above to analyze them, or use the GPT Chat Assistant for general questions!")

# Sidebar Navigation
with st.sidebar:
    st.markdown("# 🧭 Navigation")
    st.markdown("---")
    
    # Navigation buttons with better styling
    nav_options = [
        ("🏠 Home", "home"),
        ("📚 Document Chat", "chat_docs"),
        ("📝 Custom Text", "custom_text"),
        ("🤖 GPT Assistant", "gpt_chat"),
        ("🔊 Text Speech", "text_speech")
    ]
    
    for label, page in nav_options:
        if st.button(
            label,
            key=f"nav_{page}",
            help=f"Navigate to {label}",
            type="primary" if st.session_state.current_page == page else "secondary"
        ):
            st.session_state.current_page = page
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Current Status")
    
    # File counts
    pdf_count = len(st.session_state.get('pdf_files', {}))
    csv_count = len(st.session_state.get('csv_files', {}))
    text_count = len(st.session_state.get('text_files', {}))
    custom_count = len(st.session_state.get('custom_texts', {}))
    
    st.markdown(f"📄 PDFs: {pdf_count}")
    st.markdown(f"📊 CSVs: {csv_count}")
    st.markdown(f"📝 Texts: {text_count}")
    st.markdown(f"✍️ Custom: {custom_count}")

# Main content area
if st.session_state.current_page == 'home':
    show_home_page()
elif st.session_state.current_page == 'chat_docs':
    show_document_chat_interface()
elif st.session_state.current_page == 'custom_text':
    create_custom_text_interface()
elif st.session_state.current_page == 'gpt_chat':
    st.markdown('<h2 class="section-header">🤖 GPT Chat Assistant</h2>', unsafe_allow_html=True)
    create_gpt_chat_interface()
elif st.session_state.current_page == 'text_speech':
    st.markdown('<h2 class="section-header">🔊 Text Speech Interface</h2>', unsafe_allow_html=True)
    
    # Create tabs for speech functionality
    speech_tab1, speech_tab2, speech_tab3 = st.tabs(["🎤 Speech to Text", "🔊 Text to Speech", "🎙️ Meeting Transcriber"])
    
    with speech_tab1:
        create_speech_to_text_interface()
    
    with speech_tab2:
        create_text_to_speech_interface()
        
    with speech_tab3:
        create_meeting_transcriber_interface()

# Footer
st.markdown("---")
st.markdown("💡 **Tips:**")
st.markdown("- Use the **🤖 GPT Chat Assistant** for general questions and conversations")
st.markdown("- Upload multiple files to create individual chat sessions for each file")
st.markdown("- Add custom text content for personalized conversations")
st.markdown("- Mix Analysis tabs appear when you have 2+ files of the same type")
st.markdown("- Each file maintains its own separate chat history")