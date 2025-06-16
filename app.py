import streamlit as st
import warnings
import logging
from model import PDFChatModel
from vector import process_csv_file
import base64
import uuid
from typing import Dict, List, Any

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Set page config
st.set_page_config(
    page_title="Multi-Document Chat Assistant",
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
    /* Add padding to bottom of chat container to prevent messages from being hidden behind input */
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
</style>
""", unsafe_allow_html=True)

def display_pdf(pdf_base64, pdf_name):
    """Display PDF in the browser"""
    pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def create_file_chat_interface(file_info: Dict, messages_key: str, file_key: str):
    """Create chat interface for a specific file"""
    
    # Create two columns for side-by-side layout
    col1, col2 = st.columns(2)
    
    # Left column - File Preview
    with col1:
        st.markdown("### File Preview")
        preview_key = f"{file_key}_preview"
        
        if st.button(f"{file_info['type'].upper()} Preview", key=f"{file_key}_preview_button"):
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
                    import pandas as pd
                    df = pd.read_csv(file_info['file'])
                    st.dataframe(df, height=600)
                except Exception as e:
                    st.error(f"Error displaying CSV: {str(e)}")
    
    # Right column - Chat Interface
    with col2:
        st.markdown("### Chat Interface")
        
        # Process file if not already processed
        if not file_info.get('processed', False):
            with st.spinner(f"Processing {file_info['type'].upper()}..."):
                if file_info['type'] == 'pdf':
                    success, message = file_info['processor'].process_pdf(file_info['file'])
                    if success:
                        success, message = file_info['processor'].setup_rag_chain()
                        if success:
                            st.success(f"{file_info['type'].upper()} processed successfully! You can now ask questions.")
                            file_info['processed'] = True
                        else:
                            st.error(message)
                    else:
                        st.error(message)
                elif file_info['type'] == 'csv':
                    retriever, success, message = process_csv_file(file_info['file'])
                    if success:
                        file_info['retriever'] = retriever
                        file_info['processed'] = True
                        st.success(f"{file_info['type'].upper()} processed successfully! You can now ask questions.")
                    else:
                        st.error(message)
        
        # Chat Interface
        if file_info.get('processed', False):
            # Initialize messages for this file if not exists
            if messages_key not in st.session_state:
                st.session_state[messages_key] = []
            
            # Create a container for chat history
            chat_container = st.container()
            with chat_container:
                # Display chat history
                for message in st.session_state[messages_key]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input(f"Ask a question about your {file_info['type'].upper()}", key=f"{file_key}_chat_input"):
                # Add user message to chat history
                st.session_state[messages_key].append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get AI response
                with st.chat_message("assistant"):
                    with st.status("Thinking...", expanded=True) as status:
                        if file_info['type'] == 'pdf':
                            success, response, processing_time = file_info['processor'].get_response(prompt)
                            if success:
                                status.update(label=f"Response generated in {processing_time} seconds", state="complete")
                                st.markdown(response)
                                st.session_state[messages_key].append({"role": "assistant", "content": response})
                            else:
                                st.error(response)
                        elif file_info['type'] == 'csv':
                            try:
                                docs = file_info['retriever'].get_relevant_documents(prompt)
                                response = "Based on the CSV data:\n\n"
                                for doc in docs:
                                    response += f"- {doc.page_content}\n\n"
                                
                                status.update(label="Response generated", state="complete")
                                st.markdown(response)
                                st.session_state[messages_key].append({"role": "assistant", "content": response})
                            except Exception as e:
                                st.error(f"Error generating response: {str(e)}")
        else:
            st.info(f"Please wait while the {file_info['type'].upper()} is being processed...")

def create_mix_analysis_interface(file_type: str, files_list: List):
    """Create mix analysis interface for a specific file type"""
    st.markdown(f"## 🔍 {file_type.upper()} Mix Analysis")
    st.markdown(f"### Select {file_type.upper()} files for analysis")
    
    # File selection interface
    selected_key = f"selected_{file_type}_files"
    if selected_key not in st.session_state:
        st.session_state[selected_key] = []
    
    st.markdown(f"**Available {file_type.upper()} Files:**")
    
    # Create a container for file selection buttons
    with st.container():
        # Create columns for file selection buttons
        cols = st.columns(min(len(files_list), 4))
        
        for idx, (file_key, file_info) in enumerate(files_list):
            col_idx = idx % 4
            with cols[col_idx]:
                # Create a button for each file with improved styling
                is_selected = file_key in st.session_state[selected_key]
                
                # Improved button label with file name and type
                button_label = f"{file_info['tab_name']}\n({file_info['name'][:20]}...)" if len(file_info['name']) > 20 else f"{file_info['tab_name']}\n({file_info['name']})"
                
                # Add custom styling for selected state
                button_style = """
                <style>
                    div[data-testid="stButton"] button {
                        width: 100%;
                        margin: 5px 0;
                        padding: 10px;
                        border-radius: 5px;
                        transition: all 0.3s ease;
                    }
                    div[data-testid="stButton"] button:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                    }
                </style>
                """
                st.markdown(button_style, unsafe_allow_html=True)
                
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
                    st.rerun()
    
    # Display selected files with improved styling
    if st.session_state[selected_key]:
        st.markdown("### Selected Files:")
        selected_info = []
        for file_key in st.session_state[selected_key]:
            file_info = next(info for key, info in files_list if key == file_key)
            selected_info.append(f"📄 **{file_info['tab_name']}**: {file_info['name']}")
        
        st.markdown("\n".join(selected_info))
        
        # Analyze button with improved styling
        st.markdown("""
        <style>
            div[data-testid="stButton"] button[kind="primaryFormSubmit"] {
                width: 100%;
                margin: 10px 0;
                padding: 15px;
                font-size: 1.2em;
                border-radius: 8px;
                background-color: #4CAF50;
                color: white;
                transition: all 0.3s ease;
            }
            div[data-testid="stButton"] button[kind="primaryFormSubmit"]:hover {
                background-color: #45a049;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button(f"🚀 Start {file_type.upper()} Analysis", type="primary", key=f"analyze_{file_type}"):
            # Process selected files if not already processed
            all_processed = True
            
            with st.spinner(f"Processing selected {file_type.upper()} files..."):
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
                                    st.error(f"Error processing {file_info['name']}: {message}")
                                    all_processed = False
                            else:
                                st.error(f"Error processing {file_info['name']}: {message}")
                                all_processed = False
                        elif file_info['type'] == 'csv':
                            retriever, success, message = process_csv_file(file_info['file'])
                            if success:
                                file_info['retriever'] = retriever
                                file_info['processed'] = True
                            else:
                                st.error(f"Error processing {file_info['name']}: {message}")
                                all_processed = False
            
            if all_processed:
                st.success(f"All selected {file_type.upper()} files processed successfully! You can now chat with them.")
        
        # Chat interface for mix analysis with improved styling
        if st.session_state[selected_key] and all(
            next(info for key, info in files_list if key == file_key).get('processed', False)
            for file_key in st.session_state[selected_key]
        ):
            st.markdown(f"### {file_type.upper()} Mix Analysis Chat")
            
            # Initialize mix messages for this file type
            mix_messages_key = f"mix_messages_{file_type}"
            if mix_messages_key not in st.session_state:
                st.session_state[mix_messages_key] = []
            
            # Create a container for chat history with improved styling
            chat_container = st.container()
            with chat_container:
                # Display chat history
                for message in st.session_state[mix_messages_key]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Chat input with improved styling
            if prompt := st.chat_input(f"Ask questions about your selected {file_type.upper()} files", key=f"mix_{file_type}_chat_input"):
                # Add user message to chat history
                st.session_state[mix_messages_key].append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Get AI response from all selected files
                with st.chat_message("assistant"):
                    with st.status(f"Analyzing across selected {file_type.upper()} files...", expanded=True) as status:
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
                                        response += f"Error getting response: {pdf_response}\n\n"
                                
                                elif file_info['type'] == 'csv':
                                    docs = file_info['retriever'].get_relevant_documents(prompt)
                                    if docs:
                                        for doc in docs:
                                            response += f"- {doc.page_content}\n"
                                        response += "\n"
                                    else:
                                        response += "No relevant data found.\n\n"
                            
                            status.update(label="Analysis complete", state="complete")
                            st.markdown(response)
                            st.session_state[mix_messages_key].append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
    else:
        st.info(f"Please select at least one {file_type.upper()} file to start analysis.")

# Initialize session state
if 'pdf_files' not in st.session_state:
    st.session_state.pdf_files = {}
if 'csv_files' not in st.session_state:
    st.session_state.csv_files = {}

# Main title
st.title("📚 Multi-Document Chat Assistant")

# File upload section
st.markdown("## 📁 Upload Your Files")
col1, col2 = st.columns(2)

with col1:
    uploaded_pdf = st.file_uploader("Upload PDF files", type=['pdf'], key="pdf_uploader", help="Upload PDF files for analysis")

with col2:
    uploaded_csv = st.file_uploader("Upload CSV files", type=['csv'], key="csv_uploader", help="Upload CSV files for analysis")

# Process uploaded PDF
if uploaded_pdf is not None:
    # Generate unique key for this PDF
    pdf_key = f"pdf_{len(st.session_state.pdf_files) + 1}"
    
    # Check if this file is already processed
    file_exists = False
    for key, info in st.session_state.pdf_files.items():
        if info['name'] == uploaded_pdf.name and info['size'] == uploaded_pdf.size:
            file_exists = True
            break
    
    if not file_exists:
        # Improved tab naming logic
        existing_tabs = [info['tab_name'] for info in st.session_state.pdf_files.values()]
        if "PDF" not in existing_tabs:
            tab_name = "PDF"
        else:
            # Find the next available number
            numbers = [int(name.split("(")[1].split(")")[0]) for name in existing_tabs if "(" in name]
            next_num = max(numbers) + 1 if numbers else 2
            tab_name = f"PDF({next_num})"
        
        st.session_state.pdf_files[pdf_key] = {
            'file': uploaded_pdf,
            'type': 'pdf',
            'name': uploaded_pdf.name,
            'size': uploaded_pdf.size,
            'tab_name': tab_name,
            'processor': PDFChatModel(),
            'processed': False
        }

# Process uploaded CSV
if uploaded_csv is not None:
    # Generate unique key for this CSV
    csv_key = f"csv_{len(st.session_state.csv_files) + 1}"
    
    # Check if this file is already processed
    file_exists = False
    for key, info in st.session_state.csv_files.items():
        if info['name'] == uploaded_csv.name and info['size'] == uploaded_csv.size:
            file_exists = True
            break
    
    if not file_exists:
        # Improved tab naming logic
        existing_tabs = [info['tab_name'] for info in st.session_state.csv_files.values()]
        if "CSV" not in existing_tabs:
            tab_name = "CSV"
        else:
            # Find the next available number
            numbers = [int(name.split("(")[1].split(")")[0]) for name in existing_tabs if "(" in name]
            next_num = max(numbers) + 1 if numbers else 2
            tab_name = f"CSV({next_num})"
        
        st.session_state.csv_files[csv_key] = {
            'file': uploaded_csv,
            'type': 'csv',
            'name': uploaded_csv.name,
            'size': uploaded_csv.size,
            'tab_name': tab_name,
            'retriever': None,
            'processed': False
        }

# Create dynamic tabs
if st.session_state.pdf_files or st.session_state.csv_files:
    # Prepare tab names and content
    tab_names = []
    tab_content = []
    
    # Add PDF tabs
    for pdf_key, pdf_info in st.session_state.pdf_files.items():
        tab_names.append(f"📄 {pdf_info['tab_name']}")
        tab_content.append(('pdf_individual', pdf_key, pdf_info))
    
    # Add PDF Mix Analysis tab if more than one PDF
    if len(st.session_state.pdf_files) > 1:
        tab_names.append("🔍 PDF Mix Analysis")
        tab_content.append(('pdf_mix', None, None))
    
    # Add CSV tabs
    for csv_key, csv_info in st.session_state.csv_files.items():
        tab_names.append(f"📊 {csv_info['tab_name']}")
        tab_content.append(('csv_individual', csv_key, csv_info))
    
    # Add CSV Mix Analysis tab if more than one CSV
    if len(st.session_state.csv_files) > 1:
        tab_names.append("🔍 CSV Mix Analysis")
        tab_content.append(('csv_mix', None, None))
    
    # Create tabs
    tabs = st.tabs(tab_names)
    
    # Populate tabs
    for i, (tab_type, file_key, file_info) in enumerate(tab_content):
        with tabs[i]:
            if tab_type == 'pdf_individual':
                st.markdown(f"## 📄 {file_info['tab_name']} - {file_info['name']}")
                messages_key = f"messages_pdf_{file_key}"
                create_file_chat_interface(file_info, messages_key, file_key)
            
            elif tab_type == 'csv_individual':
                st.markdown(f"## 📊 {file_info['tab_name']} - {file_info['name']}")
                messages_key = f"messages_csv_{file_key}"
                create_file_chat_interface(file_info, messages_key, file_key)
            
            elif tab_type == 'pdf_mix':
                pdf_files_list = list(st.session_state.pdf_files.items())
                create_mix_analysis_interface('pdf', pdf_files_list)
            
            elif tab_type == 'csv_mix':
                csv_files_list = list(st.session_state.csv_files.items())
                create_mix_analysis_interface('csv', csv_files_list)

else:
    st.info("👆 Please upload your PDF or CSV files to get started!")

# Add footer
st.markdown("---")
st.markdown("💡 **Tips:**")
st.markdown("- Upload multiple files to create individual chat sessions for each file")
st.markdown("- Mix Analysis tabs appear when you have 2+ files of the same type")
st.markdown("- Each file maintains its own separate chat history")
st.markdown("- Select specific files in Mix Analysis to compare and analyze them together")