# PDF Chat Assistant

A Streamlit application that allows you to chat with your PDF documents using Ollama and LangChain.

## Prerequisites

1. Python 3.8 or higher
2. Ollama installed and running locally
3. Required Ollama models:
   - llama3:8b
   - nomic-embed-text

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure Ollama is running and you have the required models:
   ```bash
   ollama pull llama3:8b
   ollama pull nomic-embed-text
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)
3. Upload a PDF file using the file uploader
4. Once the PDF is processed, you can start asking questions about its content
5. The chat interface will maintain a history of your conversation

## Features

- PDF document processing and chunking
- Vector-based semantic search
- Multi-query retrieval for better context
- Chat interface with message history
- Real-time responses using Ollama LLM

## Note

Make sure you have enough system resources to run the LLM models locally. The application uses Ollama for both embeddings and text generation.
