# GPT Chat Assistant Integration

This project now includes a GPT Chat Assistant that allows users to ask general questions and have conversations, similar to ChatGPT.

## Features

- **🤖 GPT Chat Assistant**: General-purpose chatbot for any questions
- **📄 PDF Analysis**: Upload and chat with PDF documents
- **📊 CSV Analysis**: Upload and analyze CSV data
- **📝 Text Analysis**: Direct text input and analysis
- **🔍 Mix Analysis**: Compare multiple files of the same type

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the GPT Model

The GPT Chat Assistant uses the `orca-mini-3b` model. You need to download it:

```bash
# Create models directory
mkdir -p models

# Download the model (this will be done automatically by ctransformers)
# The model will be downloaded to ~/.cache/huggingface/hub/
```

### 3. Run the Application

```bash
streamlit run app.py
```

## How to Use

### GPT Chat Assistant

1. Open the application in your browser
2. Click on the **"🤖 GPT Chat Assistant"** tab
3. Start asking questions in the chat input
4. The assistant will respond with helpful answers
5. Use the "🗑️ Clear Chat" button to reset the conversation

### File Analysis

1. Upload PDF or CSV files using the file uploader
2. Add text directly using the text input section
3. Each file will get its own tab for individual analysis
4. Mix Analysis tabs appear when you have multiple files of the same type

## Model Information

- **Model**: orca-mini-3b (3B parameter model)
- **Type**: GGUF format for efficient local inference
- **Features**:
  - Conversation memory (remembers previous exchanges)
  - Fast local inference
  - No internet connection required
  - Privacy-focused (all processing happens locally)

## Troubleshooting

### Model Download Issues

If the model doesn't download automatically:

1. Check your internet connection
2. Ensure you have enough disk space (~2GB for the model)
3. Try running the application again

### Performance Issues

- The model runs locally on your CPU
- First response may take longer as the model loads
- Subsequent responses will be faster
- Consider closing other applications to free up memory

### Error Messages

- **"ChatBot not initialized"**: Check if the model file is properly downloaded
- **"Error generating response"**: Try restarting the application

## Customization

You can modify the GPT Chat Assistant by editing `gpt_chatbot.py`:

- Change the model by modifying the `model` parameter
- Adjust response parameters like `temperature`, `max_new_tokens`
- Modify the prompt template for different behavior
- Change the conversation memory length

## Requirements

- Python 3.8+
- 4GB+ RAM recommended
- 2GB+ free disk space for the model
- Internet connection for initial model download
