# 🧠 AI-Powered Multi-Modal Document & Speech Assistant

![Home Screenshot](https://github.com/user-attachments/assets/a4c1a1d1-3c47-46a0-a512-f9a1054c9afb)

A powerful **Streamlit-based** application that enables seamless interaction with **multiple document formats** (PDFs, CSVs, TXT) through local **LLMs** (via [Ollama](https://ollama.com)) and **LangChain**. This assistant combines **natural language processing**, **speech recognition**, and **semantic search** into one unified platform.

---

## 🚀 Features

- 🗂️ **Multi-Document Upload & Processing** – Analyze PDFs, CSVs, and plain text files together  
- 🔍 **Vector-Based Semantic Search** – Fast and contextual retrieval using ChromaDB embeddings  
- 🧠 **Custom GPT Chatbot** – Chat with documents using local LLMs powered by Ollama  
- 📄 **Cross-Document Summarization** – Summarize and compare content from multiple files  
- 🗣️ **Real-Time Speech-to-Text & Translation** – Transcribe and translate multilingual input (e.g., Sinhala/Singlish)  
- 🗨️ **Context-Aware Conversations** – Maintains full dialogue history across inputs  
- 🔁 **Multi-Query Retrieval** – Improves search accuracy using diverse query strategies  
- 🔊 **Text-to-Speech Output** – Convert AI responses into speech in 20+ languages  
- 📷 **Coming Soon** – OCR support for scanned image and document processing  

---

## 📸 Screenshots

![Screenshot 1](https://github.com/user-attachments/assets/cfde0872-dd7b-4f4c-9f72-4ea9361fb835)
![Screenshot 2](https://github.com/user-attachments/assets/065f3882-3d28-409e-89d5-35612b9569c0)
![Screenshot 3](https://github.com/user-attachments/assets/d7b4e69e-613a-4356-9500-3bd1d941143e)
![Screenshot 4](https://github.com/user-attachments/assets/703a1a0a-dda5-446b-879d-3d92bbe59d2e)
![Screenshot 5](https://github.com/user-attachments/assets/808e4dea-2d84-4c22-99b8-6fd886028949)
![Screenshot 6](https://github.com/user-attachments/assets/8f35308b-fac6-44d9-b5ca-764234a000a7)
![Screenshot 7](https://github.com/user-attachments/assets/eb852d80-3be3-462a-9235-a7817b426f22)
![Screenshot 8](https://github.com/user-attachments/assets/f4ba42e8-79af-4a97-b024-b7acca6b8369)
![Screenshot 9](https://github.com/user-attachments/assets/5214abd9-5286-45ab-9106-ddb6cbfffdae)
![Screenshot 10](https://github.com/user-attachments/assets/7d932204-9868-4cf6-ac25-2ef1db1edb9e)
![Screenshot 11](https://github.com/user-attachments/assets/47123e1c-912e-49f5-b633-01dd5dd60293)

---

## 🧩 Prerequisites

- Python **3.8+**
- [Ollama](https://ollama.com) installed and running locally
- Required Ollama models:
  - `llama3:8b` – for language generation
  - `nomic-embed-text` – for semantic vector embeddings

---

## 🔧 Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Start Ollama and pull required models:**

    ```bash
    ollama pull llama3:8b
    ollama pull nomic-embed-text
    ```

---

## ▶️ Usage

Start the application with:

```bash
streamlit run app.py
