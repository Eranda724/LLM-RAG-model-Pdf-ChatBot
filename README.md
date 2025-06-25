# 🧠 AI-Powered Multi-Modal Document & Speech Assistant
![home](https://github.com/user-attachments/assets/a4c1a1d1-3c47-46a0-a512-f9a1054c9afb)

A powerful Streamlit-based application that lets you **analyze, chat with, and summarize multiple documents** — including PDFs, CSVs, and plain text — using **local LLMs (via Ollama)** and **LangChain**. The assistant supports **real-time speech processing**, **custom GPT interactions**, and **semantic multi-document analysis** in one unified platform.

---

## 🚀 Features

* 🗂️ **Multi-Document Upload & Processing** – Analyze PDFs, CSVs, and text files simultaneously
* 🔍 **Vector-Based Semantic Search** – ChromaDB-powered embeddings for context-rich retrieval
* 🧠 **Custom GPT Chatbot** – Unified interface for chatting with documents, speech input, or custom text
* 📄 **Cross-Document Summarization** – Generate combined summaries and comparisons
* 🗣️ **Speech-to-Text & Translation** – Real-time transcription and multilingual translation (e.g., Sinhala/Singlish)
* 🗨️ **Context-Aware Chat Interface** – Maintains full conversation history for fluid interaction
* 🧾 **Multi-Query Retrieval** – Enhances result quality with diverse query strategies
* 🔊 **Text-to-Speech** – Generate speech output from AI responses in 20+ languages
* 📷 **Coming Soon** – OCR integration to analyze and extract content from scanned images

![Screenshot 2025-06-25 095341](https://github.com/user-attachments/assets/cfde0872-dd7b-4f4c-9f72-4ea9361fb835)
![Screenshot 2025-06-25 094959](https://github.com/user-attachments/assets/065f3882-3d28-409e-89d5-35612b9569c0)
![Screenshot 2025-06-25 102150](https://github.com/user-attachments/assets/d7b4e69e-613a-4356-9500-3bd1d941143e)
![Screenshot 2025-06-25 102209](https://github.com/user-attachments/assets/703a1a0a-dda5-446b-879d-3d92bbe59d2e)

![Screenshot (6)](https://github.com/user-attachments/assets/808e4dea-2d84-4c22-99b8-6fd886028949)
![Screenshot (5)](https://github.com/user-attachments/assets/8f35308b-fac6-44d9-b5ca-764234a000a7)
![Screenshot (9)](https://github.com/user-attachments/assets/eb852d80-3be3-462a-9235-a7817b426f22)

![Screenshot 2025-06-25 102251](https://github.com/user-attachments/assets/f4ba42e8-79af-4a97-b024-b7acca6b8369)
![Screenshot 2025-06-25 125921](https://github.com/user-attachments/assets/5214abd9-5286-45ab-9106-ddb6cbfffdae)
![Screenshot 2025-06-25 130152](https://github.com/user-attachments/assets/7d932204-9868-4cf6-ac25-2ef1db1edb9e)
![ttos re](https://github.com/user-attachments/assets/47123e1c-912e-49f5-b633-01dd5dd60293)

---

## 🧩 Prerequisites

* Python **3.8+**
* [**Ollama**](https://ollama.com/) installed and running locally
* Required Ollama models:

  * `llama3:8b` (for text generation)
  * `nomic-embed-text` (for semantic embeddings)

---

## 🔧 Installation

Clone this repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure Ollama is running and pull the required models:

```bash
ollama pull llama3:8b
ollama pull nomic-embed-text
```

---

## ▶️ Usage

Start the Streamlit application:

```bash
streamlit run app.py
```

Then:

1. Open the URL shown in your terminal (usually [http://localhost:8501](http://localhost:8501))
2. Upload one or more files (PDF, CSV, or plain text)
3. Use the chat interface to ask questions or request summaries
4. For speech input, use the microphone to transcribe and translate in real time
5. Responses can also be converted to speech in your preferred language

---

## ⚠️ Notes

* Ensure you have sufficient system memory and CPU/GPU resources to run LLMs locally
* The application uses Ollama for **both embedding generation** and **language modeling**, allowing full offline/private use
* OCR functionality for image files will be integrated in a future update

---

## 📌 Use Cases

* Academic research and multi-source summarization
* Meeting transcription and documentation
* Voice-assisted document navigation
* Cross-lingual document understanding
* Accessible document reading via text-to-speech

---

Feel free to fork, contribute, or share feedback via Issues or Pull Requests!
