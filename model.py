from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
import tempfile
import time
from PyPDF2 import PdfReader
import base64

class PDFChatModel:
    def __init__(self):
        self.vector_store = None
        self.chain = None
        self.processing_complete = False
        
        # Set environment variables
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def display_pdf_preview(self, pdf_file):
        """Display PDF preview"""
        try:
            # Read PDF
            pdf_reader = PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            
            # Get first page preview text
            if num_pages > 0:
                page = pdf_reader.pages[0]
                text = page.extract_text()
                
                # Convert PDF to base64 for display
                pdf_bytes = pdf_file.getvalue()
                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                
                return {
                    "preview_text": text,
                    "total_pages": num_pages,
                    "pdf_base64": base64_pdf,
                    "pdf_name": pdf_file.name,
                    "success": True
                }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

    def process_pdf(self, pdf_file):
        """Process the uploaded PDF file and create vector store"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name

            # Load PDF
            loader = UnstructuredPDFLoader(file_path=tmp_file_path)
            data = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(data)

            # Create vector database
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=OllamaEmbeddings(model="nomic-embed-text"),
                collection_name="local-rag"
            )

            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            self.processing_complete = True
            return True, "PDF processed successfully"
        except Exception as e:
            return False, f"Error processing PDF: {str(e)}"

    def setup_rag_chain(self):
        """Set up the RAG chain with the vector store"""
        try:
            # Set up LLM
            local_model = "llama3:8b"
            llm = ChatOllama(model=local_model)

            # Query prompt template
            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI language model assistant. Your task is to generate 2
                different versions of the given user question to retrieve relevant documents from
                a vector database. By generating multiple perspectives on the user question, your
                goal is to help the user overcome some of the limitations of the distance-based
                similarity search. Provide these alternative questions separated by newlines.
                Original question: {question}""",
            )

            # Set up retriever
            retriever = MultiQueryRetriever.from_llm(
                self.vector_store.as_retriever(),
                llm,
                prompt=QUERY_PROMPT
            )

            # RAG prompt template
            template = """Answer the question based ONLY on the following context:
            {context}
            Question: {question}
            """

            prompt = ChatPromptTemplate.from_template(template)

            # Create chain
            self.chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            return True, "RAG chain setup successfully"
        except Exception as e:
            return False, f"Error setting up RAG chain: {str(e)}"

    def get_response(self, question):
        """Get response from the model"""
        try:
            start_time = time.time()
            response = self.chain.invoke(question)
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            return True, response, processing_time
        except Exception as e:
            return False, f"Error generating response: {str(e)}", 0

    def is_ready(self):
        """Check if the model is ready for chat"""
        return self.vector_store is not None and self.chain is not None 