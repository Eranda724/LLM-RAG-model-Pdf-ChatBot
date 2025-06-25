from langchain_community.document_loaders import PyPDFLoader
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
import logging

class PDFChatModel:
    def __init__(self):
        self.vector_store = None
        self.chain = None
        self.processing_complete = False
        self.pdf_metadata = {}
        
        # Set environment variables
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def display_pdf_preview(self, pdf_file):
        """Display PDF preview with enhanced error handling"""
        try:
            # Reset file pointer
            pdf_file.seek(0)
            
            # Read PDF
            pdf_reader = PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            
            # Get metadata
            metadata = pdf_reader.metadata
            
            # Get first page preview text
            preview_text = ""
            if num_pages > 0:
                page = pdf_reader.pages[0]
                preview_text = page.extract_text()[:500]  # First 500 characters
                
            # Convert PDF to base64 for display
            pdf_file.seek(0)
            pdf_bytes = pdf_file.getvalue()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            
            # Store metadata
            self.pdf_metadata = {
                "title": metadata.get('/Title', 'Unknown') if metadata else 'Unknown',
                "author": metadata.get('/Author', 'Unknown') if metadata else 'Unknown',
                "pages": num_pages,
                "filename": pdf_file.name
            }
            
            return {
                "preview_text": preview_text,
                "total_pages": num_pages,
                "pdf_base64": base64_pdf,
                "pdf_name": pdf_file.name,
                "metadata": self.pdf_metadata,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error in PDF preview: {str(e)}")
            return {
                "error": f"Failed to load PDF preview: {str(e)}",
                "success": False
            }

    def process_pdf(self, pdf_file):
        """Process the uploaded PDF file and create vector store with progress tracking"""
        try:
            # Reset file pointer
            pdf_file.seek(0)
            
            self.logger.info("Starting PDF processing...")
            
            # Check file size to prevent memory issues
            file_size = len(pdf_file.getvalue())
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                return False, "File too large. Please upload a PDF smaller than 50MB."
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name

            self.logger.info(f"Temporary file created: {tmp_file_path}")

            # Load PDF with enhanced error handling
            try:
                loader = PyPDFLoader(file_path=tmp_file_path)
                data = loader.load()
                self.logger.info(f"PDF loaded successfully. Found {len(data)} document chunks.")
            except Exception as e:
                self.logger.error(f"Error loading PDF: {str(e)}")
                # Clean up temp file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                raise Exception(f"Failed to load PDF content: {str(e)}")

            if not data:
                # Clean up temp file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                raise Exception("No content could be extracted from the PDF")

            # Split text into chunks with optimized parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(data)
            self.logger.info(f"Text split into {len(chunks)} chunks.")

            if not chunks:
                # Clean up temp file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                raise Exception("No text chunks could be created from the PDF")

            # Create vector database with error handling
            try:
                # Initialize empty vector store first
                self.vector_store = Chroma(
                    collection_name=f"local-rag-{int(time.time())}",
                    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
                    persist_directory=tempfile.mkdtemp()
                )

                # Add documents in smaller batches to prevent memory issues
                batch_size = 5  # Reduced batch size
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    try:
                        self.vector_store.add_documents(batch)
                        # Add longer sleep to prevent frontend disconnection
                        time.sleep(0.1)
                    except Exception as batch_error:
                        self.logger.error(f"Error adding batch {i//batch_size + 1}: {str(batch_error)}")
                        # Continue with next batch instead of failing completely
                        continue

                self.logger.info("Vector store created successfully.")
            except Exception as e:
                self.logger.error(f"Error creating vector store: {str(e)}")
                # Clean up temp file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                raise Exception(f"Failed to create vector database: {str(e)}")

            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
                self.logger.info("Temporary file cleaned up.")
            except Exception as e:
                self.logger.warning(f"Could not delete temporary file: {str(e)}")
            
            self.processing_complete = True
            return True, "PDF processed successfully"
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            # Ensure cleanup on any error
            try:
                if 'tmp_file_path' in locals():
                    os.unlink(tmp_file_path)
            except:
                pass
            return False, f"Error processing PDF: {str(e)}"

    def setup_rag_chain(self):
        """Set up the RAG chain with the vector store and enhanced prompts"""
        try:
            self.logger.info("Setting up RAG chain...")
            
            # Set up LLM with error handling
            local_model = "llama3.2"  # Example model name, adjust as needed
            try:
                llm = ChatOllama(model=local_model, temperature=0.1)
                self.logger.info(f"LLM initialized with model: {local_model}")
            except Exception as e:
                self.logger.error(f"Error initializing LLM: {str(e)}")
                raise Exception(f"Failed to initialize language model. Make sure Ollama is running and {local_model} is installed.")

            # Enhanced query prompt template
            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI language model assistant. Your task is to generate 2-3
                different versions of the given user question to retrieve relevant documents from
                a vector database. By generating multiple perspectives on the user question, your
                goal is to help the user overcome some of the limitations of the distance-based
                similarity search. Provide these alternative questions separated by newlines.
                
                Original question: {question}
                
                Generate variations that:
                1. Use different keywords or synonyms
                2. Approach the topic from different angles
                3. Consider broader or more specific contexts
                """,
            )

            # Set up retriever with error handling
            try:
                retriever = MultiQueryRetriever.from_llm(
                    self.vector_store.as_retriever(search_kwargs={"k": 5}),
                    llm,
                    prompt=QUERY_PROMPT
                )
                self.logger.info("Retriever setup completed.")
            except Exception as e:
                self.logger.error(f"Error setting up retriever: {str(e)}")
                raise Exception(f"Failed to setup document retriever: {str(e)}")

            # Enhanced RAG prompt template
            template = """You are a helpful AI assistant that answers questions based on the provided context from a PDF document.

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

            # Create enhanced chain
            self.chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            self.logger.info("RAG chain setup completed successfully.")
            return True, "RAG chain setup successfully"
            
        except Exception as e:
            self.logger.error(f"Error setting up RAG chain: {str(e)}")
            return False, str(e)

    def get_response(self, question):
        """Get response from the model with enhanced error handling and context"""
        try:
            if not self.chain:
                raise Exception("RAG chain not initialized. Please process a PDF first.")
            
            self.logger.info(f"Processing question: {question[:50]}...")
            start_time = time.time()
            
            # Get response
            response = self.chain.invoke(question)
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 2)
            
            # Enhance response with metadata if available
            enhanced_response = response
            if self.pdf_metadata:
                enhanced_response += f"\n\n*Source: {self.pdf_metadata.get('filename', 'Unknown PDF')}*"
            
            self.logger.info(f"Response generated successfully in {processing_time} seconds.")
            return True, enhanced_response, processing_time
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return False, f"Error generating response: {str(e)}", 0

    def is_ready(self):
        """Check if the model is ready for chat"""
        return (self.vector_store is not None and 
                self.chain is not None and 
                self.processing_complete)
    
    def get_pdf_info(self):
        """Get information about the loaded PDF"""
        return self.pdf_metadata if self.pdf_metadata else None
    
    def reset(self):
        """Reset the model state"""
        self.vector_store = None
        self.chain = None
        self.processing_complete = False
        self.pdf_metadata = {}
        self.logger.info("Model state reset.")