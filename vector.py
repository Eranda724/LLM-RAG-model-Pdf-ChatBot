from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import tempfile
import time

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def process_csv_file(csv_file):
    """Process uploaded CSV file and create vector store"""
    tmp_file_path = None
    try:
        # Check file size
        file_size = len(csv_file.getvalue())
        if file_size > 10 * 1024 * 1024:  # 10MB limit for CSV
            return None, False, "File too large. Please upload a CSV smaller than 10MB."
        
        # Create a temporary file to store the CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(csv_file.getvalue())
            tmp_file_path = tmp_file.name

        # Read the CSV file with error handling
        try:
            df = pd.read_csv(tmp_file_path)
        except Exception as e:
            return None, False, f"Error reading CSV file: {str(e)}"
            
        if df.empty:
            return None, False, "Error: CSV file is empty"
        
        # Limit the number of rows to prevent memory issues
        if len(df) > 10000:  # Limit to 10k rows
            df = df.head(10000)
        
        # Create documents from CSV
        documents = []
        ids = []
        
        # Get column names for content and metadata
        columns = df.columns.tolist()
        
        # Use all columns for content, except the first column which will be used as metadata
        content_columns = columns[1:] if len(columns) > 1 else columns
        metadata_column = columns[0] if columns else None
        
        for i, row in df.iterrows():
            try:
                # Combine all content columns
                content = " ".join(str(row[col]) for col in content_columns)
                
                # Create metadata from the first column if it exists
                metadata = {metadata_column: str(row[metadata_column])} if metadata_column else {}
                
                document = Document(
                    page_content=content,
                    metadata=metadata,
                    id=str(i)
                )
                ids.append(str(i))
                documents.append(document)
            except Exception as row_error:
                # Skip problematic rows instead of failing completely
                continue

        if not documents:
            return None, False, "No valid data could be extracted from the CSV"

        # Create vector store with unique collection name
        vector_store = Chroma(
            collection_name=f"csv_data_{int(time.time())}",
            embedding_function=embeddings,
            persist_directory="./chroma_db" 
        )

        # Add documents to vector store in batches
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            try:
                vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                time.sleep(0.05)  # Small delay to prevent blocking
            except Exception as batch_error:
                # Continue with next batch instead of failing completely
                continue
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
        return retriever, True, "CSV processed successfully"
        
    except Exception as e:
        return None, False, f"Error processing CSV: {str(e)}"
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except:
                pass