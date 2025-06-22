from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import tempfile

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def process_csv_file(csv_file):
    """Process uploaded CSV file and create vector store"""
    df = pd.read_csv(tmp_file_path)
    if df.empty:  # Add validation
        return None, False, "Error: CSV file is empty"
    try:
        # Create a temporary file to store the CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(csv_file.getvalue())
            tmp_file_path = tmp_file.name

        # Read the CSV file
        df = pd.read_csv(tmp_file_path)
        
        # Create documents from CSV
        documents = []
        ids = []
        
        # Get column names for content and metadata
        columns = df.columns.tolist()
        
        # Use all columns for content, except the first column which will be used as metadata
        content_columns = columns[1:] if len(columns) > 1 else columns
        metadata_column = columns[0] if columns else None
        
        for i, row in df.iterrows():
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

        # Create vector store
        vector_store = Chroma(
            collection_name=f"csv_data_{int(time.time())}",
            embedding_function=embeddings,
            persist_directory="./chroma_db" 
)


        # Add documents to vector store
        vector_store.add_documents(documents=documents, ids=ids)
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return retriever, True, "CSV processed successfully"
        
    except Exception as e:
        return None, False, f"Error processing CSV: {str(e)}"