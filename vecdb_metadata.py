import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
documents_dir = os.path.join(current_dir, "documents")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Documents directory: {documents_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the documents directory exists
    if not os.path.exists(documents_dir):
        raise FileNotFoundError(
            f"The directory {documents_dir} does not exist. Please check the path."
        )

    # List all text, PDF and CSV files in the directory
    document_files = [f for f in os.listdir(documents_dir) 
        if f.endswith(".txt") or 
        f.endswith(".csv") or 
        f.endswith(".pdf")]

    # Read the content from each file and store it with metadata
    documents = []
    for document_file in document_files:
        file_path = os.path.join(documents_dir, document_file)
        if document_file.endswith(".txt"):
            loader = TextLoader(file_path)
        elif document_file.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif document_file.endswith(".pdf"):
            loader=PyPDFLoader(file_path)
        else:
            continue
        
        document_docs = loader.load()
        for doc in document_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": document_file}
            documents.append(doc)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")

