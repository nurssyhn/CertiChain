import os
from dotenv import load_dotenv

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# txt_file_path = os.path.join(current_dir, "documents", "yildizlar.txt")
csv_file_path = os.path.join(current_dir, "documents", "username_email.csv")
pdf_file_path = os.path.join(current_dir, "documents", "RHELinux.pdf")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    # if not os.path.exists(txt_file_path):
    #     raise FileNotFoundError(
    #         f"The file {txt_file_path} does not exist. Please check the path."
    #     )

    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(
            f"The file {csv_file_path} does not exist. Please check the path."
        )

    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(
            f"The file {pdf_file_path} does not exist. Please check the path."
        )


    # Read the text content from the file
    # txt_loader = TextLoader(txt_file_path)
    # txt_documents = txt_loader.load()
    
    # Load CSV data
    csv_loader = CSVLoader(file_path=csv_file_path)
    csv_documents = csv_loader.load()

    # Load PDF data
    pdf_loader = PyPDFLoader(file_path=pdf_file_path)
    pdf_documents = pdf_loader.load()


    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # txt_chunks = text_splitter.split_documents(txt_documents)
    csv_chunks = text_splitter.split_documents(csv_documents)
    pdf_chunks = text_splitter.split_documents(pdf_documents)

    # Combine all chunks
    # all_chunks = pdf_chunks + txt_chunks + csv_chunks
    all_chunks = pdf_chunks + csv_chunks


    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(all_chunks)}")
    print(f"Sample chunk:\n{all_chunks[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Update to a valid embedding model if needed
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        all_chunks, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")

