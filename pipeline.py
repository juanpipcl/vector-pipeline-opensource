import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 1. Cargar variables de entorno
load_dotenv()

def run_pipeline():
    # Configuración de nombres
    INDEX_NAME = "vector-pipeline-index"
    DATA_PATH = "./data/"
    
    # 2. Inicializar Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    # 3. Crear el índice si no existe (Capa gratuita: Serverless)
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"🏗️ Creando índice: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384, # Dimensión fija para el modelo all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # 4. Procesar archivos en la carpeta data
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            print(f"📄 Procesando: {file}...")
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            pages = loader.load()

            # Dividir texto en trozos (Chunks)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(pages)

            # 5. Subir a la base de datos vectorial
            PineconeVectorStore.from_documents(
                docs, 
                embeddings, 
                index_name=INDEX_NAME
            )
            print(f"✅ {file} vectorizado y guardado en la nube.")

if __name__ == "__main__":
    run_pipeline()