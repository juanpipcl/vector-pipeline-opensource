import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def ask_my_docs(question):
    INDEX_NAME = "vector-pipeline-index"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Conectar al índice existente
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    # Realizar la búsqueda
    print(f"\n🔍 Pregunta: {question}")
    docs = vectorstore.similarity_search(question, k=3)
    
    print("\n📍 Resultados encontrados en tus documentos:")
    for i, doc in enumerate(docs):
        print(f"\n--- Resultado {i+1} ---")
        print(doc.page_content[:300] + "...") # Mostramos solo el inicio del texto

if __name__ == "__main__":
    query = input("¿Qué quieres saber de tus documentos?: ")
    ask_my_docs(query)