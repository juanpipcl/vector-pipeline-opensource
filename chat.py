import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

load_dotenv()

def start_chatbot():
    # 1. Configuración del Cerebro (LLM)
    # Consigue tu API Key gratis en https://console.groq.com/
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.1-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # 2. Configuración de los Vectores
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index_name="vector-pipeline-index", embedding=embeddings)

    # 3. Crear la cadena de Chat (QA Chain)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

    print("🤖 Chatbot listo. ¡Pregúntame sobre tus documentos! (Escribe 'salir' para terminar)")
    
    while True:
        user_input = input("\n👤 Tú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            break
            
        print("🤔 Pensando...")
        response = qa_chain.invoke(user_input)
        print(f"🤖 Chatbot: {response['result']}")

if __name__ == "__main__":
    start_chatbot()