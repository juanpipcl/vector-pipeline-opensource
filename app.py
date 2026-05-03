import streamlit as st
import os
import warnings
import logging
from dotenv import load_dotenv

# Silenciar warnings de transformers para un log limpio
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv()

from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configuración de página
st.set_page_config(page_title="Data QA System", page_icon="📊", layout="wide")

@st.cache_resource
def init_resources():
    # Embeddings ligeros y eficientes
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        st.error("Falta PINECONE_INDEX_NAME en el archivo .env")
        st.stop()

    vectorstore = PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def main():
    st.title("📊 Asistente Inteligente de Datos")
    st.markdown("---")
    
    try:
        retriever = init_resources()
        # Usando la versión LTS estable que confirmamos hoy
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

        template = """Responde la consulta técnica basándote estrictamente en el contexto proporcionado:
        {context}

        Pregunta: {question}
        
        Respuesta técnica detallada:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        # Cadena LCEL moderna
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        user_query = st.text_input("Ingresa tu consulta sobre el pipeline o arquitectura:")

        if user_query:
            with st.spinner("Consultando base de conocimientos..."):
                response = rag_chain.invoke(user_query)
                st.markdown(response)

    except Exception as e:
        st.error(f"Error en el sistema: {e}")

if __name__ == "__main__":
    main()