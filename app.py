import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Data Bot Pro", page_icon="💬", layout="wide")

@st.cache_resource
def init_resources():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def main():
    st.title("💬 Consultoría de Datos con Memoria")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    try:
        retriever = init_resources()
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

        # Prompt optimizado
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres un asistente experto. Responde basándote solo en este contexto:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        # Mostrar mensajes previos
        for message in st.session_state.chat_history:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)

        # Entrada de usuario
        if user_query := st.chat_input("Escribe tu consulta sobre los documentos..."):
            with st.chat_message("user"):
                st.markdown(user_query)

            # --- LÓGICA DE RECUPERACIÓN MANUAL PARA EVITAR ERRORES DE DICT ---
            with st.chat_message("assistant"):
                with st.spinner("Buscando en documentos..."):
                    # 1. Recuperar documentos relevantes
                    docs = retriever.invoke(user_query)
                    context_text = "\n\n".join(doc.page_content for doc in docs)
                    
                    # 2. Definir la cadena de ejecución directa
                    # Al pasarle las variables ya procesadas como strings/listas, evitamos el error .replace()
                    chain = prompt | llm | StrOutputParser()
                    
                    # 3. Invocar con el formato exacto
                    response = chain.invoke({
                        "context": context_text,
                        "chat_history": st.session_state.chat_history,
                        "question": user_query
                    })
                    
                    st.markdown(response)

            # Actualizar historial
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

    except Exception as e:
        st.error(f"Error de sistema: {e}")
        # Log para depuración interna
        print(f"DEBUG ERROR: {type(e).__name__} - {e}")

if __name__ == "__main__":
    main()