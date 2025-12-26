import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="Chatbot Básico", page_icon="💬")
st.title("💬 Chatbot Básico con LangChain")
st.markdown(
    "Este es un chatbot de ejemplo construido con LangChain + Streamlit. "
    "¡Escribe tu pregunta abajo para comenzar!"
)

# ---------- PROMPT TEMPLATE ----------
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "Eres un asistente útil y amigable llamado ChatBot Pro. "
        "Responde de manera clara y concisa."
    ),
    (
        "human",
        "{mensaje}"
    )
])

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Configuración")

    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=0.9,
        step=0.1,
        value=0.5
    )

    st.divider()

    if st.button("🆕 Nueva conversación"):
        st.session_state.mensajes = []
        st.rerun()


# ---------- MODELO ----------
chat_model = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=temperature,
    streaming=True
)

# ---------- HISTORIAL ----------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# ---------- MOSTRAR HISTORIAL ----------
for msg in st.session_state.mensajes:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# ---------- INPUT ----------
pregunta = st.chat_input("Escribe tu pregunta:")

if pregunta:
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Crear mensajes del prompt SOLO para este input
    mensajes_prompt = prompt_template.format_messages(
        mensaje=pregunta
    )

    # Invocar modelo con historial + prompt
    respuesta = chat_model.invoke(
        st.session_state.mensajes + mensajes_prompt
    )

    # Mostrar respuesta
    with st.chat_message("assistant"):
        placeholder = st.empty()
        respuesta_completa = ""

        for chunk in chat_model.stream(
                st.session_state.mensajes + mensajes_prompt
        ):
            respuesta_completa += chunk.content
            placeholder.markdown(respuesta_completa)

    # Guardar historial
    st.session_state.mensajes.append(
        HumanMessage(content=pregunta)
    )
    st.session_state.mensajes.append(
        AIMessage(content=respuesta_completa)
    )
