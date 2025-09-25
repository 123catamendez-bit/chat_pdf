import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# App title and presentation
st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.write("Versión de Python:", platform.python_version())

# Permitir al usuario subir una imagen
uploaded_image = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, width=350, caption="Imagen cargada por el usuario")
else:
    st.info("Por favor carga una imagen para mostrarla aquí")

# Sidebar with theme selector
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Iconic_image_of_Earth_from_space.jpg/320px-Iconic_image_of_Earth_from_space.jpg", use_column_width=True)
    st.subheader("🌍 Bienvenido a tu asistente RAG")
    st.write("Aquí podrás analizar documentos PDF y resolver tus dudas con ayuda de la IA.")
    
    st.markdown("---")
    st.write("✨ Tip del día:")
    st.info("Recuerda que el conocimiento es mejor cuando se comparte.")
    
    # Theme selector
    theme = st.radio("Elige tu estilo:", ["🌞 Claro", "🌙 Oscuro", "🌈 Colorido"])

# Apply theme styles dynamically
if theme == "🌞 Claro":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

elif theme == "🌙 Oscuro":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stButton>button {
            background-color: #333333;
            color: #fafafa;
        }
        textarea, input {
            background-color: #262730 !important;
            color: #fafafa !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

elif theme == "🌈 Colorido":
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fad390, #fbc531);
            color: #000000;
        }
        .stButton>button {
            background-color: #ff6f61;
            color: white;
            border-radius: 10px;
        }
        textarea, input {
            background-color: #fff8e7 !important;
            color: #333333 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Get API key from user
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# PDF uploader
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Texto extraído: {len(text)} caracteres")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")
        
        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # User question interface
        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")
        
        # Process question when submitted
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Use a current model instead of deprecated text-davinci-003
            # Options: "gpt-3.5-turbo-instruct" or "gpt-4-turbo-preview" depending on your API access
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            
            # Load QA chain
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Run the chain
            response = chain.run(input_documents=docs, question=user_question)
            
            # Display the response
            st.markdown("### Respuesta:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        # Add detailed error for debugging
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
