import streamlit as st
import pandas as pd
import os
from datetime import datetime
import uuid
import re
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Initialize session states
def init_session_state():
    if 'email_verified' not in st.session_state:
        st.session_state.email_verified = False
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'home'
    if 'rag_instance' not in st.session_state:
        st.session_state.rag_instance = None

# MongoDB setup
def get_database():
    try:
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            st.error("Missing MongoDB URI in environment variables")
            return None
        client = MongoClient(mongo_uri)
        client.server_info()
        return client.rag_app_db
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        return None

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def log_email(email):
    try:
        db = get_database()
        if db is not None:
            user_collection = db.users
            timestamp = datetime.now()
            result = user_collection.insert_one({
                "email": email,
                "timestamp": timestamp
            })
            return bool(result.inserted_id)
        return False
    except Exception as e:
        st.error(f"Failed to log email: {str(e)}")
        return False

class UnifiedRAG:
   def __init__(self):
    # Initialize HuggingFace model and embeddings
    self.llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )
    self.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    self.vector_store = None

    def process_pdf(self, pdf_file):
        if not pdf_file:
            return 0
        try:
            temp_pdf_path = f"temp_{uuid.uuid4()}.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getvalue())
            
            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            
            self.vector_store = FAISS.from_documents(
                documents=texts,
                embedding=self.embeddings
            )
            
            os.remove(temp_pdf_path)
            return len(texts)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return 0

    def process_tabular_data(self, df):
        try:
            return create_pandas_dataframe_agent(
                llm=self.llm,
                df=df,
                verbose=True
            )
        except Exception as e:
            st.error(f"Error processing tabular data: {str(e)}")
            return None

    def get_answer(self, query):
        try:
            if not self.vector_store:
                return "Please upload a document first."
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )
            )
            
            response = qa_chain.invoke({"query": query})
            return response["result"]
        except Exception as e:
            return f"Error generating answer: {str(e)}"

def apply_custom_css():
    st.markdown("""
    <style>
        /* Modern UI Theme */
        :root {
            --primary-color: #7C3AED;
            --secondary-color: #4F46E5;
            --background-color: #F9FAFB;
            --card-background: #FFFFFF;
            --text-color: #1F2937;
        }

        .stApp {
            background-color: var(--background-color);
        }

        /* Header Styling */
        .main-header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }

        /* Card Styling */
        .app-card {
            background: var(--card-background);
            border-radius: 1rem;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .app-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.1);
        }

        /* Button Styling */
        .stButton button {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-weight: 500;
            transition: opacity 0.2s;
        }

        .stButton button:hover {
            opacity: 0.9;
        }

        /* Input Field Styling */
        .stTextInput input {
            border-radius: 0.5rem;
            border: 1px solid #E5E7EB;
            padding: 0.75rem;
        }

        /* File Uploader Styling */
        .stUploadedFile {
            background: var(--card-background);
            border-radius: 0.5rem;
            padding: 1rem;
            border: 2px dashed #E5E7EB;
        }
    </style>
    """, unsafe_allow_html=True)

def render_home():
    st.markdown("""
        <div class="main-header">
            <h1>AI Document Analysis Hub</h1>
            <p>Powered by Open Source AI Models</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="app-card">
                <h3>PDF Analysis</h3>
                <p>Extract insights from your PDF documents using advanced AI</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch PDF Assistant"):
            st.session_state.app_mode = 'pdf'
            st.rerun()

    with col2:
        st.markdown("""
            <div class="app-card">
                <h3>Spreadsheet Analysis</h3>
                <p>Analyze Excel and CSV files with natural language queries</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Spreadsheet Assistant"):
            st.session_state.app_mode = 'spreadsheet'
            st.rerun()

def render_pdf_analysis():
    st.markdown("""
        <div class="main-header">
            <h2>PDF Document Analysis</h2>
        </div>
    """, unsafe_allow_html=True)

    if 'rag_instance' not in st.session_state:
        st.session_state.rag_instance = UnifiedRAG()

    uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])
    
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Processing document..."):
                num_chunks = st.session_state.rag_instance.process_pdf(uploaded_file)
                if num_chunks > 0:
                    st.success(f"Document processed successfully! Created {num_chunks} text chunks.")

    query = st.text_input("Ask a question about your document:")
    if query:
        with st.spinner("Analyzing..."):
            answer = st.session_state.rag_instance.get_answer(query)
            st.markdown(f"""
                <div class="app-card">
                    <h4>Answer:</h4>
                    <p>{answer}</p>
                </div>
            """, unsafe_allow_html=True)

def render_spreadsheet_analysis():
    st.markdown("""
        <div class="main-header">
            <h2>Spreadsheet Analysis</h2>
        </div>
    """, unsafe_allow_html=True)

    if 'rag_instance' not in st.session_state:
        st.session_state.rag_instance = UnifiedRAG()

    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.markdown("""
                <div class="app-card">
                    <h4>Data Preview</h4>
                </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.head())
            
            agent = st.session_state.rag_instance.process_tabular_data(df)
            
            query = st.text_input("Ask a question about your data:")
            if query and agent:
                with st.spinner("Analyzing..."):
                    response = agent.run(query)
                    st.markdown(f"""
                        <div class="app-card">
                            <h4>Analysis Result:</h4>
                            <p>{response}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def main():
    st.set_page_config(
        page_title="AI Document Analysis Hub",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    apply_custom_css()
    init_session_state()

    if not st.session_state.email_verified:
        st.markdown("""
            <div class="main-header">
                <h2>Welcome to AI Document Analysis Hub</h2>
                <p>Please verify your email to continue</p>
            </div>
        """, unsafe_allow_html=True)
        
        email = st.text_input("Email Address")
        if st.button("Verify Email"):
            if validate_email(email):
                if log_email(email):
                    st.session_state.email_verified = True
                    st.rerun()
                else:
                    st.error("Failed to verify email. Please try again.")
            else:
                st.error("Please enter a valid email address")
        return

    # Navigation
    if st.sidebar.button("🏠 Home"):
        st.session_state.app_mode = 'home'
        st.rerun()

    # Render appropriate view
    if st.session_state.app_mode == 'pdf':
        render_pdf_analysis()
    elif st.session_state.app_mode == 'spreadsheet':
        render_spreadsheet_analysis()
    else:
        render_home()

if __name__ == "__main__":
    main()