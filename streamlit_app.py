"""Streamlit UI for RAG Document Assistant"""

import streamlit as st
import requests
from pathlib import Path
import time
import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Page config
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def upload_document(file):
    """Upload document to the API"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error uploading document: {str(e)}")
        return None

def query_documents(question, k=4, temperature=None, max_tokens=None):
    """Query the RAG system"""
    try:
        payload = {"question": question, "k": k}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying documents: {str(e)}")
        return None

def get_document_count():
    """Get statistics about documents and chunks in the vector store"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents/count")
        response.raise_for_status()
        return response.json()
    except:
        return None

def list_documents():
    """List all uploaded documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents/list")
        response.raise_for_status()
        return response.json()
    except:
        return None

def delete_document(filename):
    """Delete a document from the vector store"""
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{filename}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return None

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">📚 RAG Document Assistant</h1>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not running! Please start the backend server first.")
        st.code("python run.py")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Document stats
        doc_info = get_document_count()
        if doc_info:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", doc_info.get("total_documents", 0))
            with col2:
                st.metric("Chunks", doc_info.get("total_chunks", 0))
            
            st.success("✅ Vector Store Ready" if doc_info.get("ready") else "⚠️ No Documents")
        
        st.divider()
        
        # Query settings
        st.subheader("Query Parameters")
        k = st.slider("Number of sources to retrieve", 1, 10, 4)
        
        use_custom_params = st.checkbox("Custom LLM parameters")
        temperature = None
        max_tokens = None
        
        if use_custom_params:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.number_input("Max Tokens", 100, 4096, 1024, 100)
        
        st.divider()
        
        # Info
        st.info("💡 Upload documents and ask questions about them!")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["💬 Query Documents", "📤 Upload Documents", "📋 Manage Documents"])
    
    # Query Tab
    with tab1:
        st.header("Ask Questions")
        
        question = st.text_area(
            "Enter your question:",
            placeholder="What is the main topic of the documents?",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            query_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if query_button:
            if not question.strip():
                st.warning("Please enter a question!")
            else:
                with st.spinner("🤔 Thinking..."):
                    result = query_documents(question, k, temperature, max_tokens)
                
                if result:
                    # Answer
                    st.subheader("📝 Answer")
                    st.markdown(f"**{result['answer']}**")
                    
                    st.divider()
                    
                    # Sources
                    if result.get('sources'):
                        st.subheader(f"📚 Sources ({len(result['sources'])} documents)")
                        
                        for idx, source in enumerate(result['sources'], 1):
                            with st.expander(f"Source {idx}: {source.get('source', 'Unknown')}"):
                                if 'page' in source:
                                    st.caption(f"Page: {source['page']}")
                                st.markdown(source.get('content', ''))
                    else:
                        st.info("No sources found.")
    
    # Upload Tab
    with tab2:
        st.header("Upload Documents")
        st.markdown("Supported formats: **PDF**, **TXT**, **Markdown**")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'md'],
            help="Upload documents to add them to the knowledge base"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("📤 Upload", type="primary"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = upload_document(uploaded_file)
                    
                    if result:
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>✅ {result['message']}</h4>
                            <p><strong>File:</strong> {result['filename']}</p>
                            <p><strong>Chunks created:</strong> {result['chunks_created']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
        
        st.divider()
        
        # Tips
        st.subheader("💡 Tips")
        st.markdown("""
        - Upload multiple documents to build a comprehensive knowledge base
        - Larger documents are automatically split into chunks
        - You can upload the same document multiple times (it will be added again)
        - Ask specific questions for better answers
        - Use the retrieved sources to verify information
        """)
    
    # Manage Documents Tab
    with tab3:
        st.header("Manage Documents")
        
        docs_list = list_documents()
        
        if docs_list and docs_list.get('documents'):
            st.subheader(f"📚 Uploaded Documents ({docs_list['total']})")
            
            for doc in docs_list['documents']:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.text(f"📄 {doc['filename']}")
                
                with col2:
                    st.text(f"{doc['chunks']} chunks")
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{doc['filename']}"):
                        with st.spinner(f"Deleting {doc['filename']}..."):
                            result = delete_document(doc['filename'])
                        
                        if result:
                            st.success(f"✅ Deleted: {doc['filename']}")
                            time.sleep(1)
                            st.rerun()
                
                st.divider()
        else:
            st.info("📭 No documents uploaded yet. Go to the Upload tab to add documents.")
        
        # Stats
        if docs_list:
            doc_stats = get_document_count()
            if doc_stats:
                st.subheader("📊 Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Documents", doc_stats.get("total_documents", 0))
                with col2:
                    st.metric("Total Chunks", doc_stats.get("total_chunks", 0))
                with col3:
                    avg_chunks = doc_stats.get("total_chunks", 0) / doc_stats.get("total_documents", 1) if doc_stats.get("total_documents", 0) > 0 else 0
                    st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")

if __name__ == "__main__":
    main()
