import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import time
from pathlib import Path
from typing import Optional, Tuple
from streamlit_pdf_viewer import pdf_viewer
import streamlit.components.v1 as components


# --- CONFIGURATION ---
DB_PATH = "./chroma_db"
COLLECTION_NAME = "inalco_unified"
MODEL_NAME = "intfloat/multilingual-e5-base"
RAW_DOCS_PATH = r"C:\Users\Philo\inalcofiles"


@st.cache_resource
def load_model():
    """Load the SentenceTransformer model (cached)."""
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_collection():
    """Connect to ChromaDB and load the collection (cached)."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


@st.cache_data
def get_files_in_directory():
    """Cache the list of files in the raw documents directory (case-insensitive lookup)."""
    if os.path.exists(RAW_DOCS_PATH):
        return {f.lower(): f for f in os.listdir(RAW_DOCS_PATH)}
    return {}


def find_k_chunks(query: str, model: SentenceTransformer, collection, k: int) -> dict:
    """
    Find k most relevant chunks based on the user's query.
    """
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    return results


def get_original_file_path(txt_filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the original file (PDF or HTML) corresponding to a txt filename.
    Uses case-insensitive matching for robustness.
    
    Returns:
        Tuple of (file_path, file_type) where file_type is 'pdf' or 'html'
    """
    files_dict = get_files_in_directory()
    base_name = Path(txt_filename).stem.lower()
    
    # Check for PDF files first
    for ext in ['.pdf']:
        lookup_name = base_name + ext
        if lookup_name in files_dict:
            actual_filename = files_dict[lookup_name]
            return os.path.join(RAW_DOCS_PATH, actual_filename), 'pdf'
    
    # Check for HTML files
    for ext in ['.html', '.htm']:
        lookup_name = base_name + ext
        if lookup_name in files_dict:
            actual_filename = files_dict[lookup_name]
            return os.path.join(RAW_DOCS_PATH, actual_filename), 'html'
    
    return None, None


def display_pdf(file_path: str, key: str):
    """Display a PDF file using streamlit-pdf-viewer."""
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        
        pdf_viewer(
            input=pdf_bytes,
            width=800,
            height=600,
            key=key
        )
        
        # Also provide download button
        st.download_button(
            label="📥 Download PDF",
            data=pdf_bytes,
            file_name=os.path.basename(file_path),
            mime="application/pdf",
            key=f"download_{key}"
        )
        
    except Exception as e:
        st.error(f"Error loading PDF: {e}")


def display_html(file_path: str, key: str):
    """Display an HTML file using iframe component."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
        
        # Embed HTML in a styled container
        components.html(
            f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        padding: 15px;
                        background-color: white;
                        color: black;
                    }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """,
            height=600,
            scrolling=True
        )
        
        # Provide download button
        st.download_button(
            label="📥 Download HTML",
            data=html_content,
            file_name=os.path.basename(file_path),
            mime="text/html",
            key=f"download_{key}"
        )
        
    except Exception as e:
        st.error(f"Error loading HTML: {e}")


def main():
    st.set_page_config(
        page_title="INALCO Document Search",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 INALCO Document Search")
    st.markdown("Search for relevant documents using semantic similarity.")
    
    # Check database path
    if not os.path.exists(DB_PATH):
        st.error(f"❌ ERROR: Database path '{DB_PATH}' not found.")
        st.info("Make sure you're running this script from C:\\Users\\Philo")
        return
    
    # Check raw docs path
    if not os.path.exists(RAW_DOCS_PATH):
        st.warning(f"⚠️ Raw documents folder not found: {RAW_DOCS_PATH}")
    
    # Load resources with caching
    try:
        with st.spinner("Loading AI model..."):
            model = load_model()
        with st.spinner("Connecting to database..."):
            collection = load_collection()
            chunk_count = collection.count()
    except Exception as e:
        st.error(f"❌ Error loading resources: {e}")
        return
    
    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    st.sidebar.success(f"✅ Database loaded\n\n**{chunk_count:,}** chunks available")
    
    k = st.sidebar.slider(
        "Number of documents to return:",
        min_value=1,
        max_value=15,
        value=5
    )
    
    st.sidebar.divider()
    st.sidebar.caption(f"📁 Raw docs: `{RAW_DOCS_PATH}`")
    
    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = None
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = None
    if "search_time" not in st.session_state:
        st.session_state.search_time = 0
    
    # Search interface
    query = st.text_input(
        "🔎 Enter your search query:",
        placeholder="Type your question or keywords here..."
    )
    
    search_clicked = st.button("🔍 Search", type="primary")
    
    if search_clicked:
        if not query.strip():
            st.warning("⚠️ Please enter a search query.")
        else:
            start_time = time.time()
            
            with st.spinner("Searching for relevant documents..."):
                results = find_k_chunks(query, model, collection, k)
            
            elapsed = time.time() - start_time
            
            st.session_state.results = results
            st.session_state.search_time = elapsed
            # Auto-select first result
            if results['ids'][0]:
                st.session_state.selected_index = 0
            else:
                st.session_state.selected_index = None
    
    # Display results
    if st.session_state.results is not None:
        results = st.session_state.results
        num_results = len(results['ids'][0])
        
        st.success(f"**Found {num_results} result(s) in {st.session_state.search_time:.2f} seconds**")
        st.divider()
        
        if num_results == 0:
            st.info("No results found. Try different keywords.")
        else:
            # Two-column layout: results list | document preview
            col_list, col_preview = st.columns([1, 2])
            
            with col_list:
                st.markdown("### 📋 Results")
                
                for i in range(num_results):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    score = 1 - distance
                    
                    filename = metadata.get('file', 'Unknown')
                    doc_type = metadata.get('doc_type', 'Unknown')
                    file_path, file_type = get_original_file_path(filename)
                    
                    # Choose icon based on file type
                    if file_type == 'pdf':
                        icon = "📕"
                    elif file_type == 'html':
                        icon = "🌐"
                    else:
                        icon = "📄"
                    
                    # Truncate filename for display
                    display_name = Path(filename).stem
                    if len(display_name) > 30:
                        display_name = display_name[:30] + "..."
                    
                    # Determine button style
                    is_selected = st.session_state.selected_index == i
                    btn_type = "primary" if is_selected else "secondary"
                    
                    if st.button(
                        f"{icon} {display_name} ({score:.0%})",
                        key=f"result_btn_{i}",
                        use_container_width=True,
                        type=btn_type
                    ):
                        st.session_state.selected_index = i
                        st.rerun()
            
            with col_preview:
                st.markdown("### 📖 Document Preview")
                
                idx = st.session_state.selected_index
                
                if idx is not None and idx < num_results:
                    metadata = results['metadatas'][0][idx]
                    distance = results['distances'][0][idx]
                    score = 1 - distance
                    
                    filename = metadata.get('file', 'Unknown')
                    doc_type = metadata.get('doc_type', 'Unknown')
                    file_path, file_type = get_original_file_path(filename)
                    
                    # Display file info
                    st.markdown(f"**📄 {filename}**")
                    
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.caption(f"**Type:** {doc_type}")
                    with col_info2:
                        st.caption(f"**Relevance:** {score:.2%}")
                    
                    if file_path:
                        st.caption(f"📁 `{file_path}`")
                        st.divider()
                        
                        # Display the document based on type
                        if file_type == 'pdf':
                            display_pdf(file_path, key=f"pdf_{idx}")
                        elif file_type == 'html':
                            display_html(file_path, key=f"html_{idx}")
                    else:
                        st.divider()
                        st.warning(
                            f"⚠️ Original file not found.\n\n"
                            f"Searched for PDF/HTML version of `{filename}` in:\n"
                            f"`{RAW_DOCS_PATH}`"
                        )
                        
                        # Show available files for debugging
                        with st.expander("🔧 Debug: Show available files"):
                            files = get_files_in_directory()
                            base_name = Path(filename).stem.lower()
                            matching = [f for key, f in files.items() if base_name in key]
                            if matching:
                                st.write("Possibly related files:")
                                for f in matching:
                                    st.code(f)
                            else:
                                st.write(f"No files matching '{base_name}' found.")
                else:
                    st.info("👈 Click on a result to preview the document")


if __name__ == "__main__":
    main()