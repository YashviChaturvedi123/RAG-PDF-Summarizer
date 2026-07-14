import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import textwrap
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="RAG PDF Summarizer", page_icon="📄")
st.title("📄 RAG PDF Summarizer")
st.caption("Upload a PDF and generate a summary using RAG")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
   
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading summarization model...")
def load_summarizer():

    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-6-6", 
        device=-1,
        framework="pt",
    )


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)



embedder = load_embedder()
summarizer = load_summarizer()

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading PDF..."):
        reader = PdfReader(uploaded_file)
        raw_text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    if not raw_text.strip():
        st.error("No readable text found in the PDF.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)


    st.info(f"PDF parsed into **{len(chunks)} chunks**.")

    user_query = st.text_input(
        "Ask a question (optional)",
        placeholder="What is this document about?"
    )

    if st.button("✨ Generate Summary"):
        try:
            with st.spinner("Creating embeddings..."):
                embeddings = normalize(
                    embedder.encode(
                        chunks,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=32, 
                    )
                ).astype("float32")

            with st.spinner("Retrieving relevant chunks..."):
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)

                query = user_query.strip() or "Summarize this document"
                query_vec = normalize(
                    embedder.encode([query], convert_to_numpy=True)
                ).astype("float32")

                top_k = min(5, len(chunks))
                _, indices = index.search(query_vec, top_k)
                top_chunks = [chunks[i] for i in indices[0]]

            with st.spinner("Generating summary (this may take 20–60s on CPU)..."):
                context = " ".join(top_chunks)[:3000]  

                
                word_count = len(context.split())
                max_len = min(150, max(60, word_count // 3))
                min_len = min(40, max_len - 10)

                result = summarizer(
                    context,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    truncation=True,  
                )
                summary = result[0]["summary_text"]

            st.success("Summary generated!")
            st.subheader("📝 Summary")
            st.write(summary)

            with st.expander("🔍 Retrieved Chunks"):
                for i, chunk in enumerate(top_chunks, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(textwrap.fill(chunk, width=90))
                    st.divider()

        except Exception as e:
            st.error("An error occurred.")
            st.exception(e)
