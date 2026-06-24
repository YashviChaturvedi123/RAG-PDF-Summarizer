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

@st.cache_resource
def load_embedder():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_summarizer():
    return pipeline(
        "text-generation",
        model="gpt2",
        device=-1
    )

def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with st.spinner("Reading PDF..."):
        reader = PdfReader(uploaded_file)

        raw_text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if not raw_text.strip():
        st.error("No text found in PDF.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(raw_text)

    user_query = st.text_input(
        "Ask a question (optional)",
        placeholder="What is this document about?"
    )

    if st.button("✨ Generate Summary"):

        with st.spinner("Creating embeddings..."):
            embedder = load_embedder()

            embeddings = normalize(
                embedder.encode(
                    chunks,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            ).astype("float32")

        with st.spinner("Retrieving relevant chunks..."):

            dim = embeddings.shape[1]

            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

            query = (
                user_query
                if user_query.strip()
                else "Summarize this document"
            )

            query_embedding = normalize(
                embedder.encode(
                    [query],
                    convert_to_numpy=True
                )
            ).astype("float32")

            top_k = min(5, len(chunks))

            _, indices = index.search(
                query_embedding,
                top_k
            )

            top_chunks = [
                chunks[i] for i in indices[0]
            ]

        with st.spinner("Generating summary..."):

            summarizer = load_summarizer()

            context = " ".join(top_chunks)[:1500]

            prompt = f"""
            Summarize the following text:

            {context}

            Summary:
            """

            result = summarizer(
                prompt,
                max_new_tokens=150,
                do_sample=False
            )

            summary = result[0]["generated_text"]

        st.subheader("📝 Summary")
        st.write(summary)

        with st.expander("Retrieved Chunks"):
            for i, chunk in enumerate(top_chunks, 1):
                st.markdown(f"**Chunk {i}**")
                st.text(
                    textwrap.fill(chunk, width=90)
                )
                st.divider()