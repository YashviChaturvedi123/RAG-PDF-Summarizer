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
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1
    )


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


# Load models once
with st.spinner("Loading AI models (first launch may take a few minutes)..."):
    embedder = load_embedder()
    summarizer = load_summarizer()


uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    with st.spinner("Reading PDF..."):
        reader = PdfReader(uploaded_file)

        raw_text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if not raw_text.strip():
        st.error("No readable text found in the PDF.")
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

        try:

            with st.spinner("Creating embeddings..."):

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
                    user_query.strip()
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
                    chunks[i]
                    for i in indices[0]
                ]

            with st.spinner("Generating summary..."):

                context = " ".join(top_chunks)

                # DistilBART accepts about 1024 tokens
                context = context[:3000]

                result = summarizer(
                    context,
                    max_length=150,
                    min_length=40,
                    do_sample=False
                )

                summary = result[0]["summary_text"]

            st.success("Summary generated successfully!")

            st.subheader("📝 Summary")
            st.write(summary)

            with st.expander("Retrieved Chunks"):

                for i, chunk in enumerate(top_chunks, start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(
                        textwrap.fill(
                            chunk,
                            width=90
                        )
                    )
                    st.divider()

        except Exception as e:
            st.error("An error occurred while generating the summary.")
            st.exception(e)
