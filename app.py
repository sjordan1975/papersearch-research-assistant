"""PaperSearch Research Assistant — Streamlit Web UI.

Run with:
    streamlit run app.py

Features:
- Config sidebar (chunking, embedding, retriever, top-k)
- Q&A interface with natural language queries
- Answer display with inline citations
- Expandable source viewer for retrieved chunks
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.embedding.sentence_transformer import SentenceTransformerEmbedder
from src.generation.answer import AnswerGenerator
from src.generation.llm import LiteLLMClient
from src.models import Chunk, ChunkMetadata
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.stores.faiss_store import FaissVectorStore

# Page config
st.set_page_config(
    page_title="PaperSearch Research Assistant",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Caching and loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_embedder(model_name: str) -> SentenceTransformerEmbedder:
    """Load embedding model (cached across sessions)."""
    return SentenceTransformerEmbedder(model_name=model_name)


@st.cache_data
def load_chunks(chunks_path: str) -> list[dict]:
    """Load chunks from JSON file (cached)."""
    data = json.loads(Path(chunks_path).read_text())
    return data


def chunks_to_objects(chunks_data: list[dict]) -> list[Chunk]:
    """Convert chunk dicts to Chunk objects."""
    chunks = []
    for item in chunks_data:
        metadata = ChunkMetadata(**item["metadata"])
        chunk = Chunk(
            id=item["id"],
            content=item["content"],
            metadata=metadata,
        )
        chunks.append(chunk)
    return chunks


@st.cache_resource
def load_vector_store(index_path: str, dimension: int) -> FaissVectorStore:
    """Load FAISS index (cached)."""
    store = FaissVectorStore(dimension=dimension)
    store.load(index_path)
    return store


def get_available_configs(cache_dir: Path) -> list[str]:
    """Find available ingested configs."""
    chunks_dir = cache_dir / "chunks"
    if not chunks_dir.exists():
        return []
    return sorted([p.stem for p in chunks_dir.glob("*.json")])


def build_retriever(
    retriever_type: str,
    chunks: list[Chunk],
    embedder: SentenceTransformerEmbedder,
    vector_store: FaissVectorStore,
):
    """Build the selected retriever."""
    chunk_lookup = {str(c.id): c for c in chunks}

    if retriever_type == "Dense":
        return DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            chunk_lookup=chunk_lookup,
        )
    elif retriever_type == "BM25":
        return BM25Retriever(chunks=chunks)
    else:  # Hybrid
        dense = DenseRetriever(
            embedder=embedder,
            vector_store=vector_store,
            chunk_lookup=chunk_lookup,
        )
        bm25 = BM25Retriever(chunks=chunks)
        return HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render configuration sidebar."""
    st.sidebar.title("Configuration")

    cache_dir = Path("data/cache")
    configs = get_available_configs(cache_dir)

    if not configs:
        st.sidebar.warning(
            "No ingested data found. Run `python scripts/ingest.py` first."
        )
        return None

    # Config selection
    config = st.sidebar.selectbox(
        "Ingested Config",
        configs,
        help="Select a pre-ingested configuration",
    )

    # Retriever selection
    retriever_type = st.sidebar.selectbox(
        "Retriever",
        ["Hybrid", "Dense", "BM25"],
        help="Hybrid combines dense + BM25 for best results",
    )

    # Top-K
    top_k = st.sidebar.slider(
        "Top-K Results",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of chunks to retrieve",
    )

    # LLM model
    model = st.sidebar.selectbox(
        "LLM Model",
        ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"],
        help="Model for answer generation",
    )

    st.sidebar.divider()

    # Info
    st.sidebar.info(
        f"**Config:** {config}\n\n"
        f"**Retriever:** {retriever_type}\n\n"
        f"**Top-K:** {top_k}"
    )

    return {
        "config": config,
        "retriever_type": retriever_type,
        "top_k": top_k,
        "model": model,
        "cache_dir": cache_dir,
    }


def render_answer(response):
    """Render the answer with citations."""
    st.markdown("### Answer")
    st.markdown(response.answer)

    if response.citations:
        st.markdown("---")
        st.markdown("### Citations")

        for i, citation in enumerate(response.citations, 1):
            with st.expander(f"[{i}] {citation.source}", expanded=False):
                if citation.page_number:
                    st.caption(f"Page {citation.page_number}")
                st.text(citation.text_snippet)
                if citation.relevance_score:
                    st.caption(f"Relevance: {citation.relevance_score:.3f}")


def render_sources(results):
    """Render retrieved source chunks."""
    st.markdown("### Retrieved Sources")

    for i, result in enumerate(results, 1):
        chunk = result.chunk
        score = result.score
        source = chunk.metadata.source

        with st.expander(
            f"[{i}] {source} (score: {score:.3f})",
            expanded=(i == 1),
        ):
            st.markdown(chunk.content)
            st.caption(
                f"Chunk {chunk.metadata.chunk_index} | "
                f"Section {chunk.metadata.section_index or 'N/A'} | "
                f"Strategy: {chunk.metadata.chunking_strategy.value}"
            )


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    st.title("📚 PaperSearch Research Assistant")
    st.caption("Ask questions about research papers with cited answers")

    # Sidebar config
    settings = render_sidebar()

    if settings is None:
        st.stop()

    # Load resources
    config = settings["config"]
    cache_dir = settings["cache_dir"]

    chunks_path = cache_dir / "chunks" / f"{config}.json"
    index_path = cache_dir / "indices" / f"{config}.faiss"

    if not chunks_path.exists() or not index_path.exists():
        st.error(f"Missing cache files for config: {config}")
        st.stop()

    # Extract embedding model from config
    # Config format: chunking_size_overlap_embedding
    embedding_model = config.rsplit("_", 1)[-1]

    # Load components (cached)
    with st.spinner("Loading embedding model..."):
        embedder = load_embedder(embedding_model)

    chunks_data = load_chunks(str(chunks_path))
    chunks = chunks_to_objects(chunks_data)

    vector_store = load_vector_store(str(index_path), embedder.dimension)

    # Build retriever
    retriever = build_retriever(
        settings["retriever_type"],
        chunks,
        embedder,
        vector_store,
    )

    # LLM
    llm = LiteLLMClient(model=settings["model"])
    generator = AnswerGenerator(llm=llm)

    # Status
    st.success(f"Loaded {len(chunks):,} chunks from {config}")

    # Query input
    st.markdown("---")
    query = st.text_input(
        "Ask a question",
        placeholder="e.g., What are the main challenges in transformer architectures?",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("Search", type="primary", use_container_width=True)

    if query and search_button:
        with st.spinner("Retrieving relevant chunks..."):
            results = retriever.retrieve(query, top_k=settings["top_k"])

        if not results:
            st.warning("No relevant chunks found for this query.")
            st.stop()

        with st.spinner("Generating answer..."):
            response = generator.generate(query, results)

        # Display results in tabs
        tab1, tab2 = st.tabs(["Answer", "Sources"])

        with tab1:
            render_answer(response)

        with tab2:
            render_sources(results)

    # Footer
    st.markdown("---")
    st.caption(
        "Built with Streamlit | "
        "Embeddings: SentenceTransformers | "
        "LLM: LiteLLM with Langfuse tracing"
    )


if __name__ == "__main__":
    main()
