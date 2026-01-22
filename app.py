"""
Canadian Trails & Parks RAG System - Streamlit Frontend
Optimized for Hugging Face Spaces deployment
"""

import os
import re
import time
import json
from typing import Dict, List, Optional, Tuple

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests


# ============================================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ============================================================================
st.set_page_config(
    page_title="Canadian Trails & Parks Explorer",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# CONFIGURATION
# ============================================================================
PROVINCE_TO_REGIONS: Dict[str, List[str]] = {
    "Ontario": ["Ontario South", "Ontario Central", "Ontario North"],
    "Quebec": ["Quebec South", "Quebec North"],
    "British Columbia": ["British Columbia South", "British Columbia North"],
    "Alberta": ["Alberta South", "Alberta North"],
    "Manitoba": ["Manitoba"],
    "Saskatchewan": ["Saskatchewan"],
    "Nova Scotia": ["Nova Scotia"],
    "New Brunswick": ["New Brunswick"],
    "Prince Edward Island": ["Prince Edward Island"],
    "Newfoundland and Labrador": ["Newfoundland"],
    "Yukon": ["Yukon"],
    "Northwest Territories": ["Northwest Territories"],
    "Nunavut": ["Nunavut"],
}

PROVINCE_KEYWORDS: Dict[str, List[str]] = {
    "ontario": ["Ontario South", "Ontario Central", "Ontario North"],
    "quebec": ["Quebec South", "Quebec North"],
    "british columbia": ["British Columbia South", "British Columbia North"],
    "alberta": ["Alberta South", "Alberta North"],
    "bc": ["British Columbia South", "British Columbia North"],
    "nova scotia": ["Nova Scotia"],
    "new brunswick": ["New Brunswick"],
    "manitoba": ["Manitoba"],
    "saskatchewan": ["Saskatchewan"],
    "prince edward island": ["Prince Edward Island"],
    "pei": ["Prince Edward Island"],
    "newfoundland": ["Newfoundland"],
    "newfoundland and labrador": ["Newfoundland"],
    "yukon": ["Yukon"],
    "northwest territories": ["Northwest Territories"],
    "nunavut": ["Nunavut"],
}


class Config:
    """Configuration for the RAG system"""

    # Vector DB settings
    COLLECTION_NAME = "extra_large_minilm"
    DB_PATH = "./data/vector_db"

    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM settings - Groq
    LLM_PROVIDER = "groq"
    LLM_MODEL = "llama-3.1-8b-instant"

    # Retrieval settings
    TOP_K = 5


# ============================================================================
# DATA / RESOURCE LOADERS
# ============================================================================
@st.cache_data(show_spinner=False)
def load_city_to_province() -> Dict[str, str]:
    with open("./data/city_to_province.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(Config.EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_vector_db():
    db_path = f"{Config.DB_PATH}/{Config.COLLECTION_NAME}"
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(Config.COLLECTION_NAME)
    return collection


@st.cache_data(show_spinner=False)
def load_metadata_values() -> Dict[str, List[str]]:
    """
    Sample the collection to discover unique metadata values for:
    - document_type
    - difficulty
    - surface

    Returns dict like:
      {
        "document_type": ["Trail", "Path", ...],
        "difficulty": ["hiking", "unknown", ...],
        "surface": ["ground", "dirt", ...],
      }
    """
    collection = load_vector_db()

    # Sample by asking for many docs around a dummy embedding; metadata only.
    # Chroma doesn't have a pure "scan", so this is an approximation. [web:87]
    dummy_embedding = [0.0] * 384  # same dimension as MiniLM
    res = collection.query(
        query_embeddings=[dummy_embedding],
        n_results=200,
        include=["metadatas"],
    )

    types = set()
    diffs = set()
    surfaces = set()

    for md in res.get("metadatas", [[]])[0]:
        if not isinstance(md, dict):
            continue
        dt = md.get("document_type")
        if isinstance(dt, str) and dt.strip():
            types.add(dt.strip())
        dif = md.get("difficulty")
        if isinstance(dif, str) and dif.strip():
            diffs.add(dif.strip())
        surf = md.get("surface")
        if isinstance(surf, str) and surf.strip():
            surfaces.add(surf.strip())

    def _sorted(values: set) -> List[str]:
        # Simple sort with "unknown" last
        vals = sorted(v for v in values if v.lower() != "unknown")
        if any(v.lower() == "unknown" for v in values):
            vals.append("unknown")
        return vals

    return {
        "document_type": _sorted(types),
        "difficulty": _sorted(diffs),
        "surface": _sorted(surfaces),
    }


# ============================================================================
# HELPERS
# ============================================================================
def get_groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "")


def _contains_phrase(query_lower: str, phrase: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, query_lower) is not None


def extract_location_filter(query: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Returns: (regions, detected_location_str)"""
    query_lower = query.lower()

    try:
        city_to_prov = load_city_to_province()
        for city_key, prov in city_to_prov.items():
            if _contains_phrase(query_lower, city_key):
                return PROVINCE_TO_REGIONS.get(prov), city_key
    except FileNotFoundError:
        pass

    for province_key, regions in PROVINCE_KEYWORDS.items():
        if _contains_phrase(query_lower, province_key):
            return regions, province_key

    return None, None


def build_where_filter(
    regions: Optional[List[str]],
    ui_filters: Dict[str, str],
) -> Optional[Dict]:
    filters = []

    if regions:
        filters.append({"region": {"$in": regions}})

    for key, val in ui_filters.items():
        if val and val != "Any":
            filters.append({key: val})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def retrieve_documents(
    query: str,
    collection,
    embedding_model,
    ui_filters: Dict[str, str],
) -> Dict:
    start_time = time.time()
    query_embedding = embedding_model.encode(query).tolist()

    regions, detected_location = extract_location_filter(query)
    where_filter = build_where_filter(regions, ui_filters)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=Config.TOP_K * 3,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    retrieval_time = time.time() - start_time

    seen_titles = set()
    formatted_results = []

    for i, (doc, metadata, distance) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        title = metadata.get("document_title", f"doc_{i}")
        region = metadata.get("region", "")

        if title in seen_titles:
            continue

        if detected_location and regions:
            title_lower = title.lower()
            if detected_location in title_lower and region not in regions:
                continue

        seen_titles.add(title)

        formatted_results.append(
            {
                "rank": len(formatted_results) + 1,
                "content": doc,
                "metadata": metadata,
                "similarity": 1 - distance,
                "distance": distance,
            }
        )

        if len(formatted_results) >= Config.TOP_K:
            break

    return {
        "results": formatted_results,
        "retrieval_time": retrieval_time,
        "query": query,
        "filters_applied": where_filter,
        "detected_location": detected_location,
        "detected_regions": regions,
        "ui_filters": ui_filters,
    }


def generate_answer(query: str, retrieved_docs: List[Dict]) -> Dict:
    """Generate answer with detailed grounded section in paragraph format + helpful suggestions."""
    api_key = get_groq_api_key()

    if not api_key:
        return {
            "answer": "⚠️ **API Key Missing**: Please set GROQ_API_KEY (HF Spaces Secrets or local env var).",
            "generation_time": 0.0,
            "sources": [],
            "tokens_used": 0,
        }

    # Build context
    context_parts = []
    sources = []

    for i, doc in enumerate(retrieved_docs, 1):
        metadata = doc.get("metadata", {})
        content = (doc.get("content") or "")[:700]
        title = metadata.get("document_title", "Unknown")
        region = metadata.get("region", "Unknown region")
        sources.append(title)
        context_parts.append(f"[Source {i}] {title} - {region}\n{content}\n")

    context = "\n".join(context_parts)
    n_sources = len(retrieved_docs)

    # Prompt: detailed paragraph format
    prompt = f"""You are an assistant for Canadian trails and parks.

You MUST produce TWO sections with these exact headings:

## 1) Grounded Answer (from retrieved sources):
- Write in FLOWING PARAGRAPH format, NOT bullet points or numbered lists.
- Use ONLY information from the Context below.
- Discuss each trail in detail within natural paragraphs.
- For each trail mentioned, include: trail name, type, surface, difficulty, permitted activities (bikes/horses), region, and any special features from the Context.
- Connect trails thematically (e.g., group by surface type, difficulty, or location) to make the answer readable.
- Cite sources using [Source N] where N is between 1 and {n_sources} after mentioning each trail.
- Do NOT include URLs, websites, or external references unless they appear verbatim in the Context.
- Be descriptive and conversational—write as if explaining to a friend.
- If information is missing (like difficulty or activities), acknowledge it briefly but focus on what IS available.

## 2) Suggestions (general advice, NOT from sources):
- Provide brief, practical next steps in 2-3 sentences.
- Write in friendly, conversational paragraph format.
- Do NOT cite any sources in this section (no [Source N]).
- Do NOT mention specific trail names unless they appeared in the Context above.

Context (retrieved from database):
{context}

User Question: {query}

Now provide your detailed answer in flowing paragraph format:"""

    start_time = time.time()

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": Config.LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1000,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        answer = result["choices"][0]["message"]["content"]
        tokens = result.get("usage", {}).get("total_tokens", 0)

        # Validate citations
        citation_pattern = r"\[Source\s+(\d+)\]"
        found_citations = [int(n) for n in re.findall(citation_pattern, answer)]
        invalid_citations = [n for n in found_citations if n < 1 or n > n_sources]

        if invalid_citations:
            for invalid_n in set(invalid_citations):
                answer = re.sub(rf"\[Source\s+{invalid_n}\]", "", answer)
            
            answer = (
                f"⚠️ **Note:** Some citations were removed because they referenced sources not in the retrieved results.\n\n{answer}"
            )

        if n_sources == 0:
            answer = re.sub(r"\[Source\s+\d+\]", "", answer)

    except requests.exceptions.RequestException as e:
        answer = f"⚠️ **Error generating answer**: {str(e)}\n\nPlease check your API key and try again."
        tokens = 0

    generation_time = time.time() - start_time

    return {
        "answer": answer,
        "generation_time": generation_time,
        "tokens_used": tokens,
        "sources": sources,
    }



# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.markdown('<h1 class="main-header">🏔️ Canadian Trails & Parks Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by AI • 277K+ Trails • 60+ Parks</p>', unsafe_allow_html=True)

    # Load metadata values once for filters
    md_values = load_metadata_values()
    type_options = ["Any"] + md_values.get("document_type", [])
    diff_options = ["Any"] + md_values.get("difficulty", [])
    surface_options = ["Any"] + md_values.get("surface", [])

    with st.sidebar:
        st.markdown("### ⚙️ About")
        st.markdown(
            """
This RAG system helps you discover Canadian trails and parks using:
- **277,468** trail records
- **60+** Parks Canada locations
- **Free AI** (Groq + Llama 3.1)
- **Vector search** for intelligent retrieval
"""
        )

        st.markdown("---")
        st.markdown("### 🎛️ Filters (from data)")

        trail_type = st.selectbox(
            "Trail type (document_type)",
            options=type_options,
            index=0,
        )

        difficulty = st.selectbox(
            "Difficulty",
            options=diff_options,
            index=0,
        )

        surface = st.selectbox(
            "Surface",
            options=surface_options,
            index=0,
        )

        st.caption("Options are auto-populated from your ChromaDB metadata sample.")

        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        example_queries = [
            "What are hiking trails in British Columbia?",
            "Find wheelchair accessible trails in Ontario",
            "Tell me about Banff National Park",
            "What trails allow bicycles in Quebec?",
            "Find beginner-friendly trails near Toronto",
            "Easy trails near Cambridge Ontario",
        ]
        for eq in example_queries:
            if st.button(eq, key=f"ex_{eq}", use_container_width=True):
                st.session_state.query_input = eq
                st.rerun()

        st.markdown("---")
        st.markdown("### 📊 System Info")
        st.markdown(
            f"""
- **Vector DB**: {Config.COLLECTION_NAME}
- **Embeddings**: MiniLM-L6-v2
- **LLM**: Llama 3.1 8B (Groq)
- **Top-K**: {Config.TOP_K}
"""
        )

    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    query = st.text_input(
        "🔍 Ask about Canadian trails and parks:",
        value=st.session_state.query_input,
        placeholder="e.g., What are the best hiking trails near Vancouver?",
        key="main_query_input",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    if clear_button:
        st.session_state.query_input = ""
        st.rerun()

    if search_button and not query:
        st.warning("⚠️ Please enter a query")

    if search_button and query:
        ui_filters = {"document_type": trail_type, "difficulty": difficulty, "surface": surface}

        with st.spinner("🔍 Searching knowledge base..."):
            try:
                embedding_model = load_embedding_model()
                collection = load_vector_db()

                retrieval_results = retrieve_documents(query, collection, embedding_model, ui_filters)
                generation_results = generate_answer(query, retrieval_results["results"])

                st.markdown("---")

                active_filters = {k: v for k, v in ui_filters.items() if v != "Any"}
                if retrieval_results.get("detected_location") and retrieval_results.get("detected_regions"):
                    st.info(
                        f"🗺️ **Location Filter Active**: matched **{retrieval_results['detected_location']}** "
                        f"(searching in: {', '.join(retrieval_results['detected_regions'])})"
                    )
                if active_filters:
                    st.info(
                        "🎛️ **Dashboard Filters Active**: "
                        + ", ".join([f"{k}={v}" for k, v in active_filters.items()])
                    )

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(
                        "⚡ Total Time",
                        f"{retrieval_results['retrieval_time'] + generation_results['generation_time']:.2f}s",
                    )
                with c2:
                    st.metric("📚 Sources", len(retrieval_results["results"]))
                with c3:
                    st.metric("🎯 Retrieval", f"{retrieval_results['retrieval_time']:.2f}s")
                with c4:
                    st.metric("💬 Generation", f"{generation_results['generation_time']:.2f}s")

                st.markdown("---")
                st.markdown("### 💡 Answer")
                st.markdown(generation_results["answer"])

                st.markdown("---")
                st.markdown("### 📖 Retrieved Sources")
                for doc in retrieval_results["results"]:
                    with st.expander(
                        f"**{doc['rank']}. {doc['metadata'].get('document_title', 'Unknown')}** "
                        f"(Similarity: {doc['similarity']:.3f})"
                    ):
                        metadata = doc["metadata"]
                        cols = st.columns(3)
                        with cols[0]:
                            if metadata.get("region"):
                                st.markdown(f"📍 **Region**: {metadata['region']}")
                        with cols[1]:
                            if metadata.get("document_type"):
                                st.markdown(f"🏷️ **Type**: {metadata['document_type']}")
                        with cols[2]:
                            if metadata.get("difficulty"):
                                st.markdown(f"⚡ **Difficulty**: {metadata['difficulty']}")

                        st.markdown("**Content:**")
                        content = doc["content"]
                        st.markdown(content[:500] + "..." if len(content) > 500 else content)

                        if metadata.get("surface"):
                            st.markdown(f"*Surface: {metadata['surface']}*")

            except Exception as e:
                st.error(f"❌ **Error**: {str(e)}")
                st.markdown("Please check:")
                st.markdown("- Vector database exists at `./data/vector_db/extra_large_minilm`")
                st.markdown("- Collection name matches `extra_large_minilm`")
                st.markdown("- `./data/city_to_province.json` exists (or rely on province keywords)")
                st.markdown("- GROQ_API_KEY is set in env / HF Secrets")
                st.markdown("- Dependencies are installed")

    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.9rem;">'
        "Built with Streamlit • Data from OpenStreetMap & Parks Canada • Free AI by Groq"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
