import io
import os
import re
import hashlib
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Basic settings
# ---------------------------------------------------------

SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Read model name from Streamlit secrets.
# Falls back to this model if GROQ_MODEL is not present.
GROQ_MODEL = st.secrets.get(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# ---------------------------------------------------------
# Cached models
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------
# Document extraction
# ---------------------------------------------------------

def extract_pdf(file_bytes, filename):
    """Extract PDF text and keep the PDF page number."""
    pages = []

    reader = PdfReader(io.BytesIO(file_bytes))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "filename": filename,
                "page": page_number,
                "text": text.strip()
            })

    return pages


def extract_docx(file_bytes, filename):
    """Extract DOCX text. Page numbers are not reliably available."""
    document = Document(io.BytesIO(file_bytes))

    text = "\n".join(
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    if not text:
        return []

    return [{
        "filename": filename,
        "page": None,
        "text": text
    }]


def extract_txt(file_bytes, filename):
    """Extract plain text."""
    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    ).strip()

    if not text:
        return []

    return [{
        "filename": filename,
        "page": None,
        "text": text
    }]


def extract_md(file_bytes, filename):
    """Extract Markdown text."""
    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    ).strip()

    if not text:
        return []

    return [{
        "filename": filename,
        "page": None,
        "text": text
    }]


def extract_document(filename, file_bytes):
    """Choose the correct extractor from the file extension."""

    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)

    if extension == ".docx":
        return extract_docx(file_bytes, filename)

    if extension == ".txt":
        return extract_txt(file_bytes, filename)

    if extension == ".md":
        return extract_md(file_bytes, filename)

    return []


# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------

def chunk_text(pages, chunk_size=800, overlap=120):
    """
    Split extracted text into overlapping chunks.

    Every chunk keeps:
    - filename
    - page
    - text
    """

    chunks = []

    for item in pages:

        text = item["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "filename": item["filename"],
                    "page": item["page"],
                    "text": chunk
                })

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def process_document(filename, file_bytes):
    """
    Extract, chunk and embed one document.

    Streamlit caches this result, so the same document
    does not need to be processed and embedded again.
    """

    pages = extract_document(
        filename,
        file_bytes
    )

    chunks = chunk_text(pages)

    if not chunks:
        return [], np.empty(
            (0, 384),
            dtype="float32"
        )

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    return chunks, embeddings


# ---------------------------------------------------------
# FAISS
# ---------------------------------------------------------

def create_faiss_index(embeddings):
    """Create cosine-similarity FAISS index."""

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ---------------------------------------------------------
# Keyword Search
# ---------------------------------------------------------

STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "does",
    "do",
    "did",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "from",
    "as",
    "by",
    "be",
    "about",
    "can",
    "could",
    "would",
    "should",
    "tell",
    "me"
}


def important_words(text):
    """Return important words from a question or chunk."""

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    return [
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 2
    ]


def keyword_scores(question, chunks):
    """Score chunks using matching important words."""

    question_words = set(
        important_words(question)
    )

    scores = []

    for chunk in chunks:

        chunk_words = set(
            important_words(chunk["text"])
        )

        if not question_words:
            scores.append(0.0)
            continue

        matches = len(
            question_words.intersection(chunk_words)
        )

        score = matches / len(question_words)

        scores.append(score)

    return np.array(
        scores,
        dtype="float32"
    )


# ---------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------

def hybrid_search(
    question,
    chunks,
    embeddings,
    index,
    top_k=5
):
    """
    Combine semantic search and keyword search.

    70% semantic similarity
    30% keyword matching
    """

    if not chunks:
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    semantic_scores = np.zeros(
        len(chunks),
        dtype="float32"
    )

    search_count = min(
        max(top_k * 4, 10),
        len(chunks)
    )

    distances, indices = index.search(
        question_embedding,
        search_count
    )

    for score, index_number in zip(
        distances[0],
        indices[0]
    ):

        if index_number >= 0:
            semantic_scores[index_number] = float(
                score
            )

    keyword = keyword_scores(
        question,
        chunks
    )

    # Convert [-1, 1] to approximately [0, 1].
    semantic_normalized = (
        semantic_scores + 1.0
    ) / 2.0

    hybrid_scores = (
        0.70 * semantic_normalized
        +
        0.30 * keyword
    )

    best_indices = np.argsort(
        hybrid_scores
    )[::-1][:top_k]

    results = []

    for index_number in best_indices:

        result = dict(
            chunks[index_number]
        )

        result["semantic_score"] = float(
            semantic_normalized[index_number]
        )

        result["keyword_score"] = float(
            keyword[index_number]
        )

        result["hybrid_score"] = float(
            hybrid_scores[index_number]
        )

        results.append(result)

    return results


# ---------------------------------------------------------
# Google Drive
# ---------------------------------------------------------

DRIVE_ID_PATTERNS = [
    r"/file/d/([a-zA-Z0-9_-]{10,})",
    r"/folders/([a-zA-Z0-9_-]{10,})",
    r"[?&]id=([a-zA-Z0-9_-]{10,})",
    r"/d/([a-zA-Z0-9_-]{10,})",
]


def extract_drive_id(url):
    """
    Pull the Drive file/folder ID out of
    common Google Drive URL formats.
    """

    for pattern in DRIVE_ID_PATTERNS:

        match = re.search(
            pattern,
            url
        )

        if match:
            return match.group(1)

    # Also support a bare Drive ID.
    if re.fullmatch(
        r"[a-zA-Z0-9_-]{10,}",
        url.strip()
    ):
        return url.strip()

    return None


def safe_call(function, **kwargs):
    """
    Call a gdown function while remaining compatible
    with older and newer gdown versions.

    If the installed version does not support a keyword,
    remove that keyword and retry.
    """

    while True:

        try:
            return function(**kwargs)

        except TypeError as error:

            message = str(error)

            match = re.search(
                r"unexpected keyword argument '(\w+)'",
                message
            )

            if (
                match
                and match.group(1) in kwargs
            ):

                kwargs.pop(
                    match.group(1)
                )

                continue

            raise


def download_drive_file(
    url,
    output_dir
):
    """
    Download one Google Drive file.

    Works with older and newer gdown versions.
    """

    file_id = extract_drive_id(url)

    if file_id:

        direct_url = (
            f"https://drive.google.com/uc?id={file_id}"
        )

    else:

        direct_url = url

    output_path = safe_call(
        gdown.download,
        url=direct_url,
        output=str(output_dir) + os.sep,
        quiet=True,
        fuzzy=True,
    )

    if not output_path:

        output_path = safe_call(
            gdown.download,
            id=file_id,
            output=str(output_dir) + os.sep,
            quiet=True,
        )

    return (
        Path(output_path)
        if output_path
        else None
    )


def load_from_drive(url):
    """
    Download a public/shared Google Drive
    file or folder.

    Supported:
    PDF
    DOCX
    TXT
    MD

    Private files requiring Google login
    are not supported by this simple version.
    """

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="drive_docs_"
        )
    )

    # ---------------------------------------------
    # Google Drive folder
    # ---------------------------------------------

    if "/folders/" in url:

        safe_call(
            gdown.download_folder,
            url=url,
            output=str(temp_dir),
            quiet=True,
            use_cookies=False,
        )

        files = [
            path
            for path in temp_dir.rglob("*")
            if path.is_file()
        ]

    # ---------------------------------------------
    # Google Drive file
    # ---------------------------------------------

    else:

        downloaded = download_drive_file(
            url,
            temp_dir
        )

        files = (
            [downloaded]
            if downloaded
            else []
        )

        if not files:

            files = [
                path
                for path in temp_dir.rglob("*")
                if path.is_file()
            ]

    # ---------------------------------------------
    # Keep supported file types only
    # ---------------------------------------------

    return [
        path
        for path in files
        if path.suffix.lower().lstrip(".")
        in SUPPORTED_TYPES
    ]


# ---------------------------------------------------------
# File Fingerprint
# ---------------------------------------------------------

def file_fingerprint(
    filename,
    file_bytes
):
    """Create a unique fingerprint for a document."""

    return hashlib.sha256(
        filename.encode("utf-8")
        +
        file_bytes
    ).hexdigest()


# ---------------------------------------------------------
# Build Document Collection
# ---------------------------------------------------------

def build_collection(files):
    """
    Process all documents and create one combined
    FAISS index.
    """

    all_chunks = []
    all_embeddings = []

    for filename, file_bytes in files:

        chunks, embeddings = process_document(
            filename,
            file_bytes
        )

        if chunks:

            all_chunks.extend(
                chunks
            )

            all_embeddings.append(
                embeddings
            )

    if not all_embeddings:

        return (
            [],
            None,
            np.empty(
                (0, 384),
                dtype="float32"
            )
        )

    embeddings = np.vstack(
        all_embeddings
    ).astype("float32")

    index = create_faiss_index(
        embeddings
    )

    return (
        all_chunks,
        index,
        embeddings
    )


# ---------------------------------------------------------
# Groq
# ---------------------------------------------------------

def get_groq_client():
    """Read the Groq API key from Streamlit secrets."""

    api_key = st.secrets.get(
        "GROQ_API_KEY",
        ""
    )

    if not api_key:
        return None

    return Groq(
        api_key=api_key
    )


def answer_question(
    question,
    retrieved_chunks
):
    """
    Ask Groq to answer only from
    retrieved document context.
    """

    client = get_groq_client()

    if client is None:

        return (
            "GROQ_API_KEY is not configured "
            "in Streamlit secrets."
        )

    context_parts = []

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        page = (
            f"Page {chunk['page']}"
            if chunk["page"] is not None
            else "Page not available"
        )

        context_parts.append(
            f"[Source {number} | "
            f"{chunk['filename']} | "
            f"{page}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
You are an AI document assistant.

Answer the user's question ONLY using
the provided document context.

Rules:

1. Do not use outside knowledge.
2. Do not invent or guess information.
3. If the answer is not available in the
   context, say:

"I could not find that information in the
provided documents."

4. Give a clear and concise answer.
5. You may combine information from multiple
   provided chunks.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# Global Styling
# ---------------------------------------------------------

st.markdown("""
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap'
);

html,
body,
[class*="css"] {
    font-family: 'Inter',
    -apple-system,
    sans-serif;
}

:root {

    --bg-primary: #0b0e14;
    --bg-secondary: #131722;
    --bg-card: #161b27;

    --border-subtle: #232838;

    --accent-1: #6366f1;
    --accent-2: #8b5cf6;

    --accent-glow:
        rgba(99, 102, 241, 0.35);

    --text-primary: #f1f3f9;
    --text-secondary: #9ca3af;

    --success: #22c55e;
    --warning: #f59e0b;
}


.stApp {

    background:

        radial-gradient(
            circle at 15% 0%,
            rgba(99,102,241,0.12)
            0%,
            transparent 45%
        ),

        radial-gradient(
            circle at 85% 20%,
            rgba(139,92,246,0.10)
            0%,
            transparent 45%
        ),

        var(--bg-primary);
}


/* ---------- Hero ---------- */

.hero-wrap {

    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.18) 0%,
            rgba(139,92,246,0.10) 100%
        );

    border:
        1px solid var(--border-subtle);

    border-radius: 20px;

    padding:
        2.2rem 2rem;

    margin-bottom:
        1.8rem;

    box-shadow:
        0 8px 30px
        rgba(0,0,0,0.35);
}


.hero-title {

    font-size:
        2.1rem;

    font-weight:
        800;

    color:
        var(--text-primary) !important;

    margin:
        0 0 0.35rem 0;

    letter-spacing:
        -0.03em;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #c7d2fe
        );

    -webkit-background-clip:
        text;

    -webkit-text-fill-color:
        transparent;
}


.hero-sub {

    color:
        var(--text-secondary) !important;

    font-size:
        0.98rem;

    margin:
        0;
}


/* ---------- Section headers ---------- */

.section-label {

    display:
        flex;

    align-items:
        center;

    gap:
        0.6rem;

    color:
        var(--text-primary) !important;

    font-weight:
        700;

    font-size:
        1.05rem;

    margin:
        1.6rem 0 0.7rem 0;
}


.section-badge {

    display:
        inline-flex;

    align-items:
        center;

    justify-content:
        center;

    width:
        26px;

    height:
        26px;

    border-radius:
        8px;

    background:
        linear-gradient(
            135deg,
            var(--accent-1),
            var(--accent-2)
        );

    color:
        white !important;

    font-size:
        0.85rem;

    font-weight:
        700;

    flex-shrink:
        0;
}


/* ---------- Cards ---------- */

.glass-card {

    background:
        var(--bg-card);

    border:
        1px solid var(--border-subtle);

    border-radius:
        14px;

    padding:
        1.1rem 1.3rem;

    color:
        var(--text-primary) !important;
}


/* ---------- Metrics ---------- */

[data-testid="stMetric"] {

    background:
        var(--bg-card);

    border:
        1px solid var(--border-subtle);

    border-radius:
        14px;

    padding:
        1rem 1rem 0.85rem 1rem;

    box-shadow:
        0 4px 14px
        rgba(0,0,0,0.25);
}


[data-testid="stMetricLabel"] {

    color:
        var(--text-secondary) !important;
}


[data-testid="stMetricValue"] {

    color:
        var(--text-primary) !important;

    font-weight:
        700;
}


/* ---------- Buttons ---------- */

.stButton > button {

    background:
        linear-gradient(
            135deg,
            var(--accent-1),
            var(--accent-2)
        );

    color:
        #ffffff !important;

    border:
        none;

    border-radius:
        10px;

    font-weight:
        600;

    padding:
        0.55rem 1.2rem;

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease;

    box-shadow:
        0 4px 14px
        var(--accent-glow);
}


.stButton > button:hover {

    transform:
        translateY(-1px);

    box-shadow:
        0 6px 20px
        var(--accent-glow);
}


.stButton > button:active {

    transform:
        translateY(0px);
}


/* ---------- Text inputs ---------- */

.stTextInput input {

    background:
        var(--bg-card) !important;

    color:
        var(--text-primary) !important;

    border:
        1px solid
        var(--border-subtle) !important;

    border-radius:
        10px !important;
}


.stTextInput input::placeholder {

    color:
        #5c6478 !important;
}


.stTextInput input:focus {

    border-color:
        var(--accent-1) !important;

    box-shadow:
        0 0 0 3px
        var(--accent-glow) !important;
}


.stTextInput label {

    color:
        var(--text-secondary) !important;
}


/* ---------- File uploader ---------- */

[data-testid="stFileUploader"] {

    background:
        var(--bg-card) !important;

    border:
        1.5px dashed
        var(--border-subtle) !important;

    border-radius:
        14px !important;

    padding:
        0.9rem !important;
}


[data-testid="stFileUploader"] section {

    background:
        var(--bg-card) !important;
}


[data-testid="stFileUploaderDropzone"] {

    background:
        var(--bg-card) !important;
}


[data-testid="stFileUploader"] * {

    color:
        var(--text-primary) !important;
}


[data-testid="stFileUploader"] small {

    color:
        var(--text-secondary) !important;
}


[data-testid="stFileUploader"] button {

    background:
        var(--bg-secondary) !important;

    color:
        var(--text-primary) !important;

    border:
        1px solid
        var(--border-subtle) !important;
}


/* ---------- Expanders ---------- */

.streamlit-expanderHeader,
[data-testid="stExpander"] summary {

    background:
        var(--bg-card) !important;

    color:
        var(--text-primary) !important;

    border-radius:
        10px !important;

    border:
        1px solid
        var(--border-subtle) !important;

    font-weight:
        500;
}


[data-testid="stExpander"] {

    background:
        transparent !important;

    border:
        none !important;
}


[data-testid="stExpanderDetails"] {

    background:
        var(--bg-secondary) !important;

    color:
        var(--text-primary) !important;

    border-radius:
        0 0 10px 10px;

    border:
        1px solid
        var(--border-subtle);

    border-top:
        none;
}


/* ---------- Answer box ---------- */

.answer-box {

    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.14),
            rgba(139,92,246,0.06)
        );

    border:
        1px solid
        rgba(99,102,241,0.35);

    border-radius:
        14px;

    padding:
        1.3rem 1.4rem;

    color:
        var(--text-primary) !important;

    font-size:
        1.02rem;

    line-height:
        1.55;

    margin-bottom:
        1rem;
}


/* ---------- Score pills ---------- */

.score-row {

    display:
        flex;

    gap:
        0.5rem;

    margin-bottom:
        0.6rem;

    flex-wrap:
        wrap;
}


.score-pill {

    font-family:
        'JetBrains Mono',
        monospace;

    font-size:
        0.75rem;

    padding:
        0.22rem 0.6rem;

    border-radius:
        999px;

    border:
        1px solid
        var(--border-subtle);

    color:
        var(--text-secondary) !important;

    background:
        var(--bg-secondary);
}


.score-pill.hybrid {

    color:
        #c7d2fe !important;

    border-color:
        var(--accent-1);
}


/* ---------- Alerts ---------- */

[data-testid="stAlert"] {

    border-radius:
        12px !important;

    border:
        1px solid
        var(--border-subtle) !important;
}


/* ---------- Misc ---------- */

p,
span,
label,
.stMarkdown,
.stCaption {

    color:
        var(--text-primary);
}


.stCaption,
[data-testid="stCaptionContainer"] {

    color:
        var(--text-secondary) !important;
}


hr {

    border-color:
        var(--border-subtle) !important;
}


/* ---------- Sidebar ---------- */

[data-testid="stSidebar"] {

    background:
        var(--bg-secondary);

    border-right:
        1px solid
        var(--border-subtle);
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Hero Header
# ---------------------------------------------------------

st.markdown("""
<div class="hero-wrap">

    <p class="hero-title">
        📄 AI Document Assistant
    </p>

    <p class="hero-sub">
        Upload files or pull them from Google Drive,
        then ask questions answered with hybrid semantic
        + keyword search over your own documents.
    </p>

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------

if "files" not in st.session_state:
    st.session_state.files = {}


if "collection_key" not in st.session_state:
    st.session_state.collection_key = None


if "chunks" not in st.session_state:
    st.session_state.chunks = []


if "index" not in st.session_state:
    st.session_state.index = None


if "embeddings" not in st.session_state:
    st.session_state.embeddings = np.empty(
        (0, 384),
        dtype="float32"
    )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div class="section-label">
            <span class="section-badge">1</span>
            Local documents
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            data = uploaded_file.getvalue()

            key = file_fingerprint(
                uploaded_file.name,
                data
            )

            st.session_state.files[key] = {
                "filename": uploaded_file.name,
                "bytes": data
            }


    st.markdown(
        """
        <div class="section-label">
            <span class="section-badge">2</span>
            Google Drive
        </div>
        """,
        unsafe_allow_html=True
    )


    drive_url = st.text_input(
        "Google Drive link",
        placeholder="Paste a public/shared file or folder link",
        label_visibility="collapsed"
    )


    if st.button(
        "⬇ Load from Google Drive",
        use_container_width=True
    ):

        if not drive_url.strip():

            st.warning(
                "Please paste a Google Drive link first."
            )

        else:

            with st.spinner(
                "Loading Google Drive files..."
            ):

                try:

                    drive_files = load_from_drive(
                        drive_url.strip()
                    )

                    if not drive_files:

                        st.warning(
                            "No supported PDF, DOCX, TXT "
                            "or MD files were found."
                        )

                    else:

                        added = 0

                        for path in drive_files:

                            data = path.read_bytes()

                            key = file_fingerprint(
                                path.name,
                                data
                            )

                            st.session_state.files[key] = {
                                "filename": path.name,
                                "bytes": data
                            }

                            added += 1

                        st.success(
                            f"Loaded {added} supported file(s)."
                        )

                except Exception as error:

                    st.error(
                        f"Could not load the Google Drive link: {error}"
                    )


    # -----------------------------------------------------
    # Clear documents
    # -----------------------------------------------------

    if st.session_state.files:

        st.markdown("---")

        if st.button(
            "🗑 Clear all documents",
            use_container_width=True
        ):

            st.session_state.files = {}

            st.session_state.collection_key = None

            st.session_state.chunks = []

            st.session_state.index = None

            st.session_state.embeddings = np.empty(
                (0, 384),
                dtype="float32"
            )

            st.rerun()


# ---------------------------------------------------------
# Process Documents
# ---------------------------------------------------------

if st.session_state.files:

    current_key = tuple(
        sorted(
            st.session_state.files.keys()
        )
    )


    if (
        current_key
        != st.session_state.collection_key
    ):

        with st.spinner(
            "Extracting text, creating chunks and embeddings..."
        ):

            file_list = [
                (
                    item["filename"],
                    item["bytes"]
                )

                for item
                in st.session_state.files.values()
            ]


            chunks, index, embeddings = build_collection(
                file_list
            )


            st.session_state.chunks = chunks

            st.session_state.index = index

            st.session_state.embeddings = embeddings

            st.session_state.collection_key = current_key


    # -----------------------------------------------------
    # Document Overview
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="section-label">
            <span class="section-badge">📊</span>
            Document overview
        </div>
        """,
        unsafe_allow_html=True
    )


    info_columns = st.columns(4)


    info_columns[0].metric(
        "Documents",
        len(st.session_state.files)
    )


    info_columns[1].metric(
        "Chunks",
        len(st.session_state.chunks)
    )


    info_columns[2].metric(
        "Embedding Size",
        (
            st.session_state.embeddings.shape[1]
            if st.session_state.embeddings.size
            else 0
        )
    )


    info_columns[3].metric(
        "Search Ready",
        (
            "Yes"
            if st.session_state.index is not None
            else "No"
        )
    )


    # -----------------------------------------------------
    # Loaded Documents
    # -----------------------------------------------------

    with st.expander(
        "📁 View loaded documents"
    ):

        for item in st.session_state.files.values():

            extracted_pages = extract_document(
                item["filename"],
                item["bytes"]
            )


            character_count = sum(
                len(page["text"])
                for page in extracted_pages
            )


            st.markdown(
                f"**{item['filename']}**"
            )


            st.caption(
                f"Characters extracted: {character_count}"
            )


            pdf_pages = [
                page["page"]
                for page in extracted_pages
                if page["page"] is not None
            ]


            if pdf_pages:

                st.caption(
                    f"Pages: {len(pdf_pages)}"
                )

            else:

                st.caption(
                    "Pages: Not available"
                )


            st.markdown("---")


    # -----------------------------------------------------
    # Ask Question
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="section-label">
            <span class="section-badge">3</span>
            Ask a question
        </div>
        """,
        unsafe_allow_html=True
    )


    question_col, button_col = st.columns(
        [5, 1]
    )


    with question_col:

        question = st.text_input(
            "Question",
            placeholder="Ask something about your documents...",
            label_visibility="collapsed"
        )


    with button_col:

        ask_clicked = st.button(
            "✨ Ask AI",
            type="primary",
            use_container_width=True
        )


    # -----------------------------------------------------
    # Ask AI
    # -----------------------------------------------------

    if ask_clicked:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )


        elif st.session_state.index is None:

            st.warning(
                "No searchable document content is available."
            )


        else:

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                results = hybrid_search(
                    question,
                    st.session_state.chunks,
                    st.session_state.embeddings,
                    st.session_state.index,
                    top_k=5
                )


                answer = answer_question(
                    question,
                    results
                )


            st.markdown(
                "#### 💡 Answer"
            )


            st.markdown(
                f"""
                <div class="answer-box">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                "#### 🔍 Retrieved Sources"
            )


            for number, result in enumerate(
                results,
                start=1
            ):

                page_text = (
                    f"Page {result['page']}"
                    if result["page"] is not None
                    else "Page not available"
                )


                with st.expander(
                    f"{number}. "
                    f"{result['filename']} — "
                    f"{page_text}"
                ):

                    st.markdown(
                        f"""
                        <div class="score-row">

                            <span class="score-pill hybrid">
                                Hybrid
                                {result['hybrid_score']:.3f}
                            </span>

                            <span class="score-pill">
                                Semantic
                                {result['semantic_score']:.3f}
                            </span>

                            <span class="score-pill">
                                Keyword
                                {result['keyword_score']:.3f}
                            </span>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    st.write(
                        result["text"]
                    )


else:

    st.info(
        "👈 Upload at least one document or load a "
        "supported Google Drive file/folder from the "
        "sidebar to begin."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "PDF page numbers are preserved. DOCX/TXT/MD "
    "page numbers are shown as unavailable because "
    "those formats do not reliably provide page metadata."
)
