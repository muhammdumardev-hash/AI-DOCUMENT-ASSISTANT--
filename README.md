# AI Document Assistant

A simple Streamlit AI Document Assistant that lets you upload documents, search their content, and ask questions using Groq.

## Features

- Upload **PDF, DOCX, TXT and MD** files.
- Extract text with separate extraction functions.
- Preserve the filename for every extracted document section.
- Preserve **PDF page numbers** where available.
- Split documents into overlapping text chunks.
- Create Sentence Transformers embeddings for every chunk.
- Store chunk metadata together with the embeddings.
- Use **FAISS** for semantic vector search.
- Use simple keyword matching as a second search method.
- Combine semantic and keyword scores with hybrid search.
- Send only retrieved chunks to Groq.
- The AI is instructed to answer only from the provided context.
- Show retrieved sources after every answer.
- Load supported files from public/shared **Google Drive file or folder links**.
- Local uploads and Google Drive files use the same processing pipeline.
- Cache the embedding model and individual document processing so documents are not embedded again on every question.
- Read the Groq API key from Streamlit secrets.

## Project files

```text
AI Document Assistant/
│
├── app.py
├── requirements.txt
└── README.md
```

## 1. Install dependencies

Create a virtual environment if you want:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Then install the packages:

```bash
pip install -r requirements.txt
```

## 2. Add the Groq API key

Create this folder:

```text
.streamlit
```

Inside it create:

```text
secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Never put the real API key inside `app.py`.

## 3. Run the app

```bash
python -m streamlit run app.py
```

## How the pipeline works

The project follows this simple flow:

```text
Documents
   ↓
Text Extraction
   ↓
Text Chunking
   ↓
Sentence Transformer Embeddings
   ↓
FAISS Vector Index
   ↓
Hybrid Search
   ├── Semantic Search
   └── Keyword Search
   ↓
Top Relevant Chunks
   ↓
Groq
   ↓
Answer + Sources
```

## Document extraction

The app has four separate extraction functions:

- `extract_pdf()`
- `extract_docx()`
- `extract_txt()`
- `extract_md()`

PDF extraction keeps the page number.

DOCX, TXT and MD do not reliably expose page numbers through their normal text structure, so their page value is stored as `None`.

## Chunking

The extracted text is split into chunks of about 800 characters.

The chunks overlap by about 120 characters so information near a chunk boundary is less likely to be lost.

Every chunk keeps:

```text
filename
page
text
```

## Embeddings

The app uses:

```text
all-MiniLM-L6-v2
```

Each document chunk is converted into a numerical vector.

The embedding model is cached with Streamlit's `st.cache_resource`.

Individual document processing is cached with `st.cache_data`.

This means asking multiple questions does **not** recreate embeddings.

## FAISS search

The question is converted into an embedding and compared with the document chunk embeddings.

FAISS uses normalized vectors with inner-product similarity, which gives cosine-style semantic similarity.

## Keyword search

The question is also split into important words.

Common words such as:

```text
the
is
what
how
and
```

are ignored.

The remaining words are matched against each document chunk.

## Hybrid search

The final ranking uses:

```text
70% semantic similarity
30% keyword matching
```

This helps the application find both:

- conceptually similar chunks
- chunks containing important exact words

## Google Drive

Paste a public/shared Google Drive file or folder link.

Supported files:

- PDF
- DOCX
- TXT
- MD

The app uses `gdown` to download public/shared Drive files and folders.

For a Drive folder, the files are downloaded and then passed through the same extraction → chunking → embedding → FAISS pipeline as local uploads.

### Important Google Drive limitation

This simple version is intended for files/folders that `gdown` can access without Google account authentication.

It is not a full Google Drive API integration and does not implement OAuth login.

## Groq

The app sends:

```text
User question
+
Retrieved document chunks
```

to Groq.

The system prompt tells the model:

- use only the provided context
- do not use outside knowledge
- do not guess
- clearly say when the answer is not available

The current code uses:

```text
openai/gpt-oss-120b
```

You can change `GROQ_MODEL` in `app.py` if you want to use another Groq-supported model.

## Security

The Groq API key is loaded from:

```text
st.secrets["GROQ_API_KEY"]
```

Do not commit `.streamlit/secrets.toml` to GitHub.

Add this to `.gitignore`:

```text
.streamlit/secrets.toml
venv/
__pycache__/
```

## Simple explanation for a presentation

You can explain the project in six steps:

1. **Upload:** The user uploads a document or loads one from Google Drive.
2. **Extract:** The app extracts readable text from the document.
3. **Chunk:** Large text is divided into smaller overlapping pieces.
4. **Embed:** Sentence Transformers converts each chunk into a vector.
5. **Search:** FAISS performs semantic search while keyword search checks important words.
6. **Answer:** The best chunks are sent to Groq, which answers using only those chunks and displays the sources.

## Notes

The first run may take longer because the Sentence Transformers model needs to be downloaded.

After that, Streamlit caches the model and processed documents for the current application environment/session.

For very large document collections, a persistent vector database would be a better next step. This project intentionally keeps the architecture simple and uses FAISS in memory.
